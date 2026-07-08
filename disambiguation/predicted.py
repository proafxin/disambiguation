import pickle
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import optim
from tqdm import tqdm

from disambiguation.conll_scorer import conll_f1, mention_type, write_conll
from disambiguation.data import (
    BGE_MODEL,
    _apply_sentence_windows,
    _build_lexical,
    _data_cfg,
    _filter_docs,
    _word_to_subtok,
    build_docs,
    precompute_span_ctx,
)
from disambiguation.paths import DATA_DIR, MODELS_DIR
from disambiguation.stage2_context_encoder import (
    BACKBONE,
    AntecedentScorer,
    BIOTagger,
    ContextEncoder,
    load_tokenizer,
)

PRED_NOM_CACHE = DATA_DIR / "stage2_predicted_nominals.pkl"
PRED_CTX_BASE = DATA_DIR / "stage2_predicted_span_ctx"


def _sent_offsets(sentences: list) -> list:
    offs, off = [], 0
    for s in sentences:
        offs.append(off)
        off += len(s)
    return offs


def _sub2word(words: list, tokenizer) -> np.ndarray:
    # per-subtoken global word index (enc.word_ids), used to snap detector subtoken spans back to word
    # boundaries. Same tokenizer / add_special_tokens=False as the gold pipeline, so coordinates match
    # d["content_ids"] / d["span_sub"] exactly. No special tokens, so every position maps to a word.
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    return np.asarray([w if w is not None else -1 for w in enc.word_ids()], dtype=np.int64)


def _word_to_sent(gw: int, sent_offsets: list) -> tuple:
    si = 0
    for i, off in enumerate(sent_offsets):
        if gw >= off:
            si = i
        else:
            break
    return si, gw - sent_offsets[si]


def predicted_word_spans(pred_sub: list, sentences: list, tokenizer) -> list:
    # snap detector subtoken spans (start, end INCLUSIVE) to word spans (sent_idx, start, end EXCLUSIVE),
    # kept within a single sentence (Stage A spans are sentence-local). A span that crosses a sentence
    # boundary is clamped to its start sentence.
    words = [w for s in sentences for w in s]
    sent_offsets = _sent_offsets(sentences)
    sub2word = _sub2word(words, tokenizer)
    nsub = len(sub2word)
    out = []
    for ss, se in pred_sub:
        ss = max(0, min(int(ss), nsub - 1))
        se = max(ss, min(int(se), nsub - 1))
        gw0, gw1 = int(sub2word[ss]), int(sub2word[se])
        if gw0 < 0:
            continue
        if gw1 < gw0:
            gw1 = gw0
        si, a = _word_to_sent(gw0, sent_offsets)
        se_si, se_local = _word_to_sent(gw1, sent_offsets)
        b = (se_local + 1) if se_si == si else len(sentences[si])
        b = max(a + 1, min(b, len(sentences[si])))
        out.append((si, a, b))
    return out


def _sent_sub(content_ids: np.ndarray, w2s: dict, sentences: list) -> tuple:
    # per-sentence (subtoken offset, length) — mirrors data._raw_from_doc so windowing matches the gold path
    offsets = _sent_offsets(sentences)
    last = len(content_ids) - 1
    so, sl = [], []
    for si, sent in enumerate(sentences):
        gw_start, gw_end = offsets[si], offsets[si] + len(sent) - 1
        ss = w2s.get(gw_start, min(gw_start, last))
        se = w2s.get(gw_end + 1, last + 1)
        so.append(ss)
        sl.append(max(1, se - ss))
    return offsets, so, sl


def rebuild_mention_fields(word_spans: list, sentences: list, tokenizer) -> dict:
    # From word spans (sent, a, b exclusive) rebuild the exact per-mention fields Stage A consumes,
    # mirroring data._raw_from_doc: subtoken spans (inclusive), per-mention sentence sub offset/length
    # (for sentence-aligned windowing), and surfaces. Sorted by subtoken position (Stage A's invariant).
    words = [w for s in sentences for w in s]
    content_ids, w2s = _word_to_subtok(words, tokenizer)
    last = len(content_ids) - 1
    offsets, sent_so, sent_sl = _sent_sub(content_ids, w2s, sentences)
    span_sub, m_off, m_len, surf = [], [], [], []
    for si, a, b in word_spans:
        gw_start, gw_end = offsets[si] + a, offsets[si] + b - 1
        ss = w2s.get(gw_start, min(gw_start, last))
        se = min(max(w2s.get(gw_end + 1, len(content_ids)) - 1, ss), last)
        span_sub.append((ss, se))
        m_off.append(sent_so[si])
        m_len.append(sent_sl[si])
        surf.append(" ".join(sentences[si][a:b]))
    order = sorted(range(len(word_spans)), key=lambda k: span_sub[k])
    return {
        "content_ids": content_ids,
        "order": order,
        "spans": [word_spans[k] for k in order],
        "span_sub": np.asarray([span_sub[k] for k in order], dtype=np.int64),
        "sent_sub_offsets": np.asarray([m_off[k] for k in order], dtype=np.int64),
        "sent_sub_lengths": np.asarray([m_len[k] for k in order], dtype=np.int64),
        "surfaces": [surf[k] for k in order],
    }


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    # token overlap of two word ranges [a0,a1) and [b0,b1)
    return max(0, min(a1, b1) - max(a0, b0))


def align_predicted_to_gold(pred_spans: list, gold_spans: list, gold_cluster_id) -> np.ndarray:
    # cluster_id label per predicted span: exact gold-span match -> that gold cluster; else the gold
    # mention it overlaps most (same sentence, >0 overlap) -> that cluster; else a fresh singleton id.
    # Singleton ids start above the max gold id so they never collide with a real cluster.
    gold_by_key = {tuple(g): int(c) for g, c in zip(gold_spans, gold_cluster_id, strict=True)}
    next_singleton = (int(max(gold_cluster_id)) + 1) if len(gold_cluster_id) else 0
    out = []
    for ps in pred_spans:
        if tuple(ps) in gold_by_key:
            out.append(gold_by_key[tuple(ps)])
            continue
        si, a, b = ps
        best, best_ov = None, 0
        for (gsi, ga, gb), c in zip(gold_spans, gold_cluster_id, strict=True):
            if gsi != si:
                continue
            ov = _overlap(a, b, ga, gb)
            if ov > best_ov:
                best, best_ov = int(c), ov
        out.append(best if best is not None else next_singleton)
        if best is None:
            next_singleton += 1
    return np.asarray(out, dtype=np.int64)


# ── Bridge: build predicted-mention docs and run the gold-trained Stage A on them ───────────────


def _dedupe_doc_mentions(d: dict) -> None:
    # drop exact-duplicate predicted spans (keep first), applying the same keep-mask to every
    # positionally-coupled per-mention array. Duplicates are identical (same word span -> same span_sub
    # -> same ctx/bge), so first-wins is lossless; differing singleton cluster ids don't matter (dropped).
    spans = d["spans"]
    seen, keep = set(), []
    for i, s in enumerate(spans):
        if tuple(s) not in seen:
            seen.add(tuple(s))
            keep.append(i)
    if len(keep) == len(spans):
        return
    idx = np.asarray(keep, dtype=np.int64)
    d["spans"] = [spans[i] for i in keep]
    for k in ("cluster_id", "mention_bge", "win_ids", "tok_pos", "ctx_vecs"):
        v = d.get(k)
        if v is not None and len(v) == len(spans):
            d[k] = v[idx]
    mt = d.get("mention_tokens")
    if mt is not None and len(mt) == len(spans):
        d["mention_tokens"] = [mt[i] for i in keep]


def build_predicted_docs(
    det_ckpt: str,
    thr: float = 0.2,
    det_window: int = 510,
    res_window: int = 256,
    n_layers: int = 3,
    win_bs: int = 2,
    detector: str = "bio",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list:
    # End-to-end mention bridge: run the detector over the CoNLL docs, snap to word spans, rebuild
    # the exact Stage-A per-mention fields, BGE-encode the surfaces, gather frozen-RoBERTa ctx at the
    # predicted spans, align predicted->gold for cluster labels. Gold spans/clusters are preserved
    # (d["gold_*"]) for the scoring key. detector="bio"|"span"; each gets its own tagged cache so they
    # never clobber each other, and the gold caches are never touched.
    dtag = "" if detector == "bio" else f"_{detector}"
    nom_path = PRED_NOM_CACHE.with_name(PRED_NOM_CACHE.stem + dtag + ".pkl")
    ctx_base = PRED_CTX_BASE.with_name(PRED_CTX_BASE.name + dtag)
    if nom_path.exists():
        with nom_path.open("rb") as f:
            docs = pickle.load(f)
        precompute_span_ctx(None, docs, 0, 0, device, cache_dir=ctx_base, window=res_window, single=True)
        tokenizer = load_tokenizer()
        for d in docs:
            _dedupe_doc_mentions(d)
            if "gold_win_ids" not in d:  # older cache: derive the gold-mention window ids for the key
                rb = rebuild_mention_fields(list(d["gold_spans"]), d["sentences"], tokenizer)
                starts = np.asarray([s for s, _ in d["win_chunks"]], dtype=np.int64)
                wid = np.searchsorted(starts, rb["span_sub"][:, 0], side="right") - 1
                sp2win = {tuple(s): int(w) for s, w in zip(rb["spans"], wid, strict=True)}
                d["gold_win_ids"] = np.asarray([sp2win[tuple(s)] for s in d["gold_spans"]], dtype=np.int64)
        _build_lexical(docs)
        print(f"Loaded cached predicted docs ({detector}): {len(docs)}")
        return docs

    nom_cache, datasets, preco_n, _ = _data_cfg("conll")
    docs = _filter_docs(build_docs(device=device, datasets=datasets, nom_cache=nom_cache, preco_n=preco_n), "conll")
    tokenizer = load_tokenizer()
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    for d in docs:
        d["gold_spans"] = [tuple(s) for s in d["spans"]]
        d["gold_cluster_id"] = np.asarray(d["cluster_id"], dtype=np.int64)
        d["gold_span_sub"] = np.asarray(d["span_sub"], dtype=np.int64)
        d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)

    # --- detection (det_window) ---
    _apply_sentence_windows(docs, det_window)
    ck = torch.load(det_ckpt, map_location=device)
    if detector == "span":
        from disambiguation.stage2_context_encoder import SpanDetector
        from disambiguation.stage_a import _span_predict_doc

        model = SpanDetector(model_name=BACKBONE, max_span=ck.get("max_span", 30)).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        for d in tqdm(docs, desc="detect(span)"):
            d["pred_sub"] = sorted(_span_predict_doc(model, d, tokenizer, device, det_window, thr))
    else:
        from disambiguation.stage_a import _bio_predict_doc

        model = BIOTagger(n_layers=n_layers, model_name=BACKBONE).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        for d in tqdm(docs, desc="detect(bio)"):
            d["pred_sub"] = sorted(_bio_predict_doc(model, d, tokenizer, device, det_window, win_bs, n_layers, thr))
    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    # --- rebuild predicted mention set ---
    for d in docs:
        wspans = predicted_word_spans(d["pred_sub"], d["sentences"], tokenizer)
        wspans = list(dict.fromkeys(wspans))  # distinct subtoken spans can snap to one word span -> dedupe
        rb = rebuild_mention_fields(wspans, d["sentences"], tokenizer)
        d["spans"] = rb["spans"]
        d["span_sub"] = rb["span_sub"]
        d["sent_sub_offsets"] = rb["sent_sub_offsets"]
        d["sent_sub_lengths"] = rb["sent_sub_lengths"]
        d["content_ids"] = rb["content_ids"]
        d["cluster_id"] = align_predicted_to_gold(rb["spans"], d["gold_spans"], d["gold_cluster_id"])
        d["tok_pos"] = np.asarray([s for s, _ in rb["span_sub"]], dtype=np.int64)
        d["_surfaces"] = rb["surfaces"]

    # --- BGE encode predicted surfaces (unique across corpus) ---
    vocab = sorted({s for d in docs for s in d["_surfaces"]})
    bge = SentenceTransformer(BGE_MODEL, device=device)
    embs = np.asarray(
        bge.encode(vocab, normalize_embeddings=True, batch_size=512, show_progress_bar=True), dtype=np.float32
    )
    s2e = {s: embs[i] for i, s in enumerate(vocab)}
    bge_dim = embs.shape[1]
    del bge
    if device != "cpu":
        torch.cuda.empty_cache()
    for d in docs:
        d["mention_bge"] = (
            np.stack([s2e[s] for s in d["_surfaces"]]).astype(np.float32)
            if d["_surfaces"]
            else np.zeros((0, bge_dim), np.float32)
        )

    # --- frozen ctx at res_window (matches the gold-trained Stage A) ---
    _apply_sentence_windows(docs, res_window)
    encoder = ContextEncoder().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    precompute_span_ctx(encoder, docs, cls_id, sep_id, device, cache_dir=ctx_base, window=res_window, single=True)
    del encoder
    if device != "cpu":
        torch.cuda.empty_cache()
    _build_lexical(docs)

    # gold-mention window ids under the SAME sentence-aligned chunks → for the window-split gold key
    # (Stage A's proper eval credits only intra-window coreference, comparable to 89.48).
    for d in docs:
        starts = np.asarray([s for s, _ in d["win_chunks"]], dtype=np.int64)
        gss = d["gold_span_sub"][:, 0]
        d["gold_win_ids"] = (np.searchsorted(starts, gss, side="right") - 1) if len(gss) else np.array([], np.int64)

    keep = ("name", "split", "sentences", "spans", "cluster_id", "mention_bge", "win_ids", "win_chunks",
            "gold_spans", "gold_cluster_id", "gold_win_ids", "tok_pos", "mention_tokens", "mention_idf")
    slim = [{k: d[k] for k in keep if k in d} for d in docs]
    with nom_path.open("wb") as f:
        pickle.dump(slim, f)
    print(f"Cached {len(docs)} predicted docs ({detector}) to {nom_path.name}")
    return docs


def load_gold_stage_a(window: int, subset: str, channel: str, sent_aligned: bool, device: str) -> AntecedentScorer:
    from disambiguation.data import _win_names

    head_path = MODELS_DIR / f"{_win_names(window, subset, channel, sent_aligned=sent_aligned)[1]}.pt"
    scorer = AntecedentScorer(hidden=1024, use_distance=True, channel=channel).to(device)
    scorer.load_state_dict(torch.load(head_path, map_location=device)["scorer"])
    scorer.eval()
    return scorer


def _window_split_gold_key(d: dict) -> list:
    # gold clusters split at the SAME sentence-aligned window boundaries — Stage A's proper key, crediting
    # only intra-window coreference (cross-window links are Stage B's job, not charged to Stage A).
    gs, gc, gw = d["gold_spans"], d["gold_cluster_id"], d["gold_win_ids"]
    out = []
    for c in np.unique(gc):
        members = np.where(gc == c)[0]
        for w in np.unique(gw[members]):
            comp = [int(i) for i in members if gw[i] == w]
            if len(comp) >= 2:
                out.append([gs[i] for i in comp])
    return out


def eval_stage_a_predicted(docs: list, scorer: AntecedentScorer, device: str, window: int = 256) -> dict:
    # Stage A INTRA-WINDOW resolution on predicted mentions: window-local union-find clusters scored
    # against WINDOW-SPLIT gold (comparable to the gold-mention 89.48), not full-doc gold. Size-1 dropped.
    from disambiguation.stage_a import _stage_a_clusters_for_doc

    key_docs, resp_docs = [], []
    for d in docs:
        key_docs.append((d["name"], d["sentences"], _window_split_gold_key(d)))
        if "ctx_vecs" not in d or len(d["spans"]) < 2:
            resp_docs.append((d["name"], d["sentences"], []))
            continue
        per_window = _stage_a_clusters_for_doc(d, scorer, device, window)
        resp = []
        for info in per_window.values():
            gidx = info["global_idx"]
            for cl in info["clusters"]:
                spans = [d["spans"][int(gidx[i])] for i in cl]
                if len(spans) >= 2:
                    resp.append(spans)
        resp_docs.append((d["name"], d["sentences"], resp))
    key_path, resp_path = MODELS_DIR / "stagea_pred_key.conll", MODELS_DIR / "stagea_pred_resp.conll"
    write_conll(key_path, [k for k in key_docs if k[2]])
    write_conll(resp_path, [r for k, r in zip(key_docs, resp_docs, strict=True) if k[2]])
    return conll_f1(key_path, resp_path)


def _gold_key_full(d: dict) -> list:
    # full-document gold clusters (no window split) — the key for the end-to-end A+B comparison (86.46)
    gs, gc = d["gold_spans"], d["gold_cluster_id"]
    return [[gs[i] for i in np.where(gc == c)[0]] for c in np.unique(gc)]


def load_gold_stage_b(device: str, subset: str = "all8k") -> tuple:
    from disambiguation.data import _win_names
    from disambiguation.stage2_context_encoder import ClusterGNN

    ckpt_b = _win_names(256, subset, "both", sent_aligned=True)[2].replace(".pt", "_gnn_lse_lex.pt")
    gnn = ClusterGNN(channel="both", member_pool="lse", use_lexical=True).to(device)
    ck = torch.load(MODELS_DIR / ckpt_b, map_location=device)
    gnn.load_state_dict(ck["cluster_matcher"])
    gnn.eval()
    return gnn, float(ck.get("nb_offset", 0.0))


def train_stage_b_predicted(
    thr: float = 0.2,
    stage_a_ckpt: str | None = None,
    max_epochs: int = 30,
    patience: int = 5,
    doc_bs: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    pos_weight: float = 1.0,
    neg_ratio: float = 5.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    # Step 2 (B): train the cluster GNN on the PREDICTED-trained Stage A clusters, end-to-end on
    # predicted mentions. Node gold labels come from d["cluster_id"] (aligned). Mirrors the gold Stage B
    # loop; selects on full-doc CoNLL F1 vs gold (gold-trained A+B on predicted = 86.66). Own checkpoint.
    from disambiguation.stage2_context_encoder import ClusterGNN
    from disambiguation.stage_a import _stage_a_clusters_for_doc
    from disambiguation.stage_b import _gnn_doc_loss

    docs = build_predicted_docs(str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt"), thr=thr, device=device)
    sa_ckpt = stage_a_ckpt or str(MODELS_DIR / "stage2_frozen_head_predicted_conll.pt")
    scorer = AntecedentScorer(hidden=1024, use_distance=True).to(device)
    scorer.load_state_dict(torch.load(sa_ckpt, map_location=device)["scorer"])
    scorer.eval()
    clusters = [
        _stage_a_clusters_for_doc(d, scorer, device, 256) if "ctx_vecs" in d and len(d["spans"]) >= 2 else {}
        for d in tqdm(docs, desc="stage A clusters")
    ]
    tr = [(d, c) for d, c in zip(docs, clusters, strict=True) if d["split"] == "train" and c]
    val_d = [d for d, c in zip(docs, clusters, strict=True) if d["split"] == "validation"]
    val_c = [c for d, c in zip(docs, clusters, strict=True) if d["split"] == "validation"]
    test_d = [d for d, c in zip(docs, clusters, strict=True) if d["split"] == "test"]
    test_c = [c for d, c in zip(docs, clusters, strict=True) if d["split"] == "test"]
    print(f"predicted Stage B: train {len(tr)} | val {len(val_d)} | test {len(test_d)} docs")
    gnn = ClusterGNN(channel="both", member_pool="lse", use_lexical=True).to(device)
    optimizer = optim.AdamW(gnn.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.1)
    ckpt = MODELS_DIR / "stage2_cluster_matcher_predicted_conll.pt"
    best, patience_ctr = -1.0, 0
    for epoch in range(max_epochs):
        random.shuffle(tr)
        gnn.train()
        pending, total, n = [], 0.0, 0
        for d, per_window in tqdm(tr, desc=f"train_b epoch {epoch + 1}"):
            loss = _gnn_doc_loss(gnn, d, per_window, device, pos_weight, neg_ratio)
            if loss is None:
                continue
            pending.append(loss)
            if len(pending) >= doc_bs:
                optimizer.zero_grad()
                torch.stack(pending).mean().backward()
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
                optimizer.step()
                total += float(sum(lo.item() for lo in pending))
                n += len(pending)
                pending = []
        if pending:
            optimizer.zero_grad()
            torch.stack(pending).mean().backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            optimizer.step()
            total += float(sum(lo.item() for lo in pending))
            n += len(pending)
        scheduler.step()
        gnn.eval()
        f1 = eval_stage_ab_predicted(val_d, val_c, gnn, device, nb_offset=0.0)["CoNLL"]
        print(f"epoch {epoch + 1}/{max_epochs} | train loss {total / max(n, 1):.4f} | val A+B CoNLL {f1 * 100:.2f}")
        if f1 > best + 1e-4:
            best, patience_ctr = f1, 0
            torch.save({"cluster_matcher": gnn.state_dict(), "best_val_f1": best}, ckpt)
            print(f"  ✓ saved ({f1 * 100:.2f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  ⊘ early stop. best val {best * 100:.2f}")
                break
    gnn.load_state_dict(torch.load(ckpt, map_location=device)["cluster_matcher"])
    gnn.eval()
    res = eval_stage_ab_predicted(test_d, test_c, gnn, device, nb_offset=0.0)
    print("\n=== Predicted-TRAINED A+B end-to-end test (vs gold-trained-on-predicted 86.66) ===")
    print(
        f"CoNLL {res['CoNLL'] * 100:.2f} | MUC {res['muc'] * 100:.2f} "
        f"B3 {res['bcub'] * 100:.2f} CEAFe {res['ceafe'] * 100:.2f}"
    )
    return str(ckpt)


def eval_stage_ab_predicted(docs: list, stage_a_clusters: list, gnn, device: str, nb_offset: float = 0.0) -> dict:
    # Full-document A+B on predicted mentions: gold GNN merges the predicted Stage A clusters across
    # windows, scored against FULL gold clusters (the 86.46 comparison). Docs with no predicted mentions
    # still contribute their gold key (response empty) so missed mentions are charged honestly.
    from disambiguation.stage_b import _predict_full_doc_clusters_gnn

    key_docs, resp_docs = [], []
    for d, per_window in zip(docs, stage_a_clusters, strict=True):
        key = _gold_key_full(d)
        if not key:
            continue
        resp = _predict_full_doc_clusters_gnn(d, per_window, gnn, device, nb_offset) if per_window else []
        key_docs.append((d["name"], d["sentences"], key))
        resp_docs.append((d["name"], d["sentences"], resp))
    key_path, resp_path = MODELS_DIR / "stageab_pred_key.conll", MODELS_DIR / "stageab_pred_resp.conll"
    write_conll(key_path, key_docs)
    write_conll(resp_path, resp_docs)
    return conll_f1(key_path, resp_path)


def run_stage_ab_predicted(
    thr: float = 0.2,
    nb_offset: float = 0.0,
    gold_subset: str = "all8k",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    # Step 1 baseline / control: gold-trained Stage A + Stage B reused on predicted mentions.
    # gold_subset="all8k" (the 86.46 system) or "conll" (the data-matched CoNLL-only control).
    from disambiguation.stage_a import _stage_a_clusters_for_doc

    det_ckpt = str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt")
    docs = build_predicted_docs(det_ckpt, thr=thr, device=device)
    scorer = load_gold_stage_a(256, gold_subset, "both", sent_aligned=True, device=device)
    clusters = [
        _stage_a_clusters_for_doc(d, scorer, device, 256) if "ctx_vecs" in d and len(d["spans"]) >= 2 else {}
        for d in tqdm(docs, desc="stage A clusters")
    ]
    gnn, _ = load_gold_stage_b(device, gold_subset)
    res = eval_stage_ab_predicted(docs, clusters, gnn, device, nb_offset=nb_offset)
    print(f"\n=== Stage A+B on PREDICTED mentions (gold_subset={gold_subset}, thr={thr}, nb_offset={nb_offset:+.2f}) ===")
    print(
        f"CoNLL {res['CoNLL'] * 100:.2f} | MUC {res['muc'] * 100:.2f} "
        f"B3 {res['bcub'] * 100:.2f} CEAFe {res['ceafe'] * 100:.2f}"
    )
    return res


def train_stage_a_predicted(
    thr: float = 0.2,
    null_weight: float = 0.5,
    init_ckpt: str | None = None,
    max_epochs: int = 60,
    patience: int = 8,
    doc_bs: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    # Step 2: train Stage A on PREDICTED CoNLL mentions (cluster_id = matched->gold entity, spurious->
    # singleton->null). Reuses run_epoch/stage_a_batched_loss; selects on intra-window CoNLL F1 vs
    # window-split gold (the gold-trained baseline on predicted = 90.20). Own checkpoint; gold untouched.
    # init_ckpt warm-starts from the gold Stage A head — fine-tune (low lr) from its clean null
    # calibration instead of from scratch (from-scratch plateaus ~76 under the over-generation null-swamp).
    from disambiguation.stage_a import run_epoch

    docs = build_predicted_docs(str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt"), thr=thr, device=device)
    train = [d for d in docs if d["split"] == "train"]
    val = [d for d in docs if d["split"] == "validation"]
    test = [d for d in docs if d["split"] == "test"]
    print(f"predicted Stage A: train {len(train)} | val {len(val)} | test {len(test)} docs"
          f"{' | warm-start ' + init_ckpt.split('/')[-1] if init_ckpt else ' | from scratch'}")
    scorer = AntecedentScorer(hidden=1024, use_distance=True).to(device)
    if init_ckpt is not None:
        scorer.load_state_dict(torch.load(init_ckpt, map_location=device)["scorer"])
    optimizer = optim.AdamW(scorer.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.1)
    ckpt = MODELS_DIR / "stage2_frozen_head_predicted_conll.pt"
    best, patience_ctr = -1.0, 0
    print(f"null_weight={null_weight} (down-weights singleton/null targets to fight over-generation swamp)")
    for epoch in range(max_epochs):
        random.shuffle(train)
        scorer.train()
        tr = run_epoch(scorer, train, optimizer, device, doc_bs, window=256, null_weight=null_weight)
        scheduler.step()
        scorer.eval()
        f1 = eval_stage_a_predicted(val, scorer, device, window=256)["CoNLL"]
        print(f"epoch {epoch + 1}/{max_epochs} | train loss {tr:.4f} | val Stage-A CoNLL {f1 * 100:.2f}")
        if f1 > best + 1e-4:
            best, patience_ctr = f1, 0
            torch.save({"scorer": scorer.state_dict(), "best_val_f1": best}, ckpt)
            print(f"  ✓ saved ({f1 * 100:.2f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  ⊘ early stop. best val {best * 100:.2f}")
                break
    scorer.load_state_dict(torch.load(ckpt, map_location=device)["scorer"])
    scorer.eval()
    f1 = eval_stage_a_predicted(test, scorer, device, window=256)["CoNLL"]
    print("\n=== Predicted-TRAINED Stage A test (intra-window vs window-split gold; gold-trained=90.20) ===")
    print(f"Stage A CoNLL {f1 * 100:.2f}")
    return str(ckpt)


def diagnose_boundaries(thr: float = 0.2, detector: str = "bio", det_ckpt: str | None = None,
                        device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    # Characterize the boundary errors (the -6.6 lever): for each predicted mention that overlaps a gold
    # mention but isn't exact, record left/right offset (pred-gold), which side is wrong, by type/length.
    from collections import Counter

    det_ckpt = det_ckpt or str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt")
    docs = build_predicted_docs(det_ckpt, thr=thr, detector=detector, device=device)
    test = [d for d in docs if d["split"] == "test"]
    left, right, side = Counter(), Counter(), Counter()
    by_type, by_len = {}, {}
    n_overlap = n_exact = n_off = 0
    for d in test:
        gbs: dict = {}
        for g in d["gold_spans"]:
            gbs.setdefault(g[0], []).append(g)
        for psi, pa, pb in d["spans"]:
            best, bo = None, 0
            for gsi, ga, gb in gbs.get(psi, []):
                ov = max(0, min(pb, gb) - max(pa, ga))
                if ov > bo:
                    bo, best = ov, (gsi, ga, gb)
            if best is None:
                continue
            n_overlap += 1
            gsi, ga, gb = best
            words = d["sentences"][gsi][ga:gb]
            t, gl = (mention_type(words) if words else "?"), min(gb - ga, 6)
            by_type.setdefault(t, [0, 0])[1] += 1
            by_len.setdefault(gl, [0, 0])[1] += 1
            if (pa, pb) == (ga, gb):
                n_exact += 1
                by_type[t][0] += 1
                by_len[gl][0] += 1
            else:
                n_off += 1
                lo, ro = pa - ga, pb - gb
                left[max(-3, min(3, lo))] += 1
                right[max(-3, min(3, ro))] += 1
                side["both" if lo and ro else ("left-only" if lo else "right-only")] += 1
    print(f"\n=== Boundary diagnosis (predicted overlapping a gold mention, TEST) ===")
    print(f"overlapping {n_overlap} | exact {n_exact} ({100 * n_exact / max(n_overlap, 1):.1f}%) | boundary-off {n_off}")
    print(f"which side is wrong: {dict(side)}")
    print(f"left offset (pred_start - gold_start): {dict(sorted(left.items()))}")
    print(f"right offset (pred_end - gold_end): {dict(sorted(right.items()))}")
    print("exact-rate by type: " + " ".join(f"{t} {100 * c[0] / max(c[1], 1):.0f}%({c[1]})" for t, c in by_type.items()))
    print("exact-rate by gold word-len: " + " ".join(
        f"{k}{'+ ' if k == 6 else ' '}{100 * c[0] / max(c[1], 1):.0f}%({c[1]})" for k, c in sorted(by_len.items())))
    return {"left": dict(left), "right": dict(right), "side": dict(side)}


def gold_detected_ablation(thr: float = 0.2, detector: str = "bio", det_ckpt: str | None = None,
                           device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    # Signal-vs-detection test: run the gold pipeline on GOLD mentions (correct boundaries, gold
    # features) but restricted to the subset the detector actually FOUND (drop the ~5% missed). Same
    # BGE+RoBERTa signal read at correct positions. Compared to all-gold (ceiling) and to matched-only.
    from disambiguation.data import (
        _apply_sentence_windows, _build_lexical, _data_cfg, _filter_docs, _win_names, build_docs, precompute_span_ctx,
    )
    from disambiguation.stage2_context_encoder import ContextEncoder, load_tokenizer
    from disambiguation.stage_a import _stage_a_clusters_for_doc

    det_ckpt = det_ckpt or str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt")
    pdocs = build_predicted_docs(det_ckpt, thr=thr, detector=detector, device=device)
    pred_by_name = {d["name"]: list(d["spans"]) for d in pdocs}

    nom_cache, datasets, preco_n, _ = _data_cfg("all8k")
    gdocs = _filter_docs(build_docs(device=device, datasets=datasets, nom_cache=nom_cache, preco_n=preco_n), "conll")
    for d in gdocs:
        d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    _apply_sentence_windows(gdocs, 256)
    tok = load_tokenizer()
    enc = ContextEncoder().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    precompute_span_ctx(enc, gdocs, tok.cls_token_id, tok.sep_token_id, device,
                        cache_dir=_win_names(256, "all8k", "both", sent_aligned=True)[0], window=256)
    del enc
    _build_lexical(gdocs)

    scorer = load_gold_stage_a(256, "all8k", "both", sent_aligned=True, device=device)
    gnn, _ = load_gold_stage_b(device, "all8k")
    test = [d for d in gdocs if d["split"] == "test" and "ctx_vecs" in d]
    out = {}

    def run(label, detect_only):
        sub = []
        n = 0
        for d in test:
            pred = pred_by_name.get(d["name"])
            keep = (
                [i for i, (si, a, b) in enumerate(d["spans"]) if pred is not None
                 and any(psi == si and pa < b and a < pb for psi, pa, pb in pred)]
                if detect_only else list(range(len(d["spans"])))
            )
            if len(keep) < 2:
                continue
            idx = np.asarray(keep, dtype=np.int64)
            nd = dict(d)
            nd["gold_spans"] = [tuple(s) for s in d["spans"]]
            nd["gold_cluster_id"] = np.asarray(d["cluster_id"], dtype=np.int64)
            nd["spans"] = [d["spans"][i] for i in keep]
            nd["cluster_id"] = d["cluster_id"][idx]
            for k in ("mention_bge", "win_ids", "tok_pos", "ctx_vecs"):
                nd[k] = d[k][idx]
            nd["mention_tokens"] = [d["mention_tokens"][i] for i in keep]
            sub.append(nd)
            n += len(keep)
        clusters = [_stage_a_clusters_for_doc(d, scorer, device, 256) for d in sub]
        r = eval_stage_ab_predicted(sub, clusters, gnn, device, 0.0)
        out[label] = r
        print(f"{label:30s} A+B {r['CoNLL'] * 100:.2f} (MUC {r['muc'] * 100:.2f} B3 {r['bcub'] * 100:.2f} "
              f"CEAFe {r['ceafe'] * 100:.2f}) | {n} mentions")

    print("\n=== Signal vs detection (gold features, TEST only) ===")
    run("all gold (ceiling)", False)
    run("gold-DETECTED (drop missed)", True)
    return out


def _filter_doc_mentions(d: dict, keep: list) -> dict:
    # shallow copy of d with every per-mention array restricted to `keep` indices (for ablations)
    idx = np.asarray(keep, dtype=np.int64)
    nd = dict(d)
    nd["spans"] = [d["spans"][i] for i in keep]
    for k in ("cluster_id", "mention_bge", "win_ids", "tok_pos", "ctx_vecs"):
        if k in d and d[k] is not None and len(d[k]) == len(d["spans_orig_len"]):
            nd[k] = d[k][idx]
    if "mention_tokens" in d:
        nd["mention_tokens"] = [d["mention_tokens"][i] for i in keep]
    return nd


def decompose_mention_cost(thr: float = 0.2, detector: str = "bio", det_ckpt: str | None = None,
                           device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    # Test-only ablation through the GOLD all8k pipeline: all predicted mentions vs matched-only
    # (drop spurious = predicted mentions that didn't align to a gold entity). Isolates the spurious cost.
    from disambiguation.stage_a import _stage_a_clusters_for_doc

    det_ckpt = det_ckpt or str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt")
    docs = build_predicted_docs(det_ckpt, thr=thr, detector=detector, device=device)
    scorer = load_gold_stage_a(256, "all8k", "both", sent_aligned=True, device=device)
    gnn, _ = load_gold_stage_b(device, "all8k")
    test = [d for d in docs if d["split"] == "test" and "ctx_vecs" in d and len(d["spans"]) >= 2]
    out = {}

    def run(label, keep_fn):
        sub = []
        n_ment = 0
        for d in test:
            d = dict(d, spans_orig_len=d["spans"])
            keep = list(range(len(d["spans"]))) if keep_fn is None else keep_fn(d)
            nd = _filter_doc_mentions(d, keep) if keep_fn is not None else d
            if len(nd["spans"]) < 2:
                continue
            sub.append(nd)
            n_ment += len(nd["spans"])
        clusters = [_stage_a_clusters_for_doc(d, scorer, device, 256) for d in sub]
        r = eval_stage_ab_predicted(sub, clusters, gnn, device, 0.0)
        out[label] = r
        print(f"{label:26s} A+B {r['CoNLL'] * 100:.2f} (MUC {r['muc'] * 100:.2f} B3 {r['bcub'] * 100:.2f} "
              f"CEAFe {r['ceafe'] * 100:.2f}) | {n_ment} mentions")

    def matched(d):
        gset = set(int(c) for c in d["gold_cluster_id"])
        return [i for i, c in enumerate(d["cluster_id"]) if int(c) in gset]

    print("\n=== Mention-cost decomposition (gold all8k pipeline, TEST only) ===")
    run("all predicted", None)
    run("matched-only (no spurious)", matched)
    return out


def diagnose_merge_behavior(thr: float = 0.2, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    # Same gold-CoNLL GNN on predicted-A vs gold-CoNLL-A clusters: merges accepted vs gold-positive
    # node pairs (under/over-merge) + the MUC/B3/CEAFe split. Under-merge (few accepted, low CEAFe/B3)
    # = nodes don't match across windows; over-merge (many accepted, low CEAFe, high MUC) = wrong merges.
    from disambiguation.stage_a import _stage_a_clusters_for_doc
    from disambiguation.stage_b import _gnn_merge_stats, _predict_full_doc_clusters_gnn

    docs = build_predicted_docs(str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt"), thr=thr, device=device)
    test = [d for d in docs if d["split"] == "test" and "ctx_vecs" in d and len(d["spans"]) >= 2]
    gnn, _ = load_gold_stage_b(device, "conll")
    out = {}
    for tag, ckpt in (
        ("predicted-A", "stage2_frozen_head_predicted_conll.pt"),
        ("gold-CoNLL-A", "stage2_frozen_head_k256_conllonly_sent.pt"),
    ):
        s = AntecedentScorer(hidden=1024, use_distance=True).to(device)
        s.load_state_dict(torch.load(MODELS_DIR / ckpt, map_location=device)["scorer"])
        s.eval()
        merges = gpos = nodes = 0
        key_docs, resp_docs = [], []
        for d in test:
            pw = _stage_a_clusters_for_doc(d, s, device, 256)
            m, p, n = _gnn_merge_stats(d, pw, gnn, device)
            merges, gpos, nodes = merges + m, gpos + p, nodes + n
            key_docs.append((d["name"], d["sentences"], _gold_key_full(d)))
            resp_docs.append((d["name"], d["sentences"], _predict_full_doc_clusters_gnn(d, pw, gnn, device, 0.0)))
        kp, rp = MODELS_DIR / "mb_key.conll", MODELS_DIR / "mb_resp.conll"
        write_conll(kp, key_docs)
        write_conll(rp, resp_docs)
        f = conll_f1(kp, rp)
        out[tag] = {"merges": merges, "gold_pos": gpos, "nodes": nodes, "f": f}
        print(f"\n{tag}: merges accepted {merges} / gold-positive node-pairs {gpos} / total nodes {nodes}")
        print(f"  A+B CoNLL {f['CoNLL'] * 100:.2f} (MUC {f['muc'] * 100:.2f} B3 {f['bcub'] * 100:.2f} "
              f"CEAFe {f['ceafe'] * 100:.2f})")
    return out


def diagnose_stage_a_clusters(thr: float = 0.2, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    # What do the Stage A CLUSTERS look like (the GNN's input)? Over-merge = a window-local cluster
    # mixing mentions from >=2 gold entities — the GNN can never split it, so it's a permanent error
    # invisible to per-link recall. no-merge F1 (full gold key, no GNN, no window-split) is the clean
    # Stage A cluster-quality number. Compared predicted-trained vs gold-CoNLL Stage A on predicted mentions.
    from disambiguation.stage_a import _stage_a_clusters_for_doc

    docs = build_predicted_docs(str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt"), thr=thr, device=device)
    test = [d for d in docs if d["split"] == "test" and "ctx_vecs" in d and len(d["spans"]) >= 2]
    out = {}
    for tag, ckpt in (
        ("predicted-trained", "stage2_frozen_head_predicted_conll.pt"),
        ("gold-CoNLL", "stage2_frozen_head_k256_conllonly_sent.pt"),
    ):
        s = AntecedentScorer(hidden=1024, use_distance=True).to(device)
        s.load_state_dict(torch.load(MODELS_DIR / ckpt, map_location=device)["scorer"])
        s.eval()
        n_clusters = n_docs = impure = matched_in_impure = 0
        key_docs, resp_docs = [], []
        for d in test:
            gset = set(int(c) for c in d["gold_cluster_id"])
            pw = _stage_a_clusters_for_doc(d, s, device, 256)
            resp = []
            for info in pw.values():
                gi = info["global_idx"]
                for cl in info["clusters"]:
                    members = gi[np.array(cl)]
                    n_clusters += 1
                    matched = [int(c) for c in d["cluster_id"][members] if int(c) in gset]
                    if len(set(matched)) > 1:  # mentions from >=2 gold entities -> over-merge
                        impure += 1
                        matched_in_impure += len(matched)
                    spans = [d["spans"][int(m)] for m in members]
                    if len(spans) >= 2:
                        resp.append(spans)
            n_docs += 1
            key_docs.append((d["name"], d["sentences"], _gold_key_full(d)))
            resp_docs.append((d["name"], d["sentences"], resp))
        kp, rp = MODELS_DIR / "saclus_key.conll", MODELS_DIR / "saclus_resp.conll"
        write_conll(kp, key_docs)
        write_conll(rp, resp_docs)
        f = conll_f1(kp, rp)
        out[tag] = {"nodes_per_doc": n_clusters / max(n_docs, 1), "overmerge_pct": 100 * impure / max(n_clusters, 1),
                    "overmerged_mentions": matched_in_impure, "nomerge": f}
        print(f"\n{tag}:")
        print(f"  Stage A clusters/doc {out[tag]['nodes_per_doc']:.1f} | "
              f"OVER-MERGED clusters {impure}/{n_clusters} ({out[tag]['overmerge_pct']:.1f}%) "
              f"| matched mentions in over-merged clusters {matched_in_impure}")
        print(f"  no-merge F1 (full gold key): CoNLL {f['CoNLL'] * 100:.2f} "
              f"(MUC {f['muc'] * 100:.2f} B3 {f['bcub'] * 100:.2f} CEAFe {f['ceafe'] * 100:.2f})")
    return out


def diagnose_stage_a_predicted(thr: float = 0.2, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    # Head-to-head on the SAME predicted test mentions: predicted-TRAINED Stage A (76) vs gold-TRAINED
    # (90.20). Per-decision outcomes expose the failure mode directly — over-nulling (high MISSED_LINK,
    # high null_bias) = the null-swamp; wrong/false links = label-noise. Decisions scored vs aligned
    # cluster_id (the predicted-mention coref labels the head was trained on).
    from disambiguation.stage_a import _stage_a_run

    docs = build_predicted_docs(str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt"), thr=thr, device=device)
    test_idx = [i for i, d in enumerate(docs) if d["split"] == "test"]
    out = {}
    for tag, ckpt in (
        ("predicted-trained", "stage2_frozen_head_predicted_conll.pt"),
        ("gold-trained", "stage2_frozen_head_k256_all8k_sent.pt"),
    ):
        path = MODELS_DIR / ckpt
        if not path.exists():
            print(f"[skip {tag}: {ckpt} missing]")
            continue
        scorer = AntecedentScorer(hidden=1024, use_distance=True).to(device)
        scorer.load_state_dict(torch.load(path, map_location=device)["scorer"])
        scorer.eval()
        out[tag] = (_stage_a_run(docs, test_idx, scorer, 256, device), float(scorer.null_bias.item()))

    print("\n=== Stage A per-decision diagnosis on PREDICTED test mentions ===")
    for tag, (r, nb) in out.items():
        o = r["outcome"]
        tl, ml = o.get("TRUE_LINK", 0), o.get("MISSED_LINK", 0)
        wl, fl, tn = o.get("WRONG_LINK", 0), o.get("FALSE_LINK", 0), o.get("TRUE_NULL", 0)
        lr = 100 * tl / max(tl + ml, 1)
        lp = 100 * tl / max(tl + wl + fl, 1)
        print(f"\n{tag}: null_bias {nb:+.3f}")
        print(f"  TRUE_LINK {tl} | MISSED_LINK {ml} | WRONG_LINK {wl} | FALSE_LINK {fl} | TRUE_NULL {tn}")
        print(f"  link recall {lr:.1f} | link precision {lp:.1f}")
        print("  recall by type: " + "  ".join(
            f"{t} {100 * r['rec'][t][0] / max(r['rec'][t][1], 1):.1f}({r['rec'][t][1]})" for t in ("PRON", "NOUN", "PROPN")
        ))
    return out


def run_stage_a_predicted(thr: float = 0.2, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    # Baseline: gold-trained Stage A run on the detector's predicted mentions (no retraining), Stage A
    # alone (window-local, no Stage B merge) — the first end-to-end-on-predicted number.
    det_ckpt = str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt")
    docs = build_predicted_docs(det_ckpt, thr=thr, device=device)
    scorer = load_gold_stage_a(256, "all8k", "both", sent_aligned=True, device=device)
    res = eval_stage_a_predicted(docs, scorer, device, window=256)
    print(f"\n=== Stage A intra-window on PREDICTED mentions (thr={thr}, vs window-split gold; gold=89.48) ===")
    print(
        f"CoNLL {res['CoNLL'] * 100:.2f} | MUC {res['muc'] * 100:.2f} "
        f"B3 {res['bcub'] * 100:.2f} CEAFe {res['ceafe'] * 100:.2f}"
    )
    return res
