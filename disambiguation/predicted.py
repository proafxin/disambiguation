import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from disambiguation.conll_scorer import conll_f1, write_conll
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list:
    # End-to-end mention bridge: run the BIO detector over the CoNLL docs, snap to word spans, rebuild
    # the exact Stage-A per-mention fields, BGE-encode the surfaces, gather frozen-RoBERTa ctx at the
    # predicted spans, align predicted->gold for cluster labels. Gold spans/clusters are preserved
    # (d["gold_*"]) for the scoring key. Writes its own caches — the gold caches are never touched.
    if PRED_NOM_CACHE.exists():
        with PRED_NOM_CACHE.open("rb") as f:
            docs = pickle.load(f)
        precompute_span_ctx(None, docs, 0, 0, device, cache_dir=PRED_CTX_BASE, window=res_window, single=True)
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
        print(f"Loaded cached predicted docs: {len(docs)}")
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
    detector = BIOTagger(n_layers=n_layers, model_name=BACKBONE).to(device)
    detector.load_state_dict(torch.load(det_ckpt, map_location=device)["model"])
    detector.eval()
    from disambiguation.stage_a import _bio_predict_doc  # local import: stage_a imports data, avoid cycle

    for d in tqdm(docs, desc="detect"):
        d["pred_sub"] = sorted(_bio_predict_doc(detector, d, tokenizer, device, det_window, win_bs, n_layers, thr))
    del detector
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
    precompute_span_ctx(encoder, docs, cls_id, sep_id, device, cache_dir=PRED_CTX_BASE, window=res_window, single=True)
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
    with PRED_NOM_CACHE.open("wb") as f:
        pickle.dump(slim, f)
    print(f"Cached {len(docs)} predicted docs to {PRED_NOM_CACHE.name}")
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


def load_gold_stage_b(device: str) -> tuple:
    from disambiguation.data import _win_names
    from disambiguation.stage2_context_encoder import ClusterGNN

    ckpt_b = _win_names(256, "all8k", "both", sent_aligned=True)[2].replace(".pt", "_gnn_lse_lex.pt")
    gnn = ClusterGNN(channel="both", member_pool="lse", use_lexical=True).to(device)
    ck = torch.load(MODELS_DIR / ckpt_b, map_location=device)
    gnn.load_state_dict(ck["cluster_matcher"])
    gnn.eval()
    return gnn, float(ck.get("nb_offset", 0.0))


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
    thr: float = 0.2, nb_offset: float = 0.0, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    # Step 1 baseline: gold-trained Stage A + gold-trained Stage B, reused on predicted mentions.
    from disambiguation.stage_a import _stage_a_clusters_for_doc

    det_ckpt = str(MODELS_DIR / "bio_tagger_k510_L3_sent.pt")
    docs = build_predicted_docs(det_ckpt, thr=thr, device=device)
    scorer = load_gold_stage_a(256, "all8k", "both", sent_aligned=True, device=device)
    clusters = [
        _stage_a_clusters_for_doc(d, scorer, device, 256) if "ctx_vecs" in d and len(d["spans"]) >= 2 else {}
        for d in tqdm(docs, desc="stage A clusters")
    ]
    gnn, _ = load_gold_stage_b(device)
    res = eval_stage_ab_predicted(docs, clusters, gnn, device, nb_offset=nb_offset)
    print(f"\n=== Stage A+B end-to-end on PREDICTED mentions (thr={thr}, nb_offset={nb_offset:+.2f}; gold A+B=86.46) ===")
    print(
        f"CoNLL {res['CoNLL'] * 100:.2f} | MUC {res['muc'] * 100:.2f} "
        f"B3 {res['bcub'] * 100:.2f} CEAFe {res['ceafe'] * 100:.2f}"
    )
    return res


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
