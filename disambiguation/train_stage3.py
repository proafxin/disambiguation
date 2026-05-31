import json
import pickle
import random
import sys

import numpy as np
import spacy
import spacy.tokens
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from torch import optim
from tqdm import tqdm

from disambiguation.conll_scorer import conll_f1, conll_f1_by_type, write_conll
from disambiguation.paths import DATA_DIR, MODELS_DIR, SPACY_TRF_DIR
from disambiguation.stage2_context_encoder import (
    CONTENT,
    AntecedentScorer,
    ClusterMatcher,
    ContextEncoder,
    cluster_match_loss,
    encode_document_ctx,
    load_tokenizer,
)
from disambiguation.train_stage2 import (
    BGE_MODEL,
    RANDOM_SEED,
    _cluster_reps,
    _cluster_gold_cids,
    _predict_full_doc_clusters,
    _raw_from_doc,
    _stage_a_clusters_for_doc,
    eval_conll,
    eval_stage_b,
    gather_spans_np,
    run_epoch,
)

# Two evaluation protocols, both predicted-mention via spaCy, both fully separate
# from the Stage 2/A/B (gold-mention) artifacts:
#   mode="head" : candidate = single dependency-head token; scored head-match.
#   mode="span" : candidate = dependency-subtree span;     scored exact-span (SOTA protocol).
NOMINAL_POS = {"PRON", "NOUN", "PROPN"}
CONLL_SPLITS = ["train", "validation", "test"]


def _paths(mode: str) -> dict:
    return {
        "nom": DATA_DIR / f"stage3_{mode}_nominals.pkl",
        "ctx": DATA_DIR / f"stage3_{mode}_span_ctx",
        "a_ckpt": MODELS_DIR / f"stage3_{mode}_stage_a.pt",
        "b_ckpt": MODELS_DIR / f"stage3_{mode}_cluster_matcher.pt",
        "a_clusters": MODELS_DIR / f"stage3_{mode}_stage_a_clusters.pkl",
        "metrics": MODELS_DIR / f"stage3_{mode}_eval_metrics.json",
    }


def _gold_head_map(sample: dict, spacy_sents: list) -> tuple[dict, int]:
    head_map: dict[tuple[int, int], int] = {}
    max_cid = -1
    for cid, cluster in enumerate(sample["mention_clusters"]):
        for si, a, b in cluster:
            if si >= len(spacy_sents):
                continue
            sent = spacy_sents[si]
            if a >= len(sent):
                continue
            hl = sent[a : min(b, len(sent))].root.i - sent.start
            head_map.setdefault((si, hl), cid)
            max_cid = max(max_cid, cid)
    return head_map, max_cid


def _subtree_span(token, sent) -> tuple[int, int]:
    locs = [t.i - sent.start for t in token.subtree if sent.start <= t.i < sent.end]
    if not locs:
        loc = token.i - sent.start
        return loc, loc + 1
    return min(locs), max(locs) + 1


def _candidates(sample: dict, spacy_doc: spacy.tokens.Doc, mode: str) -> tuple:
    # Candidates from spaCy nominal heads; labels via gold-head map (unmatched -> singleton).
    sents = sample["sentences"]
    offsets, off = [], 0
    for s in sents:
        offsets.append(off)
        off += len(s)
    spacy_sents = list(spacy_doc.sents)
    head_map, max_cid = _gold_head_map(sample, spacy_sents)
    spans, cluster_id, surfaces = [], [], []
    next_singleton, seen = max_cid + 1, set()
    for si, sent in enumerate(spacy_sents):
        if si >= len(sents):
            break
        n = len(sents[si])
        for t in sent:
            local = t.i - sent.start
            if t.pos_ not in NOMINAL_POS or local >= n:
                continue
            if mode == "head":
                a, b = local, local + 1
            else:
                a, b = _subtree_span(t, sent)
                b = min(b, n)
                if a >= n or b <= a:
                    continue
            if (si, a, b) in seen:
                continue
            seen.add((si, a, b))
            spans.append((si, a, b))
            cid = head_map.get((si, local))
            cluster_id.append(cid if cid is not None else next_singleton)
            if cid is None:
                next_singleton += 1
            surfaces.append(" ".join(sents[si][a:b]))
    return sents, offsets, spans, cluster_id, surfaces


def _gold_key(sample: dict, spacy_sents: list, mode: str) -> list:
    # mode="head": gold mentions reduced to head tokens. mode="span": gold full spans.
    clusters = []
    for cluster in sample["mention_clusters"]:
        mentions = set()
        for si, a, b in cluster:
            if si >= len(spacy_sents):
                continue
            sent = spacy_sents[si]
            if a >= len(sent):
                continue
            if mode == "head":
                hl = sent[a : min(b, len(sent))].root.i - sent.start
                mentions.add((si, hl, hl + 1))
            else:
                mentions.add((si, a, b))
        if len(mentions) >= 2:
            clusters.append(sorted(mentions))
    return clusters


def _build_raw(mode: str, tokenizer) -> tuple[list, dict]:
    vocab = spacy.blank("en").vocab
    raw, gold_keys = [], {}
    for split in CONLL_SPLITS:
        ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
        db = spacy.tokens.DocBin().from_disk(SPACY_TRF_DIR / f"conll2012_{split}.spacy")
        for sample, sdoc in tqdm(
            zip(ds, db.get_docs(vocab), strict=True), total=len(ds), desc=f"stage3-{mode} build/{split}"
        ):
            spacy_sents = list(sdoc.sents)
            sents, offsets, spans, cluster_id, surfaces = _candidates(sample, sdoc, mode)
            name = f"conll2012/{sample['doc_id']}#{len(raw)}"
            rec = _raw_from_doc(name, split, sents, offsets, spans, cluster_id, surfaces, tokenizer)
            if not rec:
                continue
            raw.append(rec)
            gold_keys[name] = _gold_key(sample, spacy_sents, mode)
    return raw, gold_keys


def build_stage3_docs(device: str, mode: str) -> tuple[list, dict]:
    P = _paths(mode)
    tokenizer = load_tokenizer()
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    if P["nom"].exists():
        with P["nom"].open("rb") as f:
            blob = pickle.load(f)
        docs, gold_keys = blob["docs"], blob["gold_keys"]
        for d in docs:
            d["mention_bge"] = d["mention_bge"].astype(np.float32)
        print(f"[{mode}] loaded cached docs: {len(docs)}")
    else:
        raw, gold_keys = _build_raw(mode, tokenizer)
        surface_vocab = sorted({s for r in raw for s in r[5]})
        print(f"[{mode}] encoding {len(surface_vocab)} unique surfaces with BGE...")
        bge = SentenceTransformer(BGE_MODEL, device=device)
        embs = np.asarray(
            bge.encode(surface_vocab, normalize_embeddings=True, batch_size=512, show_progress_bar=True),
            dtype=np.float32,
        )
        surf2bge = {s: embs[i] for i, s in enumerate(surface_vocab)}
        del bge
        if device != "cpu":
            torch.cuda.empty_cache()
        docs = []
        for (name, split, sents, spans, cluster_id, surfaces, content_ids, span_sub, sso, ssl) in raw:
            docs.append(
                {
                    "name": name, "split": split, "sentences": sents, "spans": spans,
                    "cluster_id": np.asarray(cluster_id, dtype=np.int64),
                    "content_ids": content_ids,
                    "span_sub": np.asarray(span_sub, dtype=np.int64),
                    "tok_pos": np.asarray([s for s, _ in span_sub], dtype=np.int64),
                    "mention_bge": np.stack([surf2bge[s] for s in surfaces]).astype(np.float32),
                }
            )
        with P["nom"].open("wb") as f:
            pickle.dump(
                {"docs": [{**d, "mention_bge": d["mention_bge"].astype(np.float16)} for d in docs],
                 "gold_keys": gold_keys},
                f,
            )
        print(f"[{mode}] cached {len(docs)} docs -> {P['nom'].name}")

    P["ctx"].mkdir(parents=True, exist_ok=True)
    ctx_path = lambda i: P["ctx"] / f"{i:06d}.npy"
    if all(ctx_path(i).exists() for i in range(len(docs))):
        for i, d in enumerate(docs):
            d["ctx_vecs"] = np.load(ctx_path(i)).astype(np.float32)
            d.pop("content_ids", None)
        print(f"[{mode}] loaded cached ctx vectors")
        return docs, gold_keys
    encoder = ContextEncoder().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    with torch.inference_mode():
        for i, d in enumerate(tqdm(docs, desc=f"stage3-{mode} ctx")):
            p = ctx_path(i)
            if p.exists():
                d["ctx_vecs"] = np.load(p).astype(np.float32)
                d.pop("content_ids", None)
                continue
            ctx = encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device).float().cpu().numpy()
            cv = gather_spans_np(ctx, d["span_sub"])
            np.save(p, cv)
            d["ctx_vecs"] = cv.astype(np.float32)
            d.pop("content_ids", None)
    del encoder
    if device != "cpu":
        torch.cuda.empty_cache()
    return docs, gold_keys


def _split(docs: list, split: str) -> list:
    return [d for d in docs if d["split"] == split]


def eval_stage3_full(
    docs: list, gold_keys: dict, stage_a: AntecedentScorer, matcher: ClusterMatcher,
    device: str, tag: str, type_breakdown: bool = False,
) -> dict:
    # Response = Stage A+B clusters over spaCy candidates; key = full gold clusters.
    # mode="head" keys/cands are single tokens (head match); mode="span" are full spans (exact match).
    stage_a.eval()
    matcher.eval()
    key_docs, resp_docs = [], []
    for d in docs:
        per_window = _stage_a_clusters_for_doc(d, stage_a, device)
        resp = _predict_full_doc_clusters(d, per_window, matcher, device)
        key_docs.append((d["name"], d["sentences"], gold_keys[d["name"]]))
        resp_docs.append((d["name"], d["sentences"], resp))
    key_path = MODELS_DIR / f"{tag}_key.conll"
    resp_path = MODELS_DIR / f"{tag}_resp.conll"
    write_conll(key_path, key_docs)
    write_conll(resp_path, resp_docs)
    result = conll_f1(key_path, resp_path)
    if type_breakdown:
        result["by_type"] = conll_f1_by_type(key_docs, resp_docs, MODELS_DIR)
    return result


def train_stage3_a(
    mode: str, head_lr: float = 1e-3, max_epochs: int = 60, patience: int = 8, doc_bs: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    P = _paths(mode)
    if P["a_ckpt"].exists():
        print(f"[{mode}] Stage A checkpoint exists ({P['a_ckpt'].name}); skipping A training.")
        return
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    docs, _ = build_stage3_docs(device, mode)
    tokenizer = load_tokenizer()
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    train_docs, val_docs = _split(docs, "train"), _split(docs, "validation")
    print(f"[{mode}] Stage A — train {len(train_docs)} | val {len(val_docs)}")
    scorer = AntecedentScorer().to(device)
    optimizer = optim.AdamW(scorer.parameters(), lr=head_lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=head_lr * 0.1)
    best_f1, patience_ctr = -1.0, 0
    for epoch in range(max_epochs):
        print(f"\n[{mode}] Stage A epoch {epoch + 1}/{max_epochs}")
        random.shuffle(train_docs)
        scorer.train()
        tr = run_epoch(None, scorer, train_docs, optimizer, cls_id, sep_id, device, doc_bs)
        scheduler.step()
        scorer.eval()
        val = eval_conll(None, scorer, val_docs, cls_id, sep_id, device, f"stage3_{mode}_a_val", window=CONTENT)
        print(f"  loss {tr:.6f} | val CoNLL {val['CoNLL']:.4f}")
        if val["CoNLL"] > best_f1 + 1e-4:
            best_f1, patience_ctr = val["CoNLL"], 0
            torch.save({"scorer": scorer.state_dict(), "best_val_f1": best_f1}, P["a_ckpt"])
            print(f"  ✓ saved (val {best_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  ⊘ early stop (best {best_f1:.4f})")
                break
    print(f"[{mode}] Stage A complete")


def train_stage3_b(
    mode: str, head_lr: float = 1e-3, max_epochs: int = 30, patience: int = 5, pair_bs: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    P = _paths(mode)
    docs, gold_keys = build_stage3_docs(device, mode)
    stage_a = AntecedentScorer().to(device)
    stage_a.load_state_dict(torch.load(P["a_ckpt"], map_location=device)["scorer"])
    for p in stage_a.parameters():
        p.requires_grad_(False)
    stage_a.eval()

    if P["a_clusters"].exists():
        with P["a_clusters"].open("rb") as f:
            all_clusters = pickle.load(f)
    else:
        all_clusters = []
        with torch.inference_mode():
            for d in tqdm(docs, desc=f"stage3-{mode} A clusters"):
                all_clusters.append(_stage_a_clusters_for_doc(d, stage_a, device) if "ctx_vecs" in d else {})
        with P["a_clusters"].open("wb") as f:
            pickle.dump(all_clusters, f)

    it = [i for i, d in enumerate(docs) if d["split"] == "train"]
    iv = [i for i, d in enumerate(docs) if d["split"] == "validation"]
    train_docs, train_clusters = [docs[i] for i in it], [all_clusters[i] for i in it]
    val_docs, val_clusters = [docs[i] for i in iv], [all_clusters[i] for i in iv]

    matcher = ClusterMatcher().to(device)
    optimizer = optim.AdamW(matcher.parameters(), lr=head_lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=head_lr * 0.1)
    best_f1, patience_ctr = -1.0, 0
    for epoch in range(max_epochs):
        print(f"\n[{mode}] Stage B epoch {epoch + 1}/{max_epochs}")
        order = list(range(len(train_docs)))
        random.shuffle(order)
        matcher.train()
        total, n_pairs, pending = 0.0, 0, []
        for bi in tqdm(order, desc=f"stage3-{mode} train_b"):
            d, per_window = train_docs[bi], train_clusters[bi]
            if not per_window:
                continue
            ctx_all, bge_all, cid = d["ctx_vecs"], d["mention_bge"], d["cluster_id"]
            windows = sorted(per_window.keys())
            for i in range(len(windows)):
                for j in range(i + 1, len(windows)):
                    wi, wj = per_window[windows[i]], per_window[windows[j]]
                    lc, rc = wi["clusters"], wj["clusters"]
                    if not lc or not rc:
                        continue
                    ci, bi_, cj, bj = (
                        ctx_all[wi["global_idx"]], bge_all[wi["global_idx"]],
                        ctx_all[wj["global_idx"]], bge_all[wj["global_idx"]],
                    )
                    left = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(cl, ci, bi_) for cl in lc]]
                    right = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(cl, cj, bj) for cl in rc]]
                    lcid = [_cluster_gold_cids([wi["global_idx"][li] for li in cl], cid) for cl in lc]
                    rcid = [_cluster_gold_cids([wj["global_idx"][li] for li in cl], cid) for cl in rc]
                    pending.append((left, right, lcid, rcid))
                    n_pairs += 1
                    if len(pending) >= pair_bs:
                        optimizer.zero_grad()
                        losses = [cluster_match_loss(matcher(l, r), lc, rc, matcher.null_bias) for l, r, lc, rc in pending]
                        torch.stack(losses).mean().backward()
                        torch.nn.utils.clip_grad_norm_(matcher.parameters(), 1.0)
                        optimizer.step()
                        total += sum(lo.item() for lo in losses)
                        pending = []
        if pending:
            optimizer.zero_grad()
            losses = [cluster_match_loss(matcher(l, r), lc, rc, matcher.null_bias) for l, r, lc, rc in pending]
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), 1.0)
            optimizer.step()
            total += sum(lo.item() for lo in losses)
        scheduler.step()
        val = eval_stage_b(val_docs, val_clusters, matcher, device, f"stage3_{mode}_b_val")
        print(f"  loss {total / max(n_pairs, 1):.6f} | val CoNLL {val['CoNLL']:.4f}")
        if val["CoNLL"] > best_f1 + 1e-4:
            best_f1, patience_ctr = val["CoNLL"], 0
            torch.save({"cluster_matcher": matcher.state_dict(), "best_val_f1": best_f1}, P["b_ckpt"])
            print(f"  ✓ saved (val {best_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  ⊘ early stop (best {best_f1:.4f})")
                break
    print(f"[{mode}] Stage B complete")

    matcher.load_state_dict(torch.load(P["b_ckpt"], map_location=device)["cluster_matcher"])
    test_docs = _split(docs, "test")
    metrics = eval_stage3_full(test_docs, gold_keys, stage_a, matcher, device, f"stage3_{mode}_test", type_breakdown=True)
    proto = "head-match" if mode == "head" else "exact-span"
    print(
        f"\n[{mode}] Stage 3 test (spaCy mentions, {proto}): CoNLL {metrics['CoNLL']:.4f} "
        f"| MUC {metrics['muc']:.4f} | B3 {metrics['bcub']:.4f} | CEAFe {metrics['ceafe']:.4f}"
    )
    metrics["setting"] = f"spacy_mentions_{mode}_{proto}"
    metrics["eval_dataset"] = "conll2012_test"
    metrics["train_datasets"] = ["conll2012"]
    with P["metrics"].open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_stage3(mode: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
    train_stage3_a(mode, device=device)
    train_stage3_b(mode, device=device)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for m in (["head", "span"] if which == "both" else [which]):
        print(f"\n========== Stage 3 mode = {m} ==========")
        run_stage3(m)
