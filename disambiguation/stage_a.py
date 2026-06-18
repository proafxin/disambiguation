import datetime
import json
import pickle
import random
from collections import Counter

import numpy as np
import spacy.tokens
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from disambiguation.conll_scorer import conll_f1, conll_f1_by_type, write_conll
from disambiguation.data import (
    _PRONOUNS,
    RANDOM_SEED,
    _apply_sentence_windows,
    _build_lexical,
    _data_cfg,
    _filter_docs,
    _key_clusters,
    _train_idx,
    _win_ids,
    _win_names,
    _word_to_subtok,
    build_docs,
    precompute_full_ctx,
    precompute_span_ctx,
)
from disambiguation.paths import DATA_DIR, MODELS_DIR, SPACY_TRF_DIR, TENSORBOARD_DIR
from disambiguation.stage2_context_encoder import (
    BACKBONE,
    BGE_DIM,
    CONTENT,
    CTX_DIM,
    ENCODER_TAG,
    AntecedentScorer,
    BIOTagger,
    ContextEncoder,
    MentionDetector,
    MentionTransformer,
    decode_antecedents,
    load_tokenizer,
)


def doc_scores(d: dict, scorer, device, window: int = CONTENT) -> list[tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    # Partition the doc into fixed CONTENT-subtoken windows and score each window's mention pairs (Stage A).
    ctx = torch.from_numpy(d["ctx_vecs"]).to(device).float()
    bge = torch.from_numpy(d["mention_bge"]).to(device).float()
    win_ids = _win_ids(d, window)
    results = []
    for w in np.unique(win_ids):
        idx = np.where(win_ids == w)[0]
        if len(idx) < 2:
            continue
        w_scores, w_mask = scorer(ctx[idx], bge[idx])
        results.append((w_scores, w_mask, idx))
    return results


def stage_a_batched_loss(scorer, windows: list, device: str) -> torch.Tensor:
    # windows: list of (ctx (M,2,CTX), bge (M,BGE), cid (M,) long, weight float), all M>=2.
    # One mention_rep + one chunked FFNN over the global within-window pair list, then a
    # vectorized MLL over a padded (B, maxM, maxM) score tensor. Equivalent to looping
    # mll_loss per window (window-level mean), but batched so the GPU isn't fed one window
    # at a time. Window losses are combined as a dataset-weighted mean.
    sizes = [w[0].shape[0] for w in windows]
    b, max_m = len(windows), max(sizes)
    g = scorer.mention_rep(torch.cat([w[0] for w in windows], 0), torch.cat([w[1] for w in windows], 0))
    gi, gj, li, lj, wid, dist = [], [], [], [], [], []
    off = 0
    for w, m in enumerate(sizes):
        idx = torch.arange(m, device=device)
        ii, jj = (idx.unsqueeze(0) < idx.unsqueeze(1)).nonzero(as_tuple=True)  # ii>jj (antecedent jj)
        gi.append(ii + off)
        gj.append(jj + off)
        li.append(ii)
        lj.append(jj)
        wid.append(torch.full_like(ii, w))
        dist.append(ii - jj)
        off += m
    gi, gj, li, lj, wid, dist = (torch.cat(t) for t in (gi, gj, li, lj, wid, dist))
    pair = torch.empty(gi.shape[0], device=device, dtype=g.dtype)
    for s0 in range(0, gi.shape[0], scorer.chunk):
        sl = slice(s0, s0 + scorer.chunk)
        parts = [g[gi[sl]], g[gj[sl]]]
        if scorer.use_distance:
            bucket = torch.bucketize(dist[sl].clamp(min=0), scorer.dist_bounds, right=True)
            parts.append(scorer.dist_emb(bucket))
        feat = torch.cat(parts, dim=-1)
        pair[sl] = scorer.ffnn(feat).squeeze(-1).to(g.dtype)

    scores = torch.zeros(b, max_m, max_m, device=device, dtype=g.dtype)
    scores[wid, li, lj] = pair
    return _padded_mll(scores, sizes, [w[2] for w in windows], [w[3] for w in windows], scorer.null_bias, device)


def _padded_mll(scores, sizes, cids, weights, null_bias, device) -> torch.Tensor:
    # Vectorized MLL over a padded (B, maxM, maxM) dense score tensor: each mention i>=1 softmaxes
    # over {ε} ∪ {valid j<i}; gold = earlier mentions sharing its cluster id (else ε). Per-window
    # mean, then dataset-weighted mean. Shared by the FFNN head (scatter-built scores) and the
    # relational MentionTransformer (transformer-built scores). Pairs touching padded positions are
    # masked to neg, so padded rows contribute 0 (no NaN even if their reps are 0/NaN).
    b, max_m = scores.shape[0], scores.shape[1]
    neg = torch.finfo(scores.dtype).min
    ar = torch.arange(max_m, device=device)
    size_t = torch.tensor(sizes, device=device)
    valid = ar.unsqueeze(0) < size_t.unsqueeze(1)  # (B, maxM)
    ante = (ar.unsqueeze(1) > ar.unsqueeze(0)).unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)  # [b,i,j]: j<i
    cid = torch.full((b, max_m), -1, device=device, dtype=torch.long)
    for w in range(b):
        cid[w, : sizes[w]] = cids[w]
    null = null_bias
    null_col = null.view(1, 1, 1).expand(b, max_m, 1)
    denom = torch.logsumexp(torch.cat([null_col, scores.masked_fill(~ante, neg)], dim=2), dim=2)  # (B, maxM)
    gold = (cid.unsqueeze(2) == cid.unsqueeze(1)) & ante  # [b,i,j]: same cluster, j<i
    has_gold = gold.any(dim=2)
    num = torch.where(has_gold, torch.logsumexp(scores.masked_fill(~gold, neg), dim=2), null.squeeze().expand(b, max_m))
    per_ment = denom - num  # (B, maxM)
    loss_mask = valid & (ar.unsqueeze(0) >= 1)
    w_loss = (per_ment * loss_mask).sum(1) / loss_mask.sum(1).clamp(min=1)  # (B,) per-window mean
    wt = torch.tensor(weights, device=device, dtype=w_loss.dtype)
    return (w_loss * wt).sum() / wt.sum()


def mention_transformer_loss(scorer, windows: list, device: str) -> torch.Tensor:
    # Relational Stage A loss: pad the batch's windows to (B, maxM), run one padded transformer
    # forward (mentions attend within each window), then the shared padded MLL.
    sizes = [w[0].shape[0] for w in windows]
    b, max_m = len(windows), max(sizes)
    ctx = torch.zeros(b, max_m, 2, CTX_DIM, device=device)
    bge = torch.zeros(b, max_m, BGE_DIM, device=device)
    pad = torch.ones(b, max_m, dtype=torch.bool, device=device)
    for k, w in enumerate(windows):
        m = sizes[k]
        ctx[k, :m], bge[k, :m], pad[k, :m] = w[0], w[1], False
    scores = scorer.batched_scores(ctx, bge, pad)
    return _padded_mll(scores, sizes, [w[2] for w in windows], [w[3] for w in windows], scorer.null_bias, device)


def _collect_windows(batch: list, device: str, window: int, loss_weights) -> list:
    # turn a batch of docs into a flat list of (ctx, bge, cid, weight) windows
    windows = []
    for d in batch:
        ctx = torch.from_numpy(d["ctx_vecs"]).to(device).float()
        bge = torch.from_numpy(d["mention_bge"]).to(device).float()
        cid = torch.from_numpy(d["cluster_id"]).to(device)
        win_ids = _win_ids(d, window)
        wt = 1.0 if loss_weights is None else float(loss_weights.get(d["name"].split("/")[0], 1.0))
        for w in np.unique(win_ids):
            idx = np.where(win_ids == w)[0]
            if len(idx) >= 2:
                windows.append((ctx[idx], bge[idx], cid[idx], wt))
    return windows


def run_epoch(
    scorer,
    docs,
    optimizer,
    device,
    doc_bs,
    window: int = CONTENT,
    loss_weights: dict | None = None,
) -> float:
    train = optimizer is not None
    total, ndoc = 0.0, 0
    for s in tqdm(range(0, len(docs), doc_bs), desc="train" if train else "val"):
        batch = docs[s : s + doc_bs]
        if train:
            optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            windows = _collect_windows(batch, device, window, loss_weights)
            if not windows:
                continue
            loss = (
                mention_transformer_loss(scorer, windows, device)
                if isinstance(scorer, MentionTransformer)
                else stage_a_batched_loss(scorer, windows, device)
            )
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(scorer.parameters()), 1.0)
            optimizer.step()
        total += loss.item() * len(batch)
        ndoc += len(batch)
    return total / max(ndoc, 1)


def predict_clusters(d: dict, scorer, device, window: int = CONTENT) -> list:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
        windows = doc_scores(d, scorer, device, window=window)
    clusters = []
    null_bias = float(scorer.null_bias.item())
    for w_scores, w_mask, idx in windows:
        groups = decode_antecedents(w_scores.float().cpu().numpy(), w_mask.cpu().numpy(), null_bias)
        clusters.extend([d["spans"][idx[i]] for i in g] for g in groups)
    return clusters


def eval_conll(scorer, docs, device, tag, type_breakdown=False, window: int | None = None) -> dict:
    key_docs = [(d["name"], d["sentences"], _key_clusters(d, window)) for d in docs]
    key_path = MODELS_DIR / f"stage2_{tag}_key.conll"
    write_conll(key_path, key_docs)
    resp_docs = [
        (d["name"], d["sentences"], predict_clusters(d, scorer, device, window=window or CONTENT))
        for d in tqdm(docs, desc=f"eval/{tag}")
    ]
    resp_path = MODELS_DIR / f"stage2_{tag}_response.conll"
    write_conll(resp_path, resp_docs)
    result = conll_f1(key_path, resp_path)
    if type_breakdown:
        result["by_type"] = conll_f1_by_type(key_docs, resp_docs, MODELS_DIR)
    return result


def train_stage_a(
    head_lr: float = 1e-3,
    max_epochs: int | None = None,
    patience: int | None = None,
    doc_bs: int | None = None,
    window: int = CONTENT,
    subset: str = "all",
    dropout: float = 0.3,
    weight_decay: float = 0.1,
    loss_weights: dict | None = None,
    channel: str = "both",
    sent_aligned: bool = False,
    raw: bool = False,
    hidden: int = 1024,
    use_distance: bool = True,
    relational: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(
        f"\nBuilding Stage 2 nominal data... "
        f"(window={window}, subset={subset}, channel={channel}, sent_aligned={sent_aligned}, "
        f"raw={raw}, hidden={hidden}, use_distance={use_distance}, relational={relational})"
    )
    print("Evaluation setting: gold mentions")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    nom_cache, datasets, preco_n, single_ctx = _data_cfg(subset)
    docs = build_docs(device=device, datasets=datasets, nom_cache=nom_cache, preco_n=preco_n)
    docs = _filter_docs(docs, subset)
    for d in docs:  # retain mention subtoken start positions before precompute drops span_sub
        d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    if sent_aligned:  # before precompute drops the sentence offsets
        _apply_sentence_windows(docs, window)
    tokenizer = load_tokenizer()
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    conll_val_docs = [d for d in docs if d["split"] == "validation" and d["name"].startswith("conll2012/")]
    conll_test_docs = [d for d in docs if d["split"] == "test" and d["name"].startswith("conll2012/")]
    train_docs = [docs[i] for i in _train_idx(docs, subset)]
    val_sets = {
        "conll2012": conll_val_docs,
        "litbank": [d for d in docs if d["split"] == "validation" and d["name"].startswith("litbank/")],
        "preco": [d for d in docs if d["split"] == "validation" and d["name"].startswith("preco/")],
        "corefud": [d for d in docs if d["split"] == "validation" and d["name"].startswith("corefud/")],
    }

    ctx_dir, frozen_name, _, _ = _win_names(window, subset, channel, sent_aligned=sent_aligned)
    if raw:  # raw head is a different architecture — tag it so it never collides with / warm-starts the projected head
        frozen_name += "_raw"
    if not use_distance:  # distance-off head has a different ffnn input width — tag it too
        frozen_name += "_nodist"
    if relational:  # relational head (MentionTransformer) — different architecture, separate checkpoint
        frozen_name += "_rel"
        scorer = MentionTransformer(
            proj_dim=hidden, hidden=hidden, dropout=dropout, channel=channel, use_distance=use_distance
        ).to(device)
    else:
        scorer = AntecedentScorer(
            dropout=dropout, channel=channel, raw=raw, hidden=hidden, use_distance=use_distance
        ).to(device)
    encoder = ContextEncoder().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    precompute_span_ctx(encoder, docs, cls_id, sep_id, device, cache_dir=ctx_dir, window=window, single=single_ctx)
    del encoder
    if device != "cpu":
        torch.cuda.empty_cache()
    max_epochs, patience, doc_bs = max_epochs or 60, patience or 8, doc_bs or 8
    optimizer = optim.AdamW(list(scorer.parameters()), lr=head_lr, weight_decay=weight_decay)
    ckpt_path, tag = MODELS_DIR / f"{frozen_name}.pt", "frozen"
    head_params = sum(p.numel() for p in scorer.parameters())
    print(f"head params: {head_params:,} | doc_bs={doc_bs} epochs={max_epochs}")
    print(f"train docs: {len(train_docs)} | conll val: {len(conll_val_docs)} | conll test: {len(conll_test_docs)}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=head_lr * 0.1)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(TENSORBOARD_DIR / f"stage2_{tag}_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
    best_f1, patience_ctr, disk_best_f1 = -1.0, 0, -1.0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        scorer.load_state_dict(ckpt["scorer"])
        disk_best_f1 = ckpt.get("best_val_f1", -1.0)
        print(f"Warm-started from {ckpt_path.name} (best_val_f1 {disk_best_f1 * 100:.2f}); training fresh from epoch 0")

    for epoch in range(max_epochs):
        print(f"\n=== Stage A (K={window}) Epoch {epoch + 1}/{max_epochs} ===")
        random.shuffle(train_docs)
        scorer.train()
        tr_loss = run_epoch(scorer, train_docs, optimizer, device, doc_bs, window, loss_weights)
        scheduler.step()
        scorer.eval()
        with torch.inference_mode():
            val_losses = {
                ds: run_epoch(scorer, vdocs, None, device, doc_bs, window) for ds, vdocs in val_sets.items() if vdocs
            }
        val_loss = val_losses["conll2012"]
        val_scores = {
            ds: eval_conll(scorer, vdocs, device, f"{tag}_val_{ds}", window=window)
            for ds, vdocs in val_sets.items()
            if vdocs
        }
        val = val_scores["conll2012"]
        print(f"Loss: {tr_loss:.6f} | Val loss: {val_loss:.6f}")
        for ds, sc in val_scores.items():
            print(
                f"  {ds:10s} loss {val_losses[ds]:.4f}  CoNLL {sc['CoNLL'] * 100:.2f} "
                f"(MUC {sc['muc'] * 100:.2f} B3 {sc['bcub'] * 100:.2f} CEAFe {sc['ceafe'] * 100:.2f})"
            )
        tb.add_scalar("loss/train", tr_loss, epoch + 1)
        tb.add_scalar("loss/val", val_loss, epoch + 1)
        for ds, lo in val_losses.items():
            tb.add_scalar(f"loss/val/{ds}", lo, epoch + 1)
        tb.add_scalar("model/null_bias", scorer.null_bias.item(), epoch + 1)
        for ds, sc in val_scores.items():
            for k in ("CoNLL", "muc", "bcub", "ceafe"):
                tb.add_scalar(f"val_f1/{ds}/{k}", sc[k], epoch + 1)
        if val["CoNLL"] > best_f1 + 1e-4:
            best_f1, patience_ctr = val["CoNLL"], 0
            if best_f1 > disk_best_f1 + 1e-4:
                disk_best_f1 = best_f1
                torch.save({"scorer": scorer.state_dict(), "best_val_f1": disk_best_f1}, ckpt_path)
                print(f"✓ Best model saved (val CoNLL F1 {disk_best_f1 * 100:.2f})")
            else:
                print(
                    f"Improved this run to {best_f1 * 100:.2f} (all-time best {disk_best_f1 * 100:.2f}; not overwriting)"
                )
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"\n⊘ Early stopping. Best val CoNLL F1 {best_f1 * 100:.2f}")
                break

    print("\n✓ Training complete")
    tb.close()
    ckpt = torch.load(ckpt_path, map_location=device)
    scorer.load_state_dict(ckpt["scorer"])
    scorer.eval()
    print("\n=== Test ===")
    metrics = eval_conll(scorer, conll_test_docs, device, f"{tag}_test", type_breakdown=True, window=window)
    print(
        f"CoNLL {metrics['CoNLL'] * 100:.2f} | MUC {metrics['muc'] * 100:.2f} | "
        f"B3 {metrics['bcub'] * 100:.2f} | CEAFe {metrics['ceafe'] * 100:.2f}"
    )
    if "by_type" in metrics:
        print("\nType breakdown (official CoNLL F1 on cluster subsets):")
        for bucket, sc in sorted(metrics["by_type"].items()):
            print(
                f"  {bucket:12s}  CoNLL {sc['CoNLL'] * 100:.2f}  MUC {sc['muc'] * 100:.2f}  "
                f"B3 {sc['bcub'] * 100:.2f}  CEAFe {sc['ceafe'] * 100:.2f}"
            )
    metrics["best_val_f1"] = best_f1
    metrics["setting"] = "gold_mentions"
    metrics["random_seed"] = RANDOM_SEED
    metrics["train_datasets"] = ["conll2012", "litbank", "preco", "corefud"]
    metrics["eval_dataset"] = "conll2012_test"
    final_val_scores = {
        ds: eval_conll(scorer, vdocs, device, f"{tag}_final_val_{ds}", window=window)
        for ds, vdocs in val_sets.items()
        if vdocs
    }
    for ds, sc in final_val_scores.items():
        print(
            f"  val/{ds:10s} CoNLL {sc['CoNLL'] * 100:.2f} "
            f"(MUC {sc['muc'] * 100:.2f} B3 {sc['bcub'] * 100:.2f} CEAFe {sc['ceafe'] * 100:.2f})"
        )
        metrics[f"val_{ds}"] = sc
    with (MODELS_DIR / f"stage2_eval_metrics_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _stage_a_clusters_for_doc(d: dict, scorer: AntecedentScorer, device: str, window: int = CONTENT) -> dict[int, dict]:
    # Returns {win_id: {"clusters": list[list[int]],  # window-local indices
    #                   "global_idx": np.ndarray}}    # local->global mention index map
    # ctx/bge arrays are NOT stored — loaded from d at training time
    ctx_all = d["ctx_vecs"]
    bge_all = d["mention_bge"]
    win_ids = _win_ids(d, window)
    per_window: dict[int, dict] = {}
    with torch.inference_mode():
        for w in np.unique(win_ids):
            global_idx = np.where(win_ids == w)[0]
            ctx_w = torch.from_numpy(ctx_all[global_idx]).to(device).float()
            bge_w = torch.from_numpy(bge_all[global_idx]).to(device).float()
            Mw = len(global_idx)
            if Mw < 2:
                local_clusters = [[i] for i in range(Mw)]
            else:
                w_scores, w_mask = scorer(ctx_w, bge_w)
                groups = decode_antecedents(
                    w_scores.float().cpu().numpy(), w_mask.cpu().numpy(), float(scorer.null_bias.item())
                )
                grouped = {i for g in groups for i in g}
                local_clusters = [list(g) for g in groups]
                local_clusters += [[i] for i in range(Mw) if i not in grouped]
            per_window[int(w)] = {
                "clusters": local_clusters,
                "global_idx": global_idx,
            }
    return per_window


def precompute_stage_a_clusters(
    docs: list,
    scorer: AntecedentScorer,
    device: str,
    window: int = CONTENT,
    subset: str = "all",
    channel: str = "both",
    sent_aligned: bool = False,
    name_tag: str = "",
) -> list[dict[int, dict]]:
    # name_tag distinguishes clusters from a non-default Stage A architecture (e.g. raw/nodist head)
    cache_path = MODELS_DIR / _win_names(window, subset, channel, sent_aligned=sent_aligned)[3].replace(
        ".pkl", f"{name_tag}.pkl"
    )
    if cache_path.exists():
        print("Loading cached Stage A clusters...")
        with cache_path.open("rb") as f:
            return pickle.load(f)
    print("Precomputing Stage A clusters...")
    result = []
    scorer.eval()
    with torch.inference_mode():
        for d in tqdm(docs, desc="stage A clusters"):
            if "ctx_vecs" not in d:
                result.append({})
                continue
            result.append(_stage_a_clusters_for_doc(d, scorer, device, window))
    with cache_path.open("wb") as f:
        pickle.dump(result, f)
    print(f"Cached Stage A clusters to {cache_path.name}")
    return result


def _mention_type(toks: tuple, surf: list) -> str:
    if len(toks) == 1 and toks[0] in _PRONOUNS:
        return "PRON"
    if any(w[:1].isupper() for w in surf):
        return "PROPN"
    return "NOUN"


def _stage_a_run(docs: list, test_idx: list, scorer, window: int, device: str) -> dict:
    # one frozen Stage A head over CoNLL test: per-decision outcomes, recall by type, missed-by-head,
    # precision-error count, BGE-cosine on different-head links (resolved vs missed), conditions.
    nb = float(scorer.null_bias.item())
    outcome: Counter = Counter()
    rec = {t: [0, 0] for t in ("PRON", "NOUN", "PROPN")}  # [correct links, gold-positive] per anaphor type
    precs = {t: [0, 0] for t in ("PRON", "NOUN", "PROPN")}  # [correct links, links made] per anaphor type
    missed_head: Counter = Counter()
    cond: dict = {}
    prec = 0  # total precision errors (wrong + false links)
    bge_true, bge_miss = [], []  # diff-head BGE cos to gold antecedent

    def bump(name: str, ic: bool) -> None:
        c = cond.setdefault(name, [0, 0])
        c[0] += int(ic)
        c[1] += 1

    for i in test_idx:
        d = docs[i]
        if "ctx_vecs" not in d:
            continue
        cid, win_ids = d["cluster_id"], _win_ids(d, window)
        toks, spans, sents, bge_all = d["mention_tokens"], d["spans"], d["sentences"], d["mention_bge"]
        surf = [sents[si][a:b] for (si, a, b) in spans]
        types = [_mention_type(toks[k], surf[k]) for k in range(len(spans))]
        for w in np.unique(win_ids):
            gidx = np.where(win_ids == w)[0]
            if len(gidx) < 2:
                continue
            ctx = torch.from_numpy(d["ctx_vecs"][gidx]).to(device).float()
            bge = torch.from_numpy(bge_all[gidx]).to(device).float()
            with torch.inference_mode():
                sc, mask = scorer(ctx, bge)
            sc, mask = sc.float().cpu().numpy(), mask.cpu().numpy()
            wc, bg = cid[gidx], bge_all[gidx]
            wt = [types[k] for k in gidx]
            wh = [toks[k][-1] for k in gidx]
            for a in range(1, len(gidx)):
                cand = np.where(mask[a])[0]
                if len(cand) == 0:
                    continue
                gold_ante = [b for b in cand if wc[b] == wc[a]]
                for b in cand:
                    ic = bool(wc[b] == wc[a])
                    bump("ALL", ic)
                    if wh[a] == wh[b]:
                        bump("head_match", ic)
                    if wt[a] == wt[b]:
                        bump(f"sametype_{wt[a]}", ic)
                jb = int(cand[np.argmax(sc[a, cand])])
                linked = sc[a, jb] > nb
                if gold_ante:
                    rec[wt[a]][1] += 1
                    diffhead = all(wh[a] != wh[b] for b in gold_ante)
                    mc = max(float(bg[a] @ bg[b]) for b in gold_ante) if diffhead else None
                    if linked and wc[jb] == wc[a]:
                        outcome["TRUE_LINK"] += 1
                        rec[wt[a]][0] += 1
                        precs[wt[a]][0] += 1
                        precs[wt[a]][1] += 1
                        if mc is not None:
                            bge_true.append(mc)
                    elif linked:
                        outcome["WRONG_LINK"] += 1
                        prec += 1
                        precs[wt[a]][1] += 1
                    else:
                        outcome["MISSED_LINK"] += 1
                        missed_head["shared" if any(wh[a] == wh[b] for b in gold_ante) else "diff"] += 1
                        if mc is not None:
                            bge_miss.append(mc)
                elif linked:
                    outcome["FALSE_LINK"] += 1
                    prec += 1
                    precs[wt[a]][1] += 1
                else:
                    outcome["TRUE_NULL"] += 1
    return {
        "outcome": outcome,
        "rec": rec,
        "precs": precs,
        "missed_head": missed_head,
        "cond": cond,
        "prec": prec,
        "bge_true": bge_true,
        "bge_miss": bge_miss,
    }


def _window_of(pos: int, cstarts: np.ndarray | None, window: int) -> int:
    # subtoken position -> window id: searchsorted over sentence-chunk starts when sent-aligned,
    # else the fixed-K block pos // window.
    if cstarts is None:
        return pos // window
    return int(np.searchsorted(cstarts, pos, side="right") - 1)


def _doc_sentence_splits(d: dict, window: int) -> tuple[int, int]:
    # (sentences split across a window boundary, total sentences); sent-aligned splits only a
    # sentence that alone exceeds the window.
    off, ln = d.get("sent_sub_offsets"), d.get("sent_sub_lengths")
    if off is None or ln is None:
        return 0, 0
    chunks = d.get("win_chunks")
    cstarts = np.asarray([s for s, _ in chunks], dtype=np.int64) if chunks is not None else None
    split = sum(
        _window_of(int(off[s]), cstarts, window) != _window_of(int(off[s]) + int(ln[s]) - 1, cstarts, window)
        for s in range(len(off))
    )
    return int(split), len(off)


def _doc_pair_splits(d: dict, window: int) -> tuple[int, int]:
    # (gold intra-sentence coref pairs landing in different windows, total such pairs)
    win_ids, cid, spans = _win_ids(d, window), d["cluster_id"], d["spans"]
    si = np.array([s for (s, _, _) in spans])
    split = tot = 0
    for x in range(len(spans)):
        for y in range(x):
            if si[x] == si[y] and cid[x] == cid[y]:
                tot += 1
                split += int(win_ids[x] != win_ids[y])
    return split, tot


def _sentence_split_stats(docs: list, test_idx: list, window: int) -> tuple:
    # head-independent: fraction of sentences split across a window boundary, and fraction of gold
    # intra-sentence coref pairs that land in different windows (= lost to Stage A, handed to Stage B).
    split_sent = tot_sent = split_pair = tot_pair = 0
    for i in test_idx:
        ss, ts = _doc_sentence_splits(docs[i], window)
        sp, tp = _doc_pair_splits(docs[i], window)
        split_sent, tot_sent, split_pair, tot_pair = split_sent + ss, tot_sent + ts, split_pair + sp, tot_pair + tp
    return split_sent, tot_sent, split_pair, tot_pair


def _run_error_channels(
    docs: list, test_idx: list, window: int, subset: str, sent_aligned: bool, device: str
) -> dict:
    results = {}
    for ch in ("both", "bge", "ctx"):
        hp = MODELS_DIR / f"{_win_names(window, subset, ch, sent_aligned=sent_aligned)[1]}.pt"
        if not hp.exists():
            print(f"(skip channel '{ch}': {hp.name} missing)")
            continue
        sco = AntecedentScorer(channel=ch).to(device)
        sco.load_state_dict(torch.load(hp, map_location=device)["scorer"])
        sco.eval()
        results[ch] = _stage_a_run(docs, test_idx, sco, window, device)
    # relational head (both-channel MentionTransformer) shares the both-channel ctx cache — A/B column
    rp = MODELS_DIR / f"{_win_names(window, subset, 'both', sent_aligned=sent_aligned)[1]}_rel.pt"
    if rp.exists():
        sd = torch.load(rp, map_location=device)["scorer"]
        sco = MentionTransformer(  # infer width from the checkpoint so it never shape-mismatches
            proj_dim=sd["P_ctx.weight"].shape[0], hidden=sd["node_in.weight"].shape[0],
            channel="both", use_distance="dist_emb.weight" in sd,
        ).to(device)
        sco.load_state_dict(sd)
        sco.eval()
        results["rel"] = _stage_a_run(docs, test_idx, sco, window, device)
    return results


def _print_both_channel_detail(r: dict | None) -> None:
    if not r:
        return
    outcome, tot = r["outcome"], sum(r["outcome"].values())
    print("\n--- both-channel head detail ---")
    for k in ("TRUE_LINK", "WRONG_LINK", "MISSED_LINK", "TRUE_NULL", "FALSE_LINK"):
        print(f"  {k:12s} {outcome[k]:6d}  ({100 * outcome[k] / max(tot, 1):4.1f}%)")
    ms, md = r["missed_head"]["shared"], r["missed_head"]["diff"]
    print(
        f"\nMISSED links: head-shared {ms} ({100 * ms / max(ms + md, 1):.0f}%) | "
        f"head-different {md} ({100 * md / max(ms + md, 1):.0f}%)"
    )
    print(f"precision errors (wrong+false links): {r['prec']}")
    bt, bm = r["bge_true"], r["bge_miss"]
    print("\ndifferent-head NOUN/PROPN links — BGE cosine to gold antecedent (does BGE separate them?):")
    print(f"  RESOLVED (true link): mean {np.mean(bt):.3f}  median {np.median(bt):.3f}  (n={len(bt)})")
    print(f"  MISSED:               mean {np.mean(bm):.3f}  median {np.median(bm):.3f}  (n={len(bm)})")
    print("  (MISSED << RESOLVED -> BGE separates, misses are low-similarity = need better embeddings;")
    print("   MISSED ~ RESOLVED -> BGE has the signal but the head isn't using it)")
    p0 = r["cond"]["ALL"][0] / max(r["cond"]["ALL"][1], 1)
    print(f"\nconditions (prior P(coref)={100 * p0:.1f}%):")
    for name in ("head_match", "sametype_NOUN", "sametype_PROPN"):
        if name in r["cond"]:
            c, t = r["cond"][name]
            print(f"  {name:14s} P(coref)={100 * c / max(t, 1):5.1f}%   [{t} pairs]")


def stage_a_error_analysis(
    window: int = CONTENT,
    subset: str = "all",
    channel: str = "both",
    sent_aligned: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    nom_cache, datasets, preco_n, single_ctx = _data_cfg(subset)
    docs = _filter_docs(build_docs(device=device, datasets=datasets, nom_cache=nom_cache, preco_n=preco_n), subset)
    for d in docs:
        d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    if sent_aligned:  # set d["win_ids"]/d["win_chunks"] so analysis uses the same windows as the head
        _apply_sentence_windows(docs, window)
    test_idx = [i for i, d in enumerate(docs) if d["split"] == "test" and d["name"].startswith("conll2012/")]
    _build_lexical([docs[i] for i in test_idx])
    ctx_dir = _win_names(window, subset, "both", sent_aligned=sent_aligned)[0]
    for i in test_idx:
        p = ctx_dir / f"{i:06d}.npy"
        if p.exists():
            docs[i]["ctx_vecs"] = np.load(p).astype(np.float16)

    results = _run_error_channels(docs, test_idx, window, subset, sent_aligned, device)
    chans = list(results)
    print("\n=== Stage A within-window LINK recall by type, per channel (CoNLL-2012 test) ===")
    print(f"  {'type':6s} " + "  ".join(f"{ch:>8s}" for ch in chans))
    for t in ("PROPN", "NOUN", "PRON"):
        cells = [f"{100 * results[ch]['rec'][t][0] / max(results[ch]['rec'][t][1], 1):8.1f}" for ch in chans]
        print(f"  {t:6s} " + "  ".join(cells))

    print("\n=== Stage A within-window LINK precision by type, per channel (correct / links made) ===")
    print(f"  {'type':6s} " + "  ".join(f"{ch:>8s}" for ch in chans))
    for t in ("PROPN", "NOUN", "PRON"):
        cells = [f"{100 * results[ch]['precs'][t][0] / max(results[ch]['precs'][t][1], 1):8.1f}" for ch in chans]
        print(f"  {t:6s} " + "  ".join(cells))

    _print_both_channel_detail(results.get("both"))

    ss, ts, sp, tp = _sentence_split_stats(docs, test_idx, window)
    print("\n=== windowing vs sentence borders ===")
    print(f"sentences split across a window boundary:               {ss}/{ts} ({100 * ss / max(ts, 1):.1f}%)")
    print(f"gold intra-sentence coref pairs split into diff windows: {sp}/{tp} ({100 * sp / max(tp, 1):.1f}%)")


# ── Mention detection (predicted spans) ─────────────────────────────────────────


def _detector_windows(d: dict, window: int, device: str) -> list:
    # -> list of (toks (T,CTX) float, gold {(i,j) local inclusive}, cs) per encoding window.
    # Sentence-aligned windows never split a mention, so window-local detection is complete.
    full = torch.from_numpy(d["full_ctx"]).to(device).float()
    n = full.shape[0]
    chunks = d.get("win_chunks") or [(s, min(s + window, n)) for s in range(0, n, window)]
    out = []
    for cs, ce in chunks:
        gold = {(int(ss) - cs, int(se) - cs) for ss, se in d["span_sub"] if cs <= ss and se < ce}
        out.append((full[cs:ce], gold, cs))
    return out


def detector_loss(detector, windows: list, device: str, neg_ratio: int = 5) -> torch.Tensor | None:
    # BCE over candidate spans with HARD-negative mining (no pos_weight): per window, all gold spans
    # plus the neg_ratio*n_pos highest-scoring NON-gold candidates. Random negatives are almost all
    # trivially-junk spans, so the model never learns to reject the hard look-alikes (-> low
    # precision); mining the top-scoring negatives forces it to push down its confident mistakes.
    losses = []
    for toks, gold, _cs in windows:
        T = toks.shape[0]
        if T < 1:
            continue
        ii, jj, logit = detector(toks)
        if logit.numel() == 0:
            continue
        goldmat = torch.zeros(T, detector.max_span, device=device)
        for a, b in gold:
            if 0 <= b - a < detector.max_span:
                goldmat[a, b - a] = 1.0
        labels = goldmat[ii, jj - ii]  # (P,)
        pos = labels.nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            continue
        neg = (labels == 0).nonzero(as_tuple=True)[0]
        k = min(neg_ratio * pos.numel(), neg.numel())
        hard = neg[torch.topk(logit[neg].detach(), k).indices]  # highest-scoring non-gold spans
        sel = torch.cat([pos, hard])
        losses.append(F.binary_cross_entropy_with_logits(logit[sel], labels[sel]))
    return torch.stack(losses).mean() if losses else None


def _detector_epoch(detector, docs, idx, optimizer, device, doc_bs, window) -> float:
    # Per-WINDOW backward with gradient accumulation: the candidate graph (T*max_span pairs through
    # the bilinear) is large, so accumulating a whole batch's windows before backward OOMs the 8 GB
    # GPU. Backward per window frees each graph immediately; one optimizer step per doc_bs docs.
    order = list(idx)
    random.shuffle(order)
    total, n = 0.0, 0
    for s in tqdm(range(0, len(order), doc_bs), desc="train"):
        batch = order[s : s + doc_bs]
        windows = [w for i in batch for w in _detector_windows(docs[i], window, device)]
        if not windows:
            continue
        optimizer.zero_grad()
        batch_loss = 0.0
        for w in windows:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                loss = detector_loss(detector, [w], device)
            if loss is None:
                continue
            (loss / len(windows)).backward()
            batch_loss += loss.item() / len(windows)
        torch.nn.utils.clip_grad_norm_(list(detector.parameters()), 1.0)
        optimizer.step()
        total += batch_loss * len(batch)
        n += len(batch)
    return total / max(n, 1)


def _scored_candidates(detector, docs, idx, window, device) -> tuple:
    # one forward pass over the split: (sigmoid scores, is_gold flags, total gold count). Gold
    # spans longer than max_span aren't candidates -> never matched -> counted in n_gold (true recall).
    scores, golds, n_gold = [], [], 0
    with torch.inference_mode():
        for i in idx:
            d = docs[i]
            gold_global = {(int(s), int(e)) for s, e in d["span_sub"]}
            n_gold += len(gold_global)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                for toks, _g, cs in _detector_windows(d, window, device):
                    ii, jj, logit = detector(toks)
                    if logit.numel() == 0:
                        continue
                    sig = torch.sigmoid(logit.float()).cpu().numpy()
                    for a, b, sc in zip(ii.tolist(), jj.tolist(), sig, strict=True):
                        scores.append(sc)
                        golds.append((cs + a, cs + b) in gold_global)
    return np.asarray(scores), np.asarray(golds, dtype=bool), n_gold


def _sweep_threshold(scores, golds, n_gold, grid) -> tuple:
    # pick the decision threshold maximizing span-detection F1; returns (f1, p, r, thr)
    best = (-1.0, 0.0, 0.0, 0.5)
    for t in grid:
        keep = scores > t
        tp = int(golds[keep].sum())
        fp = int(keep.sum()) - tp
        fn = n_gold - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        if f > best[0]:
            best = (f, p, r, float(t))
    return best


def train_mention_detector(
    window: int = 256,
    sent_aligned: bool = True,
    max_epochs: int = 40,
    patience: int = 5,
    doc_bs: int = 8,
    head_lr: float = 1e-3,
    dropout: float = 0.2,
    weight_decay: float = 0.1,
    max_span: int = 30,
    proj: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(
        f"\nTraining mention detector (conll2012 only, window={window}, sent_aligned={sent_aligned}, "
        f"dropout={dropout}, patience={patience}, max_span={max_span}, 1:1 neg sampling)"
    )
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    nom_cache, datasets, preco_n, _ = _data_cfg("conll")
    docs = _filter_docs(build_docs(device=device, datasets=datasets, nom_cache=nom_cache, preco_n=preco_n), "conll")
    for d in docs:
        d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    if sent_aligned:
        _apply_sentence_windows(docs, window)
    tokenizer = load_tokenizer()
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    encoder = ContextEncoder().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    cache_dir = MODELS_DIR / f"fullctx_conll_k{window}{'_sent' if sent_aligned else ''}{ENCODER_TAG}"
    precompute_full_ctx(encoder, docs, cls_id, sep_id, device, cache_dir, window)
    del encoder
    if device != "cpu":
        torch.cuda.empty_cache()

    train_i = [i for i, d in enumerate(docs) if d["split"] == "train"]
    val_i = [i for i, d in enumerate(docs) if d["split"] == "validation"]
    test_i = [i for i, d in enumerate(docs) if d["split"] == "test"]
    n_long = sum(1 for d in docs for s, e in d["span_sub"] if e - s >= max_span)
    n_all = sum(len(d["span_sub"]) for d in docs)
    print(f"train {len(train_i)} | val {len(val_i)} | test {len(test_i)} docs | "
          f"gold mentions >= max_span (unreachable): {n_long}/{n_all} ({100 * n_long / max(n_all, 1):.2f}%)")

    detector = MentionDetector(proj=proj, dropout=dropout, max_span=max_span).to(device)
    optimizer = optim.AdamW(list(detector.parameters()), lr=head_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=head_lr * 0.1)
    print(f"detector params: {sum(p.numel() for p in detector.parameters()):,}")
    grid = np.arange(0.05, 0.96, 0.05)
    ckpt = MODELS_DIR / f"mention_detector_k{window}{'_sent' if sent_aligned else ''}{ENCODER_TAG}.pt"
    best_f1, best_thr, patience_ctr = -1.0, 0.5, 0

    for epoch in range(max_epochs):
        print(f"\n=== Detector Epoch {epoch + 1}/{max_epochs} ===")
        detector.train()
        tr_loss = _detector_epoch(detector, docs, train_i, optimizer, device, doc_bs, window)
        scheduler.step()
        detector.eval()
        sc, go, ng = _scored_candidates(detector, docs, val_i, window, device)
        f1, p, r, thr = _sweep_threshold(sc, go, ng, grid)
        print(f"train loss {tr_loss:.4f} | val span P {100 * p:.2f} R {100 * r:.2f} F1 {100 * f1:.2f} @thr {thr:.2f}")
        if f1 > best_f1 + 1e-4:
            best_f1, best_thr, patience_ctr = f1, thr, 0
            torch.save({"detector": detector.state_dict(), "threshold": thr, "max_span": max_span, "proj": proj}, ckpt)
            print(f"✓ saved (val F1 {100 * f1:.2f})")
        else:
            patience_ctr += 1
            print(f"no improvement {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"⊘ early stop. best val F1 {100 * best_f1:.2f}")
                break

    detector.load_state_dict(torch.load(ckpt, map_location=device)["detector"])
    detector.eval()
    sc, go, ng = _scored_candidates(detector, docs, test_i, window, device)
    keep = sc > best_thr
    tp = int(go[keep].sum())
    fp = int(keep.sum()) - tp
    fn = ng - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    print(f"\n=== Detector TEST (conll2012) @thr {best_thr:.2f} ===")
    print(f"span P {100 * p:.2f} | R {100 * r:.2f} | F1 {100 * f1:.2f}  (tp {tp} fp {fp} fn {fn}, gold {ng})")


# ── BIO mention tagging (fine-tuned RoBERTa, stacked by nesting depth) ───────────


def _bio_doc_labels(d: dict, n_layers: int) -> np.ndarray:
    # (n_layers, n_subtok) class labels {O=0, B=1, I=2}. Each gold mention is written to the head
    # matching its containment depth (0=flat/outermost). CoNLL nesting is clean containment, so
    # same-depth mentions never overlap -> no per-head conflict. Mentions deeper than n_layers
    # (depth>=n_layers, ~0.06% at L=3) are dropped rather than clamped (clamping would collide).
    n = len(d["content_ids"])
    lab = np.zeros((n_layers, n), dtype=np.int64)
    spans = [(int(s), int(e)) for s, e in d["span_sub"]]
    for k, (ss, se) in enumerate(spans):
        depth = sum(
            1 for j, (os, oe) in enumerate(spans) if j != k and os <= ss and se <= oe and (os, oe) != (ss, se)
        )
        if depth >= n_layers:
            continue
        lab[depth, ss] = 1
        if se > ss:
            lab[depth, ss + 1 : se + 1] = 2
    return lab


def _bio_windows(d: dict, window: int) -> list[tuple[int, int]]:
    # sentence-aligned encoding chunks (never split a mention) if present, else fixed-K blocks
    n = len(d["content_ids"])
    return d.get("win_chunks") or [(s, min(s + window, n)) for s in range(0, n, window)]


def _bio_pack(content_slices: list, tokenizer, device: str) -> tuple:
    # list of content-id arrays -> padded (B, W) input_ids + attention_mask ([cls] content [sep])
    cls, sep, pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    W = max(len(c) for c in content_slices) + 2
    ids = np.full((len(content_slices), W), pad, dtype=np.int64)
    mask = np.zeros((len(content_slices), W), dtype=np.int64)
    for k, c in enumerate(content_slices):
        ids[k, 0] = cls
        ids[k, 1 : 1 + len(c)] = c
        ids[k, 1 + len(c)] = sep
        mask[k, : 2 + len(c)] = 1
    return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)


def _bio_loss(model, content_slices: list, label_slices: list, tokenizer, device: str) -> torch.Tensor:
    # per-token cross-entropy summed over the L depth heads, content tokens only (skip cls/sep/pad)
    ids, mask = _bio_pack(content_slices, tokenizer, device)
    logits = model(ids, mask)  # (L, B, W, 3)
    losses = []
    for k, (c, lab) in enumerate(zip(content_slices, label_slices, strict=True)):
        w = len(c)
        lg = logits[:, k, 1 : 1 + w, :].reshape(-1, 3)
        tg = torch.from_numpy(lab).to(device).reshape(-1)
        losses.append(F.cross_entropy(lg, tg))
    return torch.stack(losses).mean()


def _decode_bio(labels: np.ndarray) -> list[tuple[int, int]]:
    # greedy B…I run decode -> (start, end) inclusive spans. An orphan I (no preceding B) opens a span.
    spans, start = [], None
    for t, l in enumerate(labels):
        if l == 1:
            if start is not None:
                spans.append((start, t - 1))
            start = t
        elif l == 2:
            if start is None:
                start = t
        else:
            if start is not None:
                spans.append((start, t - 1))
                start = None
    if start is not None:
        spans.append((start, len(labels) - 1))
    return spans


def _span_overlap(a1: int, b1: int, a2: int, b2: int) -> bool:
    # inclusive-endpoint span intersection
    return a1 <= b2 and a2 <= b1


def _gold_depths(spans: list) -> list:
    # containment depth (# of mentions strictly containing it) per gold span
    return [
        sum(1 for j, (os, oe) in enumerate(spans) if j != k and os <= ss and se <= oe and (os, oe) != (ss, se))
        for k, (ss, se) in enumerate(spans)
    ]


def _bio_predict_doc(model, d: dict, tokenizer, device, window, win_bs, n_layers) -> set:
    # argmax decode every depth head over every window, union to a set of (start, end) inclusive spans
    cids = d["content_ids"]
    chunks = _bio_windows(d, window)
    pred = set()
    for t in range(0, len(chunks), win_bs):
        sub = chunks[t : t + win_bs]
        slices = [np.asarray(cids[cs:ce], dtype=np.int64) for cs, ce in sub]
        ids, mask = _bio_pack(slices, tokenizer, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            cls = model(ids, mask).argmax(-1).cpu().numpy()  # (L, B, W)
        for k, (cs, ce) in enumerate(sub):
            w = ce - cs
            for layer in range(n_layers):
                for a, b in _decode_bio(cls[layer, k, 1 : 1 + w]):
                    pred.add((cs + a, cs + b))
    return pred


def _bio_eval(model, docs, idx, tokenizer, device, window, win_bs, n_layers) -> dict:
    # Span-detection diagnostics under three match criteria, plus per-depth recall and an FP split.
    #  exact   : same (start, end) subtoken span
    #  overlap : predicted span intersects a gold span (boundary-agnostic — is the mention there at all?)
    #  end     : predicted end == gold end (head-proxy: English NPs are usually head-final)
    # FP split (under exact): boundary = the FP overlaps some gold mention (a near-miss); disjoint = it
    # overlaps no gold mention (a singleton NP the coref layer omitted, or genuine garbage — Tier 2 splits these).
    n_gold = n_pred = 0
    ex_g = ov_g = end_g = ex_p = ov_p = end_p = 0
    fp_boundary = fp_disjoint = 0
    dgold = [0] * n_layers
    drec = [0] * n_layers
    with torch.inference_mode():
        for i in idx:
            d = docs[i]
            gold = [(int(s), int(e)) for s, e in d["span_sub"]]
            gset = set(gold)
            depths = _gold_depths(gold)
            pred = _bio_predict_doc(model, d, tokenizer, device, window, win_bs, n_layers)
            n_gold += len(gold)
            n_pred += len(pred)
            for (gs, ge), dep in zip(gold, depths, strict=True):
                hit = (gs, ge) in pred
                ex_g += hit
                ov_g += any(_span_overlap(gs, ge, ps, pe) for ps, pe in pred)
                end_g += any(pe == ge for _, pe in pred)
                if dep < n_layers:
                    dgold[dep] += 1
                    drec[dep] += hit
            for ps, pe in pred:
                exact = (ps, pe) in gset
                ov = any(_span_overlap(ps, pe, gs, ge) for gs, ge in gold)
                ex_p += exact
                ov_p += ov
                end_p += any(pe == ge for _, ge in gold)
                if not exact:
                    fp_boundary += ov
                    fp_disjoint += not ov

    def prf(tp_g: int, tp_p: int) -> tuple:
        r = tp_g / max(n_gold, 1)
        p = tp_p / max(n_pred, 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)

    return {
        "exact": prf(ex_g, ex_p),
        "overlap": prf(ov_g, ov_p),
        "end": prf(end_g, end_p),
        "n_gold": n_gold,
        "n_pred": n_pred,
        "fp_boundary": fp_boundary,
        "fp_disjoint": fp_disjoint,
        "depth_recall": [(drec[k], dgold[k]) for k in range(n_layers)],
    }


def _fmt_bio_eval(m: dict) -> str:
    ex, ov, en = m["exact"], m["overlap"], m["end"]
    depth = " ".join(f"d{k}={100 * r / max(g, 1):.1f}%({g})" for k, (r, g) in enumerate(m["depth_recall"]))
    return (
        f"exact P{100 * ex[0]:.1f} R{100 * ex[1]:.1f} F{100 * ex[2]:.1f} | "
        f"overlap P{100 * ov[0]:.1f} R{100 * ov[1]:.1f} F{100 * ov[2]:.1f} | "
        f"endR {100 * en[1]:.1f} | depthR {depth} | "
        f"FP boundary {m['fp_boundary']} disjoint {m['fp_disjoint']} (pred {m['n_pred']}, gold {m['n_gold']})"
    )


def _plausible_spans_for_docs(docs, idx, tokenizer) -> dict:
    # our doc index -> set of plausible-mention (start, end) inclusive SUBTOKEN spans, read from the
    # cached spaCy DocBins (token-aligned 1:1 to dataset words, verified), so no parser model is
    # needed. "Plausible mention" = any noun chunk, named entity, or pronoun — what a syntactic
    # detector would propose — mapped into the same subtoken coordinate as gold/pred.
    vocab = spacy.blank("en").vocab
    by_words = {tuple(w for s in docs[i]["sentences"] for w in s): i for i in idx}
    out: dict = {}
    for split in sorted({docs[i]["split"] for i in idx}):
        ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
        db = spacy.tokens.DocBin().from_disk(SPACY_TRF_DIR / f"conll2012_{split}.spacy")
        for sample, sdoc in zip(ds, db.get_docs(vocab), strict=False):
            words = [w for s in sample["sentences"] for w in s]
            di = by_words.get(tuple(words))
            if di is None:
                continue
            content_ids, w2s = _word_to_subtok(words, tokenizer)
            last = len(content_ids) - 1
            wspans = [(c.start, c.end) for c in sdoc.noun_chunks]
            wspans += [(e.start, e.end) for e in sdoc.ents]
            wspans += [(t.i, t.i + 1) for t in sdoc if t.pos_ == "PRON"]
            spans = set()
            for ws, we in wspans:
                ss = w2s.get(ws, min(ws, last))
                se = min(max(w2s.get(we, len(content_ids)) - 1, ss), last)
                spans.add((ss, se))
            out[di] = spans
    return out


def _bio_singleton_breakdown(model, docs, idx, tokenizer, device, window, win_bs, n_layers) -> None:
    # Of the disjoint FPs (predicted spans overlapping NO gold mention), how many are valid noun
    # phrases (singletons the coref layer didn't annotate) vs. genuine garbage. This is the test that
    # confirms whether the precision loss is the OntoNotes singleton convention — Stage A's problem —
    # rather than the tagger emitting junk.
    plaus_by_doc = _plausible_spans_for_docs(docs, idx, tokenizer)
    disjoint = plausible = garbage = 0
    with torch.inference_mode():
        for i in tqdm(idx, desc="singleton breakdown"):
            d = docs[i]
            gold = [(int(s), int(e)) for s, e in d["span_sub"]]
            gset = set(gold)
            pred = _bio_predict_doc(model, d, tokenizer, device, window, win_bs, n_layers)
            disj = [
                (ps, pe)
                for ps, pe in pred
                if (ps, pe) not in gset and not any(_span_overlap(ps, pe, gs, ge) for gs, ge in gold)
            ]
            plaus = plaus_by_doc.get(i, set())
            for ps, pe in disj:
                disjoint += 1
                if any(_span_overlap(ps, pe, qs, qe) for qs, qe in plaus):
                    plausible += 1
                else:
                    garbage += 1
    print(
        f"disjoint FPs: {disjoint} | valid NP (likely singleton, Stage A handles): {plausible} "
        f"({100 * plausible / max(disjoint, 1):.1f}%) | garbage: {garbage} ({100 * garbage / max(disjoint, 1):.1f}%)"
    )


def _bio_epoch(model, docs, idx, optimizer, tokenizer, device, doc_bs, win_bs, window, n_layers) -> float:
    # one optimizer step per doc_bs docs; windows forwarded in win_bs sub-batches with grad
    # accumulation (fine-tuning roberta-large over ~512-token windows is the 8 GB constraint).
    order = list(idx)
    random.shuffle(order)
    total, n = 0.0, 0
    for s in tqdm(range(0, len(order), doc_bs), desc="train"):
        batch = order[s : s + doc_bs]
        cslices, lslices = [], []
        for i in batch:
            d = docs[i]
            lab = _bio_doc_labels(d, n_layers)
            cids = d["content_ids"]
            for cs, ce in _bio_windows(d, window):
                cslices.append(np.asarray(cids[cs:ce], dtype=np.int64))
                lslices.append(lab[:, cs:ce])
        if not cslices:
            continue
        optimizer.zero_grad()
        bl = 0.0
        for t in range(0, len(cslices), win_bs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                loss = _bio_loss(model, cslices[t : t + win_bs], lslices[t : t + win_bs], tokenizer, device)
            (loss * len(cslices[t : t + win_bs]) / len(cslices)).backward()
            bl += loss.item() * len(cslices[t : t + win_bs]) / len(cslices)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += bl * len(batch)
        n += len(batch)
    return total / max(n, 1)


def train_bio_tagger(
    window: int = 256,
    sent_aligned: bool = True,
    n_layers: int = 3,
    max_epochs: int = 20,
    patience: int = 4,
    doc_bs: int = 4,
    win_bs: int = 4,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    dropout: float = 0.2,
    weight_decay: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(
        f"\nTraining BIO mention tagger (conll2012 only, window={window}, sent_aligned={sent_aligned}, "
        f"L={n_layers} depth heads, fine-tuned {BACKBONE}, backbone_lr={backbone_lr})"
    )
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    nom_cache, datasets, preco_n, _ = _data_cfg("conll")
    docs = _filter_docs(build_docs(device=device, datasets=datasets, nom_cache=nom_cache, preco_n=preco_n), "conll")
    for d in docs:
        d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    if sent_aligned:
        _apply_sentence_windows(docs, window)
    tokenizer = load_tokenizer()

    train_i = [i for i, d in enumerate(docs) if d["split"] == "train"]
    val_i = [i for i, d in enumerate(docs) if d["split"] == "validation"]
    test_i = [i for i, d in enumerate(docs) if d["split"] == "test"]
    n_drop = sum(
        1
        for d in docs
        for k, (ss, se) in enumerate([(int(s), int(e)) for s, e in d["span_sub"]])
        if sum(
            1
            for j, (os, oe) in enumerate([(int(s), int(e)) for s, e in d["span_sub"]])
            if j != k and os <= ss and se <= oe and (os, oe) != (ss, se)
        )
        >= n_layers
    )
    n_all = sum(len(d["span_sub"]) for d in docs)
    print(
        f"train {len(train_i)} | val {len(val_i)} | test {len(test_i)} docs | "
        f"gold mentions deeper than L={n_layers} (dropped): {n_drop}/{n_all} ({100 * n_drop / max(n_all, 1):.2f}%)"
    )

    model = BIOTagger(n_layers=n_layers, dropout=dropout).to(device)
    optimizer = optim.AdamW(
        [
            {"params": model.roberta.parameters(), "lr": backbone_lr},
            {"params": model.heads.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=backbone_lr * 0.1)
    print(f"tagger params: {sum(p.numel() for p in model.parameters()):,} (backbone fine-tuned)")
    ckpt = MODELS_DIR / f"bio_tagger_k{window}_L{n_layers}{'_sent' if sent_aligned else ''}{ENCODER_TAG}.pt"
    best_f1, patience_ctr = -1.0, 0

    for epoch in range(max_epochs):
        print(f"\n=== BIO Epoch {epoch + 1}/{max_epochs} ===")
        model.train()
        tr_loss = _bio_epoch(model, docs, train_i, optimizer, tokenizer, device, doc_bs, win_bs, window, n_layers)
        scheduler.step()
        model.eval()
        m = _bio_eval(model, docs, val_i, tokenizer, device, window, win_bs, n_layers)
        f1 = m["exact"][2]
        print(f"train loss {tr_loss:.4f} | val {_fmt_bio_eval(m)}")
        if f1 > best_f1 + 1e-4:
            best_f1, patience_ctr = f1, 0
            torch.save({"model": model.state_dict(), "n_layers": n_layers}, ckpt)
            print(f"✓ saved (val exact F1 {100 * f1:.2f})")
        else:
            patience_ctr += 1
            print(f"no improvement {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"⊘ early stop. best val F1 {100 * best_f1:.2f}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    m = _bio_eval(model, docs, test_i, tokenizer, device, window, win_bs, n_layers)
    print("\n=== BIO tagger TEST (conll2012) ===")
    print(_fmt_bio_eval(m))
    # Tier 2 — prove what the disjoint FPs are: valid NP (singleton the coref layer omitted) vs garbage.
    _bio_singleton_breakdown(model, docs, test_i, tokenizer, device, window, win_bs, n_layers)
