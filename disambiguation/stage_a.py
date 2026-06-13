import datetime
import json
import pickle
import random
from collections import Counter

import numpy as np
import torch
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
    build_docs,
    precompute_span_ctx,
)
from disambiguation.paths import MODELS_DIR, TENSORBOARD_DIR
from disambiguation.stage2_context_encoder import (
    CONTENT,
    AntecedentScorer,
    ContextEncoder,
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

    neg = torch.finfo(g.dtype).min
    scores = torch.zeros(b, max_m, max_m, device=device, dtype=g.dtype)
    scores[wid, li, lj] = pair
    ar = torch.arange(max_m, device=device)
    size_t = torch.tensor(sizes, device=device)
    valid = ar.unsqueeze(0) < size_t.unsqueeze(1)  # (B, maxM)
    ante = (ar.unsqueeze(1) > ar.unsqueeze(0)).unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)  # [b,i,j]: j<i
    cid = torch.full((b, max_m), -1, device=device, dtype=torch.long)
    for w, win in enumerate(windows):
        cid[w, : sizes[w]] = win[2]
    null = scorer.null_bias
    null_col = null.view(1, 1, 1).expand(b, max_m, 1)
    denom = torch.logsumexp(torch.cat([null_col, scores.masked_fill(~ante, neg)], dim=2), dim=2)  # (B, maxM)
    gold = (cid.unsqueeze(2) == cid.unsqueeze(1)) & ante  # [b,i,j]: same cluster, j<i
    has_gold = gold.any(dim=2)
    num = torch.where(has_gold, torch.logsumexp(scores.masked_fill(~gold, neg), dim=2), null.squeeze().expand(b, max_m))
    per_ment = denom - num  # (B, maxM)
    loss_mask = valid & (ar.unsqueeze(0) >= 1)
    w_loss = (per_ment * loss_mask).sum(1) / loss_mask.sum(1).clamp(min=1)  # (B,) per-window mean
    wt = torch.tensor([w[3] for w in windows], device=device, dtype=w_loss.dtype)
    return (w_loss * wt).sum() / wt.sum()


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
            loss = stage_a_batched_loss(scorer, windows, device)
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(
        f"\nBuilding Stage 2 nominal data... "
        f"(window={window}, subset={subset}, channel={channel}, sent_aligned={sent_aligned}, "
        f"raw={raw}, hidden={hidden}, use_distance={use_distance})"
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
