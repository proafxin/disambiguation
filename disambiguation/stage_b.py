import random

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from disambiguation.conll_scorer import conll_f1, conll_f1_by_type, write_conll
from disambiguation.data import (
    _PRON_AGR,
    _apply_sentence_windows,
    _build_lexical,
    _content,
    _data_cfg,
    _filter_docs,
    _is_subseq,
    _key_clusters,
    _train_idx,
    _win_names,
    build_docs,
    load_span_ctx_single,
)
from disambiguation.paths import MODELS_DIR
from disambiguation.stage2_context_encoder import (
    CONTENT,
    AntecedentScorer,
    ClusterGNN,
    _uf_find,
    antecedent_mll_loss,
)
from disambiguation.stage_a import precompute_stage_a_clusters


def _doc_cluster_nodes(d: dict, per_window: dict[int, dict]) -> tuple[list[np.ndarray], list[int], list[int]]:
    # Flatten Stage A clusters into an ordered node list for the GNN quotient graph:
    # returns (member_idx, win_ids, gold) where member_idx[k] are the global mention indices
    # of node k, win_ids[k] its window index, gold[k] its majority gold entity id. Nodes are
    # ordered by first-mention subtoken position (discourse order) so antecedent ranking is causal.
    cid, tok_pos = d["cluster_id"], d["tok_pos"]
    nodes = []
    for w in sorted(per_window):
        gi = per_window[w]["global_idx"]
        for cl in per_window[w]["clusters"]:
            members = gi[np.array(cl)]
            nodes.append((members, int(w), int(np.bincount(cid[members]).argmax()), int(tok_pos[members].min())))
    nodes.sort(key=lambda n: n[3])
    return [n[0] for n in nodes], [n[1] for n in nodes], [n[2] for n in nodes]


def _gnn_node_tensors(d: dict, member_idx: list[np.ndarray], device: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    ctx_all, bge_all = d["ctx_vecs"], d["mention_bge"]
    return [
        (torch.from_numpy(ctx_all[m]).to(device).float(), torch.from_numpy(bge_all[m]).to(device).float())
        for m in member_idx
    ]


def _gnn_lex_matrix(member_idx: list[np.ndarray], mention_tokens: list, idf: dict) -> np.ndarray:
    # (C, C, 3) node-pair lexical features [IDF-Jaccard, containment, exact-match]. Each node's
    # member surfaces are folded into ONE token set before any pairwise work, so the cost is
    # O(M + C²) (build sets once, then C² set-overlaps) — never mention-pairs.
    surf = [[mention_tokens[m] for m in members] for members in member_idx]
    tset = [{t for mt in s for t in mt} for s in surf]
    mass = [sum(idf.get(t, 0.0) for t in s) for s in tset]
    cont = [[mt for mt in s if _content(mt)] for s in surf]
    C = len(member_idx)
    out = np.zeros((C, C, 3), dtype=np.float32)
    for i in range(C):
        ci = set(cont[i])
        for j in range(C):
            inter = tset[i] & tset[j]
            im = sum(idf.get(t, 0.0) for t in inter)
            um = mass[i] + mass[j] - im
            out[i, j, 0] = im / um if um > 0 else 0.0
            out[i, j, 2] = 1.0 if ci & set(cont[j]) else 0.0
            out[i, j, 1] = 1.0 if any(_is_subseq(a, b) or _is_subseq(b, a) for a in cont[i] for b in cont[j]) else 0.0
    return out


def _cluster_agr(members: np.ndarray, mention_tokens: list) -> tuple[int, int, int]:
    # (number, gender, person) profile from a cluster's pronoun members; 0 = unknown, and
    # conflicting known values within a cluster collapse to unknown.
    nums, gens, pers = set(), set(), set()
    for m in members:
        mt = mention_tokens[m]
        if len(mt) == 1 and mt[0] in _PRON_AGR:
            n, g, p = _PRON_AGR[mt[0]]
            if n:
                nums.add(n)
            if g:
                gens.add(g)
            if p:
                pers.add(p)

    def pick(s: set) -> int:
        return next(iter(s)) if len(s) == 1 else 0

    return pick(nums), pick(gens), pick(pers)


def _gnn_agr_matrix(member_idx: list[np.ndarray], mention_tokens: list) -> np.ndarray:
    # (C, C, 6): for each of [number, gender, person] a (match, clash) indicator between the two
    # clusters' profiles — match=both known and equal, clash=both known and differ. O(C²).
    prof = np.array([_cluster_agr(m, mention_tokens) for m in member_idx], dtype=np.int64)  # (C, 3)
    C = len(member_idx)
    out = np.zeros((C, C, 6), dtype=np.float32)
    for a in range(3):
        v = prof[:, a]
        both = (v != 0)[:, None] & (v != 0)[None, :]
        eq = v[:, None] == v[None, :]
        out[:, :, 2 * a] = (both & eq).astype(np.float32)
        out[:, :, 2 * a + 1] = (both & ~eq).astype(np.float32)
    return out


def _gnn_agr_for_doc(gnn: ClusterGNN, d: dict, member_idx: list[np.ndarray], device: str) -> torch.Tensor | None:
    if not getattr(gnn, "use_agreement", False):
        return None
    if "mention_tokens" not in d:
        _build_lexical([d])
    cache = d.get("_gnn_agr")
    if cache is None or cache.shape[0] != len(member_idx):
        cache = _gnn_agr_matrix(member_idx, d["mention_tokens"])
        d["_gnn_agr"] = cache
    return torch.from_numpy(cache).to(device)


def _gnn_lex_for_doc(gnn: ClusterGNN, d: dict, member_idx: list[np.ndarray], device: str) -> torch.Tensor | None:
    # constant per doc (nodes are fixed Stage A clusters) -> compute once and cache on d
    if not getattr(gnn, "use_lexical", False):
        return None
    if "mention_tokens" not in d:
        _build_lexical([d])
    cache = d.get("_gnn_lex")
    if cache is None or cache.shape[0] != len(member_idx):
        cache = _gnn_lex_matrix(member_idx, d["mention_tokens"], d["mention_idf"])
        d["_gnn_lex"] = cache
    return torch.from_numpy(cache).to(device)


def _subsample_neg_candidates(ante: torch.Tensor, gold_id: torch.Tensor, neg_ratio: float) -> torch.Tensor:
    # Per cluster keep all gold antecedents + up to neg_ratio*max(n_gold,1) random negatives
    # (floor 4), so each softmax sees a balanced candidate set instead of all ~20 distractors.
    goldmat = (gold_id.unsqueeze(0) == gold_id.unsqueeze(1)) & ante
    out = goldmat.clone()
    for i in range(ante.shape[0]):
        cap = max(int(neg_ratio * max(int(goldmat[i].sum()), 1)), 4)
        negs = (ante[i] & ~goldmat[i]).nonzero(as_tuple=True)[0]
        sel = negs[torch.randperm(negs.numel(), device=ante.device)[:cap]] if negs.numel() > cap else negs
        out[i, sel] = True
    return out


def _gnn_doc_loss(
    gnn: ClusterGNN,
    d: dict,
    per_window: dict[int, dict],
    device: str,
    pos_weight: float = 1.0,
    neg_ratio: float | None = None,
) -> torch.Tensor | None:
    member_idx, win_ids, gold = _doc_cluster_nodes(d, per_window)
    if len(member_idx) < 2:
        return None
    clusters = _gnn_node_tensors(d, member_idx, device)
    lex = _gnn_lex_for_doc(gnn, d, member_idx, device)
    agr = _gnn_agr_for_doc(gnn, d, member_idx, device)
    scores, ante = gnn(clusters, torch.tensor(win_ids, device=device), lex, agr)
    gold_t = torch.tensor(gold, device=device)
    if neg_ratio is not None:
        ante = _subsample_neg_candidates(ante, gold_t, neg_ratio)
    return antecedent_mll_loss(scores, ante, gold_t, gnn.null_bias, pos_weight)


def _predict_full_doc_clusters_gnn(d: dict, per_window: dict[int, dict], gnn: ClusterGNN, device: str) -> list[list]:
    # Each cluster points to its single best earlier cluster (or null); the pointer forest is
    # the entity partition. Unmerged nodes that are themselves >=2 mentions stay as entities.
    member_idx, win_ids, _ = _doc_cluster_nodes(d, per_window)
    C = len(member_idx)
    if C == 0:
        return []
    if C == 1:
        spans = [d["spans"][m] for m in member_idx[0]]
        return [spans] if len(spans) >= 2 else []
    with torch.inference_mode():
        lex = _gnn_lex_for_doc(gnn, d, member_idx, device)
        agr = _gnn_agr_for_doc(gnn, d, member_idx, device)
        scores, ante = gnn(_gnn_node_tensors(d, member_idx, device), torch.tensor(win_ids, device=device), lex, agr)
    s, a = scores.float().cpu().numpy(), ante.cpu().numpy()
    nb = float(gnn.null_bias.item())
    parent = list(range(C))
    for i in range(1, C):
        cand = np.where(a[i])[0]
        if len(cand) == 0:
            continue
        j = int(cand[np.argmax(s[i, cand])])
        if s[i, j] > nb:
            parent[_uf_find(parent, i)] = _uf_find(parent, j)
    groups: dict[int, list[int]] = {}
    for i in range(C):
        groups.setdefault(_uf_find(parent, i), []).append(i)
    result = []
    for ns in groups.values():
        spans = [d["spans"][m] for ni in ns for m in member_idx[ni]]
        if len(spans) >= 2:
            result.append(spans)
    return result


def _gnn_merge_stats(d: dict, per_window: dict[int, dict], gnn: ClusterGNN, device: str) -> tuple[int, int, int]:
    # (merges accepted, gold-positive nodes, total nodes) — exposes the all-null collapse directly.
    member_idx, win_ids, gold = _doc_cluster_nodes(d, per_window)
    C = len(member_idx)
    if C < 2:
        return 0, 0, C
    with torch.inference_mode():
        lex = _gnn_lex_for_doc(gnn, d, member_idx, device)
        agr = _gnn_agr_for_doc(gnn, d, member_idx, device)
        sc, ante = gnn(_gnn_node_tensors(d, member_idx, device), torch.tensor(win_ids, device=device), lex, agr)
    s, a = sc.float().cpu().numpy(), ante.cpu().numpy()
    g = np.array(gold)
    goldmat = (g[:, None] == g[None, :]) & a
    nb = float(gnn.null_bias.item())
    merges = 0
    for i in range(C):
        c = np.where(a[i])[0]
        if len(c) and s[i, c[np.argmax(s[i, c])]] > nb:
            merges += 1
    return merges, int(goldmat.any(axis=1).sum()), C


def eval_stage_b(
    docs: list,
    stage_a_clusters: list[dict[int, list[list[int]]]],
    cluster_matcher: ClusterGNN,
    device: str,
    tag: str,
    type_breakdown: bool = False,
) -> dict:
    cluster_matcher.eval()
    key_docs, resp_docs = [], []
    g_merge = g_pos = g_nodes = 0
    for d, per_window in zip(docs, stage_a_clusters):
        if not per_window:
            continue
        key_docs.append((d["name"], d["sentences"], _key_clusters(d)))
        resp_docs.append(
            (d["name"], d["sentences"], _predict_full_doc_clusters_gnn(d, per_window, cluster_matcher, device))
        )
        m, p, n = _gnn_merge_stats(d, per_window, cluster_matcher, device)
        g_merge, g_pos, g_nodes = g_merge + m, g_pos + p, g_nodes + n
    key_path = MODELS_DIR / f"stageb_{tag}_key.conll"
    resp_path = MODELS_DIR / f"stageb_{tag}_resp.conll"
    write_conll(key_path, key_docs)
    write_conll(resp_path, resp_docs)
    result = conll_f1(key_path, resp_path)
    print(
        f"    [gnn:{tag}] merges accepted {g_merge} / gold-positive nodes {g_pos} "
        f"({g_nodes} nodes, null_bias {float(cluster_matcher.null_bias.item()):+.3f})"
    )
    if type_breakdown:
        result["by_type"] = conll_f1_by_type(key_docs, resp_docs, MODELS_DIR)
    return result


def train_stage_b(
    head_lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    doc_bs: int = 8,
    window: int = CONTENT,
    neg_ratio: float | None = None,
    subset: str = "all",
    dropout: float = 0.3,
    weight_decay: float = 0.1,
    channel: str = "both",
    pos_weight: float = 1.0,
    member_pool: str = "lse",
    lexical: bool = False,
    agreement: bool = False,
    sent_aligned: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_only: bool = False,
    force: bool = False,
) -> None:
    print(
        f"\nTraining Stage B GNN... (window={window}, subset={subset}, "
        f"member_pool={member_pool}, channel={channel}, sent_aligned={sent_aligned})"
    )
    ctx_dir, frozen_base, ckpt_b, _ = _win_names(window, subset, channel, sent_aligned=sent_aligned)
    ckpt_b = ckpt_b.replace(".pt", f"_gnn_{member_pool}{'_lex' if lexical else ''}{'_agr' if agreement else ''}.pt")
    if not eval_only and not force and (MODELS_DIR / ckpt_b).exists():
        print(f"✓ Stage B checkpoint {ckpt_b} exists; loading for eval instead of retraining (force=True to retrain).")
        eval_only = True
    nom_cache, datasets, preco_n, single_ctx = _data_cfg(subset)
    docs = build_docs(device=device, datasets=datasets, nom_cache=nom_cache, preco_n=preco_n)
    docs = _filter_docs(docs, subset)
    for d in docs:
        if "tok_pos" not in d:
            d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    if sent_aligned:  # reuse the exact windows Stage A used (matches the sent-aligned head + ctx)
        _apply_sentence_windows(docs, window)
    if lexical or agreement:
        _build_lexical(docs)  # mention tokens feed the gnn lexical and/or agreement channels

    stage_a = AntecedentScorer(channel=channel).to(device)
    stage_a.load_state_dict(torch.load(MODELS_DIR / f"{frozen_base}.pt", map_location=device)["scorer"])
    for p in stage_a.parameters():
        p.requires_grad_(False)
    stage_a.eval()

    if single_ctx:
        load_span_ctx_single(docs, ctx_dir)
    else:
        for i, d in enumerate(docs):
            p = ctx_dir / f"{i:06d}.npy"
            if p.exists() and "ctx_vecs" not in d:
                d["ctx_vecs"] = np.load(p).astype(np.float16)

    train_idx = _train_idx(docs, subset)
    train_docs = [docs[i] for i in train_idx]
    val_docs = [d for d in docs if d["split"] == "validation" and d["name"].startswith("conll2012/")]
    test_docs = [d for d in docs if d["split"] == "test" and d["name"].startswith("conll2012/")]

    # precompute Stage A clusters once — avoids re-running Stage A every epoch
    all_stage_a = precompute_stage_a_clusters(docs, stage_a, device, window, subset, channel, sent_aligned=sent_aligned)
    train_clusters = [all_stage_a[i] for i in train_idx]
    val_clusters = [
        all_stage_a[i] for i, d in enumerate(docs) if d["split"] == "validation" and d["name"].startswith("conll2012/")
    ]
    test_clusters = [
        all_stage_a[i] for i, d in enumerate(docs) if d["split"] == "test" and d["name"].startswith("conll2012/")
    ]

    all_train_pairs_count = sum(
        len(per_window) * (len(per_window) - 1) // 2 for per_window in train_clusters if per_window
    )
    print(f"Training cluster pairs per epoch: {all_train_pairs_count}")
    cluster_matcher = ClusterGNN(
        dropout=dropout, channel=channel, member_pool=member_pool, use_lexical=lexical, use_agreement=agreement
    ).to(device)
    if eval_only:
        cluster_matcher.load_state_dict(torch.load(MODELS_DIR / ckpt_b, map_location=device)["cluster_matcher"])
        cluster_matcher.eval()
        print("\n=== Stage B Test (eval only) ===")
        test_scores = eval_stage_b(test_docs, test_clusters, cluster_matcher, device, "test", type_breakdown=True)
        print(
            f"Test CoNLL {test_scores['CoNLL'] * 100:.2f} MUC {test_scores['muc'] * 100:.2f} "
            f"B3 {test_scores['bcub'] * 100:.2f} CEAFe {test_scores['ceafe'] * 100:.2f}"
        )
        return
    optimizer = optim.AdamW(cluster_matcher.parameters(), lr=head_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=head_lr * 0.1)
    ckpt_path = MODELS_DIR / ckpt_b
    best_val_f1, patience_ctr = -1.0, 0

    for epoch in range(max_epochs):
        print(f"\n=== Stage B (K={window}) Epoch {epoch + 1}/{max_epochs} ===")
        order = list(range(len(train_docs)))
        random.shuffle(order)
        cluster_matcher.train()
        total, n_pairs = 0.0, 0
        # Per-doc quotient graph; accumulate doc losses over doc_bs docs, one step (the C-node
        # forward is tiny, so gradient accumulation batches docs for free).
        pending_g: list[torch.Tensor] = []
        for bi in tqdm(order, desc="train_b"):
            per_window = train_clusters[bi]
            if not per_window:
                continue
            loss = _gnn_doc_loss(cluster_matcher, train_docs[bi], per_window, device, pos_weight, neg_ratio)
            if loss is None:
                continue
            pending_g.append(loss)
            n_pairs += 1
            if len(pending_g) >= doc_bs:
                optimizer.zero_grad()
                torch.stack(pending_g).mean().backward()
                torch.nn.utils.clip_grad_norm_(cluster_matcher.parameters(), 1.0)
                optimizer.step()
                total += float(sum(lo.item() for lo in pending_g))
                pending_g = []
        if pending_g:
            optimizer.zero_grad()
            torch.stack(pending_g).mean().backward()
            torch.nn.utils.clip_grad_norm_(cluster_matcher.parameters(), 1.0)
            optimizer.step()
            total += float(sum(lo.item() for lo in pending_g))
        scheduler.step()
        tr_loss = total / max(n_pairs, 1)

        val_sets = {
            "conll2012": (val_docs, val_clusters),
            "litbank": (
                [d for d in docs if d["split"] == "validation" and d["name"].startswith("litbank/")],
                [
                    all_stage_a[i]
                    for i, d in enumerate(docs)
                    if d["split"] == "validation" and d["name"].startswith("litbank/")
                ],
            ),
            "preco": (
                [d for d in docs if d["split"] == "validation" and d["name"].startswith("preco/")],
                [
                    all_stage_a[i]
                    for i, d in enumerate(docs)
                    if d["split"] == "validation" and d["name"].startswith("preco/")
                ],
            ),
            "corefud": (
                [d for d in docs if d["split"] == "validation" and d["name"].startswith("corefud/")],
                [
                    all_stage_a[i]
                    for i, d in enumerate(docs)
                    if d["split"] == "validation" and d["name"].startswith("corefud/")
                ],
            ),
        }
        val_scores_all = {
            ds: eval_stage_b(vdocs, vclusters, cluster_matcher, device, f"val_{ds}_epoch{epoch + 1}")
            for ds, (vdocs, vclusters) in val_sets.items()
            if vdocs
        }
        val_scores = val_scores_all["conll2012"]
        print(f"Loss: {tr_loss:.6f}")
        for ds, sc in val_scores_all.items():
            print(
                f"  {ds:10s} CoNLL {sc['CoNLL'] * 100:.2f} "
                f"(MUC {sc['muc'] * 100:.2f} B3 {sc['bcub'] * 100:.2f} CEAFe {sc['ceafe'] * 100:.2f})"
            )

        if val_scores["CoNLL"] > best_val_f1 + 1e-4:
            best_val_f1, patience_ctr = val_scores["CoNLL"], 0
            torch.save({"cluster_matcher": cluster_matcher.state_dict(), "best_val_f1": best_val_f1}, ckpt_path)
            print(f"✓ Best cluster matcher saved (val CoNLL F1 {best_val_f1 * 100:.2f})")
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"\n⊘ Early stopping. Best val CoNLL F1 {best_val_f1 * 100:.2f}")
                break

    print("\n✓ Stage B training complete")
    cluster_matcher.load_state_dict(torch.load(ckpt_path, map_location=device)["cluster_matcher"])
    cluster_matcher.eval()
    print("\n=== Stage B Test ===")
    test_scores = eval_stage_b(test_docs, test_clusters, cluster_matcher, device, "test", type_breakdown=True)
    print(
        f"CoNLL {test_scores['CoNLL'] * 100:.2f} | MUC {test_scores['muc'] * 100:.2f} | "
        f"B3 {test_scores['bcub'] * 100:.2f} | CEAFe {test_scores['ceafe'] * 100:.2f}"
    )
    if "by_type" in test_scores:
        for bucket, sc in sorted(test_scores["by_type"].items()):
            print(
                f"  {bucket:12s}  CoNLL {sc['CoNLL'] * 100:.2f}  MUC {sc['muc'] * 100:.2f}  "
                f"B3 {sc['bcub'] * 100:.2f}  CEAFe {sc['ceafe'] * 100:.2f}"
            )
