import json
import pickle
import random
import torch
import torch.optim as optim
import numpy as np
import spacy
import spacy.tokens
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from datasets import load_from_disk

from disambiguation.signals.abstract_features import N_CAT, N_CONT_FULL, N_PAIR_SYNT
from disambiguation.signals.stage1_intrasentence import (
    DepGraphTransformer, build_sentence_graph, mention_ranking_loss,
)
from disambiguation.signals.train_full import _clusters, _sent_lens

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
DATA_DIR = CACHE_DIR.parent / "data"
SPACY_TRF_DIR = DATA_DIR / "spacy_trf"
MODELS_DIR = CACHE_DIR / "models"
FEATURES_CACHE = DATA_DIR / "stage1_graph_features.pkl"

DATASET_CONFIG = [
    ("preco", ["train", "validation"]),
    ("litbank", ["train", "validation", "test"]),
    ("corefud", ["train", "validation"]),
    ("conll2012", ["train", "validation", "test"]),
]


def build_stage1_data() -> list:
    if FEATURES_CACHE.exists():
        with FEATURES_CACHE.open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            data = data[0] + data[1]
            with FEATURES_CACHE.open("wb") as f:
                pickle.dump(data, f)
        if data and len(data[0]) != 10:
            print("Cache format outdated — rebuilding...")
            FEATURES_CACHE.unlink()
            return build_stage1_data()
        # Validate inner array shapes match current feature spec.
        _, cat, cont, _, _, _, _, pair_synt, _, _ = data[0]
        if cat.shape[1] != N_CAT or cont.shape[1] != N_CONT_FULL or pair_synt.shape[2] != N_PAIR_SYNT:
            print("Feature dimensions changed — rebuilding cache...")
            FEATURES_CACHE.unlink()
            return build_stage1_data()
        print(f"Loaded cached features: {len(data)} rows")
        return data

    vocab = spacy.blank("en").vocab
    rows = []
    n_sents_total = 0
    n_filtered_total = 0

    for ds_name, splits in DATASET_CONFIG:
        ds_dict = load_from_disk(str(DATA_DIR / ds_name))

        for split_name in splits:
            if split_name not in ds_dict:
                continue

            split_ds = ds_dict[split_name]
            doc_bin = spacy.tokens.DocBin().from_disk(
                SPACY_TRF_DIR / f"{ds_name}_{split_name}.spacy"
            )
            doc_iter = iter(doc_bin.get_docs(vocab))
            is_train = split_name == "train"

            n_sents = 0
            n_filtered = 0

            for doc_id, (sample, full_doc) in enumerate(tqdm(
                zip(split_ds, doc_iter, strict=True),
                desc=f"{ds_name}/{split_name}",
                total=len(split_ds),
            )):
                doc_clusters = _clusters(ds_name, sample)
                sls = _sent_lens(ds_name, sample)
                sents = list(full_doc.sents)

                # Map each gold mention span to its syntactic head token only (span.root),
                # so one mention contributes exactly one nominal — avoids labeling the
                # internal tokens of a multi-token mention (e.g. "New York") as coreferent.
                mentions_by_sent: dict[int, dict[int, int]] = {}
                cluster_counts_by_sent: dict[int, dict[int, int]] = {}
                for cluster_id, cluster in enumerate(doc_clusters):
                    for m_sent_idx, m_start, m_end in cluster:
                        if m_sent_idx >= len(sents):
                            continue
                        sent_span = sents[m_sent_idx]
                        if m_start >= len(sent_span):
                            continue
                        span = sent_span[m_start:min(m_end, len(sent_span))]
                        head_local = span.root.i - sent_span.start
                        token_map = mentions_by_sent.setdefault(m_sent_idx, {})
                        token_map[head_local] = cluster_id
                        counts = cluster_counts_by_sent.setdefault(m_sent_idx, {})
                        counts[cluster_id] = counts.get(cluster_id, 0) + 1

                for sent_idx, sent_len in enumerate(sls):
                    n_sents += 1
                    counts = cluster_counts_by_sent.get(sent_idx, {})
                    if not any(c >= 2 for c in counts.values()):
                        continue

                    mention_map = mentions_by_sent.get(sent_idx, {})
                    sent = sents[sent_idx]
                    cat, cont, edges, etypes, nominal_positions, nom_lex, pair_synt = build_sentence_graph(sent, sent_len)

                    if len(nominal_positions) < 2:
                        continue

                    n_filtered += 1
                    M = len(nominal_positions)

                    # gold_ante[i, j] = 1 if nominal j is a valid antecedent for nominal i (same cluster, j < i)
                    # gold_ante[i, M] = 1 (null) if nominal i has no antecedent among j < i
                    gold_ante = np.zeros((M, M + 1), dtype=np.float32)
                    for i, ti in enumerate(nominal_positions):
                        ci = mention_map.get(ti)
                        has_ante = False
                        for j in range(i):
                            tj = nominal_positions[j]
                            cj = mention_map.get(tj)
                            if ci is not None and cj is not None and ci == cj:
                                gold_ante[i, j] = 1.0
                                has_ante = True
                        if not has_ante:
                            gold_ante[i, M] = 1.0

                    nom_idx = np.array(nominal_positions, dtype=np.int64)
                    # Full gold clusters (all gold-mention heads, any POS) for this
                    # sentence, so evaluation needs no spaCy doc re-read.
                    gold_groups: dict[int, set[int]] = {}
                    for hl, cid in mention_map.items():
                        gold_groups.setdefault(cid, set()).add(hl)
                    gold_clusters = [frozenset(m) for m in gold_groups.values() if len(m) >= 2]
                    # key = (ds_name, split_name, doc_id, sent_idx); retained for kfold grouping
                    key = (ds_name, split_name, doc_id, sent_idx)
                    rows.append((key, cat, cont, edges, etypes, nom_idx, nom_lex, pair_synt, gold_ante, gold_clusters))

            n_sents_total += n_sents
            n_filtered_total += n_filtered
            rate = n_filtered * 100.0 / max(n_sents, 1)
            print(f"  {ds_name}/{split_name}: {n_filtered}/{n_sents} ({rate:.1f}%)")

    print(f"\nTotal Stage 1 data: {n_filtered_total}/{n_sents_total} ({n_filtered_total*100.0/max(n_sents_total, 1):.1f}%)")

    data = rows

    with FEATURES_CACHE.open("wb") as f:
        pickle.dump(data, f)
    print(f"Cached features to {FEATURES_CACHE.name}")

    return data


def kfold_split(data: list, n_folds: int, fold: int) -> tuple[list, list]:
    rng = np.random.default_rng(42)
    by_ds: dict[str, list[int]] = {}
    for i, (key, *_) in enumerate(data):
        by_ds.setdefault(key[0], []).append(i)  # key[0] = ds_name

    train_idx, val_idx = [], []
    for indices in by_ds.values():
        perm = rng.permutation(len(indices))
        fold_size = len(perm) // n_folds
        val_start = fold * fold_size
        val_end = val_start + fold_size
        for j, idx in enumerate(perm):
            (val_idx if val_start <= j < val_end else train_idx).append(indices[idx])

    # training strips key + gold_clusters; val keeps the full row for evaluation
    return [data[i][1:-1] for i in train_idx], [data[i] for i in val_idx]


def make_batches(data: list, max_tokens: int = 32768, max_rows: int = 512) -> list[list]:
    order = sorted(range(len(data)), key=lambda i: data[i][0].shape[0])
    batches = []
    cur: list[int] = []
    cur_tmax = 0
    for i in order:
        tlen = data[i][0].shape[0]
        tmax = max(cur_tmax, tlen)
        if cur and ((len(cur) + 1) * tmax > max_tokens or len(cur) + 1 > max_rows):
            batches.append([data[j] for j in cur])
            cur = []
            cur_tmax = 0
        cur.append(i)
        cur_tmax = max(cur_tmax, tlen)
    if cur:
        batches.append([data[j] for j in cur])
    return batches


def collate_batch(batch: list, device: torch.device | str, model: "DepGraphTransformer") -> tuple:
    B = len(batch)
    l_max = max(item[0].shape[0] for item in batch)
    n_max = max(len(item[4]) for item in batch)

    cat_buf = np.zeros((B, l_max, N_CAT), dtype=np.int64)
    cont_buf = np.zeros((B, l_max, N_CONT_FULL), dtype=np.float32)
    pad_mask = np.ones((B, l_max), dtype=bool)
    nom_idx_buf = np.zeros((B, n_max), dtype=np.int64)
    nom_mask_buf = np.zeros((B, n_max), dtype=bool)
    nom_lex_buf = np.full((B, n_max, 2), -1, dtype=np.int64)
    synt_buf = np.zeros((B, n_max, n_max, N_PAIR_SYNT), dtype=np.float32)
    gold_buf = np.zeros((B, n_max, n_max + 1), dtype=np.float32)
    edge_list = []
    lengths = []

    for b, (cat, cont, edges, etypes, ni, nl, ps, ga) in enumerate(batch):
        L = cat.shape[0]
        cat_buf[b, :L] = cat
        cont_buf[b, :L] = cont
        pad_mask[b, :L] = False
        k = len(ni)
        nom_idx_buf[b, :k] = ni
        nom_mask_buf[b, :k] = True
        nom_lex_buf[b, :k] = nl
        synt_buf[b, :k, :k] = ps
        gold_buf[b, :k, :k] = ga[:, :k]
        gold_buf[b, :k, n_max] = ga[:, -1]
        edge_list.append((edges, etypes))
        lengths.append(L)

    cat_t = torch.from_numpy(cat_buf).to(device)
    cont_t = torch.from_numpy(cont_buf).to(device)
    pad_t = torch.from_numpy(pad_mask).to(device)
    nom_t = torch.from_numpy(nom_idx_buf).to(device)
    nom_mask_t = torch.from_numpy(nom_mask_buf).to(device)
    nom_lex_t = torch.from_numpy(nom_lex_buf).to(device)
    synt_t = torch.from_numpy(synt_buf).to(device)
    gold_t = torch.from_numpy(gold_buf).to(device)
    attn_bias = model._build_batch_attn_bias(edge_list, lengths, l_max, device)

    return cat_t, cont_t, pad_t, nom_t, nom_mask_t, nom_lex_t, synt_t, gold_t, attn_bias


def run_epoch(
    model: DepGraphTransformer,
    batches: list,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device | str,
) -> float:
    train = optimizer is not None
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(batches, desc="train" if train else "val"):
        cat, cont, pad_mask, nom_idx, nom_mask, nom_lex, pair_synt, gold_ante, attn_bias = collate_batch(batch, device, model)

        if train:
            optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            scores = model(cat, cont, pad_mask, nom_idx, nom_mask, nom_lex, pair_synt, attn_bias)
            loss = mention_ranking_loss(scores, gold_ante, nom_mask)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# Mentions are head-token indices (sentence-relative), matching the training unit
# (gold spans mapped to span.root). Clusters are sets of those indices.
Cluster = frozenset[int]


def _uf_find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _pred_clusters(nom_idx: np.ndarray, scores: np.ndarray, null_margin: float) -> list[Cluster]:
    M = len(nom_idx)
    parent = list(range(M))
    for i in range(M):
        masked = scores[i].copy()
        masked[i:M] = float("-inf")
        masked[M] -= null_margin
        ante = int(np.argmax(masked))
        if ante < M:
            parent[_uf_find(parent, i)] = _uf_find(parent, ante)
    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(_uf_find(parent, i), []).append(int(nom_idx[i]))
    return [frozenset(members) for members in groups.values() if len(members) >= 2]


def _muc(pred: list[Cluster], gold: list[Cluster]) -> tuple[int, int, int, int]:
    def _score(a: list[Cluster], b: list[Cluster]) -> tuple[int, int]:
        b_mentions = set().union(*b) if b else set()
        num = den = 0
        for c in a:
            if len(c) < 2:
                continue
            den += len(c) - 1
            # partitions = response clusters intersecting c + c-mentions absent
            # from the response (each counts as its own singleton partition)
            partitions = sum(1 for bc in b if c & bc) + len(c - b_mentions)
            num += len(c) - partitions
        return num, den
    pn, pd = _score(pred, gold)
    rn, rd = _score(gold, pred)
    return pn, pd, rn, rd


def _b3(pred: list[Cluster], gold: list[Cluster]) -> tuple[float, float, int]:
    pred_map: dict[int, Cluster] = {m: c for c in pred for m in c}
    gold_map: dict[int, Cluster] = {m: c for c in gold for m in c}
    mentions = set(pred_map) | set(gold_map)
    if not mentions:
        return 0.0, 0.0, 0
    p = sum(len(pred_map.get(m, frozenset({m})) & gold_map.get(m, frozenset({m}))) / len(pred_map.get(m, frozenset({m}))) for m in mentions)
    r = sum(len(pred_map.get(m, frozenset({m})) & gold_map.get(m, frozenset({m}))) / len(gold_map.get(m, frozenset({m}))) for m in mentions)
    return p, r, len(mentions)


def _ceafe(pred: list[Cluster], gold: list[Cluster]) -> tuple[float, float]:
    if not pred or not gold:
        return 0.0, 0.0
    cost = np.array([[2 * len(p & g) / (len(p) + len(g)) for g in gold] for p in pred])
    ri, ci = linear_sum_assignment(-cost)
    score = cost[ri, ci].sum()
    return score / len(pred), score / len(gold)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _collect_predictions(model: DepGraphTransformer, device: str, val_data: list) -> list[tuple[np.ndarray, np.ndarray, list[Cluster]]]:
    # Run the model once per val sentence; gold clusters are cached, so no doc re-read.
    collected: list[tuple[np.ndarray, np.ndarray, list[Cluster]]] = []
    for row in tqdm(val_data, desc="eval"):
        key, cat, cont, edges, etypes, nom_idx, nom_lex, pair_synt, gold_ante, gold_clusters = row
        M = len(nom_idx)
        with torch.inference_mode():
            attn_bias = model._build_batch_attn_bias([(edges, etypes)], [cat.shape[0]], cat.shape[0], device)
            scores = model(
                torch.from_numpy(cat).unsqueeze(0).to(device),
                torch.from_numpy(cont).unsqueeze(0).to(device),
                torch.zeros(1, cat.shape[0], dtype=torch.bool, device=device),
                torch.from_numpy(nom_idx).unsqueeze(0).to(device),
                torch.ones(1, M, dtype=torch.bool, device=device),
                torch.from_numpy(nom_lex).unsqueeze(0).to(device),
                torch.from_numpy(pair_synt).unsqueeze(0).to(device),
                attn_bias,
            )
        scores_np = scores.squeeze(0).cpu().numpy()
        collected.append((nom_idx, scores_np, gold_clusters))
    return collected


def _score_predictions(collected: list[tuple[np.ndarray, np.ndarray, list[Cluster]]], null_margin: float, linking_only: bool = False) -> dict[str, float]:
    muc_pn = muc_pd = muc_rn = muc_rd = 0
    b3_p = b3_r = 0.0
    b3_n = 0
    ceafe_p = ceafe_r = 0.0
    n_sents = 0

    for nom_idx, scores_np, gold in collected:
        if linking_only:
            noms = {int(x) for x in nom_idx}
            gold = [c & noms for c in gold]
            gold = [c for c in gold if len(c) >= 2]
        pred = _pred_clusters(nom_idx, scores_np, null_margin)

        pn, pd, rn, rd = _muc(pred, gold)
        muc_pn += pn; muc_pd += pd; muc_rn += rn; muc_rd += rd

        bp, br, bn = _b3(pred, gold)
        b3_p += bp; b3_r += br; b3_n += bn

        cp, cr = _ceafe(pred, gold)
        ceafe_p += cp; ceafe_r += cr
        n_sents += 1

    muc_f = _f1(muc_pn / max(muc_pd, 1), muc_rn / max(muc_rd, 1))
    b3_f = _f1(b3_p / max(b3_n, 1), b3_r / max(b3_n, 1))
    ceafe_f = _f1(ceafe_p / max(n_sents, 1), ceafe_r / max(n_sents, 1))
    return {"MUC": muc_f, "B3": b3_f, "CEAFe": ceafe_f, "CoNLL": (muc_f + b3_f + ceafe_f) / 3, "n_sents": n_sents}


def run_stage1_eval(model: DepGraphTransformer, device: str, val_data: list) -> None:
    # end_to_end: gold = all gold-mention heads (includes non-nominal heads the model
    #   cannot reach — the honest metric with the POS recall ceiling baked in).
    # linking_only: gold restricted to nominals the model actually scores — isolates
    #   clustering quality from candidate-set recall.
    collected = _collect_predictions(model, device, val_data)
    margins = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    results: dict[str, list] = {}
    for label, linking_only in [("end_to_end", False), ("linking_only", True)]:
        print(f"\n=== {label} null margin sweep ===")
        rows = []
        for null_margin in margins:
            m = _score_predictions(collected, null_margin, linking_only)
            m["null_margin"] = null_margin
            rows.append(m)
            print(f"  null_margin={null_margin:.1f}  MUC={m['MUC']:.4f}  B3={m['B3']:.4f}  CEAFe={m['CEAFe']:.4f}  CoNLL={m['CoNLL']:.4f}  ({m['n_sents']} sents)")
        results[label] = rows
    with (MODELS_DIR / "stage1_eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def train_stage1(
    learning_rate: float = 1e-3,
    max_epochs: int = 20,
    patience: int = 6,
    n_folds: int = 4,
    fold: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print("\nBuilding Stage 1 training data...")
    data = build_stage1_data()
    print(f"Total rows: {len(data)}")

    train_data, val_data = kfold_split(data, n_folds, fold)
    print(f"Fold {fold}/{n_folds}: {len(train_data)} train, {len(val_data)} val")
    val_data_stripped = [row[1:-1] for row in val_data]

    print(f"\nTraining Stage 1 on {device}")
    model = DepGraphTransformer(d_model=256, n_heads=8, n_layers=4).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\n=== Model Complexity ===")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {model_bytes / 1024:.1f} KB")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_batches = make_batches(train_data)
    val_batches = make_batches(val_data_stripped)
    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = MODELS_DIR / "stage1_train_log.json"
    history: list[dict] = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    min_delta = 1e-4
    start_epoch = 0

    ckpt_path = MODELS_DIR / "stage1_graph_transformer.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_val_loss = ckpt["best_val_loss"]
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {ckpt['epoch'] + 1}, best_val_loss={best_val_loss:.6f}")

    for epoch in range(start_epoch, max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        random.shuffle(train_batches)

        model.train()
        avg_loss = run_epoch(model, train_batches, optimizer, device)
        print(f"Loss: {avg_loss:.6f}")

        model.eval()
        with torch.inference_mode():
            avg_val_loss = run_epoch(model, val_batches, None, device)
        print(f"Val loss: {avg_val_loss:.6f}")

        scheduler.step()

        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": avg_val_loss})
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, MODELS_DIR / "stage1_graph_transformer.pt")
            print("✓ Best model saved")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n⊘ Early stopping at epoch {epoch + 1}")
                print(f"Best: epoch {best_epoch + 1}, val_loss: {best_val_loss:.6f}")
                break

    print("\n✓ Training complete")

    if ckpt_path.exists():
        best = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best["model"] if isinstance(best, dict) and "model" in best else best)
    model.eval()
    run_stage1_eval(model, device, val_data)


if __name__ == "__main__":
    train_stage1()
