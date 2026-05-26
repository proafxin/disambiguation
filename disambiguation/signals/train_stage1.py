import datetime
import json
import pickle
import random

import numpy as np
import spacy
import spacy.tokens
import torch
import torch.optim as optim
from datasets import load_from_disk
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from disambiguation.signals.abstract_features import POS_IDS
from disambiguation.signals.stage1_intrasentence import (
    NominalCorefScorer, pair_bce_loss, decode_clusters,
    NOMINAL_POS, MAX_LEN, BGE_DIM, MODELS_DIR, CKPT_NAME, BGE_MODEL,
)
from disambiguation.signals.train_full import _clusters, _sent_lens

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
DATA_DIR = CACHE_DIR.parent / "data"
SPACY_TRF_DIR = DATA_DIR / "spacy_trf"
FEATURES_CACHE = DATA_DIR / "stage1_coref_features.pkl"
BGE_CACHE = DATA_DIR / "bge_vocab.pkl"
LEX_MASK = 0x7FFFFFFFFFFFFFFF

DATASET_CONFIG = [
    ("preco", ["train", "validation"]),
    ("litbank", ["train", "validation", "test"]),
    ("corefud", ["train", "validation"]),
    ("conll2012", ["train", "validation", "test"]),
]

Row = tuple  # (key, words, nom_idx, gold_cid, pos_ids, nom_lex)


def build_stage1_data() -> list:
    if FEATURES_CACHE.exists():
        with FEATURES_CACHE.open("rb") as f:
            data = pickle.load(f)
        print(f"Loaded cached features: {len(data)} rows")
        return data

    vocab = spacy.blank("en").vocab
    rows: list = []
    n_sents_total = 0
    n_kept_total = 0

    for ds_name, splits in DATASET_CONFIG:
        ds_dict = load_from_disk(str(DATA_DIR / ds_name))
        for split_name in splits:
            if split_name not in ds_dict:
                continue
            split_ds = ds_dict[split_name]
            doc_bin = spacy.tokens.DocBin().from_disk(SPACY_TRF_DIR / f"{ds_name}_{split_name}.spacy")
            doc_iter = iter(doc_bin.get_docs(vocab))
            n_sents = 0
            n_kept = 0

            for doc_id, (sample, full_doc) in enumerate(tqdm(
                zip(split_ds, doc_iter, strict=True), desc=f"{ds_name}/{split_name}", total=len(split_ds),
            )):
                doc_clusters = _clusters(ds_name, sample)
                sls = _sent_lens(ds_name, sample)
                sents = list(full_doc.sents)

                # Map each gold mention span to its syntactic head token (span.root),
                # so one mention = one nominal head. cluster_id is document-level.
                mentions_by_sent: dict[int, dict[int, int]] = {}
                counts_by_sent: dict[int, dict[int, int]] = {}
                for cluster_id, cluster in enumerate(doc_clusters):
                    for m_sent_idx, m_start, m_end in cluster:
                        if m_sent_idx >= len(sents):
                            continue
                        sent_span = sents[m_sent_idx]
                        if m_start >= len(sent_span):
                            continue
                        head_local = sent_span[m_start:min(m_end, len(sent_span))].root.i - sent_span.start
                        mentions_by_sent.setdefault(m_sent_idx, {})[head_local] = cluster_id
                        c = counts_by_sent.setdefault(m_sent_idx, {})
                        c[cluster_id] = c.get(cluster_id, 0) + 1

                for sent_idx, _ in enumerate(sls):
                    n_sents += 1
                    counts = counts_by_sent.get(sent_idx, {})
                    if not any(v >= 2 for v in counts.values()):
                        continue
                    if sent_idx >= len(sents):
                        continue
                    sent = sents[sent_idx]
                    L = min(len(sent), MAX_LEN)
                    nominal_positions = [ti for ti in range(L) if sent[ti].pos_ in NOMINAL_POS]
                    if len(nominal_positions) < 2:
                        continue

                    mention_map = mentions_by_sent.get(sent_idx, {})
                    words = [sent[ti].text for ti in range(L)]
                    nom_idx = np.array(nominal_positions, dtype=np.int64)
                    gold_cid = np.array([mention_map.get(ti, -1) for ti in nominal_positions], dtype=np.int64)
                    _, cnts = np.unique(gold_cid[gold_cid >= 0], return_counts=True)
                    if cnts.size == 0 or cnts.max() < 2:  # need >=2 nominal heads sharing a cluster
                        continue
                    pos_ids = np.array([POS_IDS.get(sent[ti].pos_, len(POS_IDS)) for ti in nominal_positions], dtype=np.int64)
                    nom_lex = np.array(
                        [[sent[ti].lemma & LEX_MASK, sent[ti].lower & LEX_MASK] for ti in nominal_positions],
                        dtype=np.int64,
                    )
                    key = (ds_name, split_name, doc_id, sent_idx)
                    rows.append((key, words, nom_idx, gold_cid, pos_ids, nom_lex))
                    n_kept += 1

            n_sents_total += n_sents
            n_kept_total += n_kept
            print(f"  {ds_name}/{split_name}: {n_kept}/{n_sents} sentences kept")

    print(f"\nTotal: {n_kept_total}/{n_sents_total} sentences")
    with FEATURES_CACHE.open("wb") as f:
        pickle.dump(rows, f)
    print(f"Cached features to {FEATURES_CACHE.name}")
    return rows


def build_bge_cache(data: list) -> tuple[dict, np.ndarray]:
    if BGE_CACHE.exists():
        with BGE_CACHE.open("rb") as f:
            word2id, matrix = pickle.load(f)
        print(f"Loaded BGE cache: {len(word2id)} words")
        return word2id, matrix

    from sentence_transformers import SentenceTransformer

    vocab = sorted({w for _, words, *_ in data for w in words})
    print(f"Encoding {len(vocab)} unique words with BGE...")
    bge = SentenceTransformer(BGE_MODEL)
    embs = bge.encode(vocab, normalize_embeddings=True, batch_size=512, show_progress_bar=True)
    matrix = np.asarray(embs, dtype=np.float32)
    word2id = {w: i for i, w in enumerate(vocab)}
    with BGE_CACHE.open("wb") as f:
        pickle.dump((word2id, matrix.astype(np.float16)), f)
    print(f"Cached BGE vocab to {BGE_CACHE.name}")
    return word2id, matrix


def kfold_split(data: list, n_folds: int, fold: int) -> tuple[list, list]:
    rng = np.random.default_rng(42)
    by_ds: dict[str, list[int]] = {}
    for i, (key, *_) in enumerate(data):
        by_ds.setdefault(key[0], []).append(i)
    train_idx, val_idx = [], []
    for indices in by_ds.values():
        perm = rng.permutation(len(indices))
        fold_size = len(perm) // n_folds
        lo, hi = fold * fold_size, fold * fold_size + fold_size
        for j, idx in enumerate(perm):
            (val_idx if lo <= j < hi else train_idx).append(indices[idx])
    return [data[i] for i in train_idx], [data[i] for i in val_idx]


def make_batches(data: list, max_pairs: int = 20000, max_rows: int = 128) -> list[list]:
    # Memory is driven by the (B, M, M, ...) pairwise tensor, so budget by M^2 (nominals), not tokens.
    order = sorted(range(len(data)), key=lambda i: len(data[i][2]))
    batches, cur, m_max = [], [], 0
    for i in order:
        m = len(data[i][2])
        nm = max(m_max, m)
        if cur and ((len(cur) + 1) * nm * nm > max_pairs or len(cur) + 1 > max_rows):
            batches.append([data[j] for j in cur])
            cur, m_max = [], 0
        cur.append(i)
        m_max = max(m_max, m)
    if cur:
        batches.append([data[j] for j in cur])
    return batches


def collate_batch(batch: list, word2id: dict, matrix: np.ndarray, device: str) -> tuple:
    B = len(batch)
    L_max = max(len(r[1]) for r in batch)
    M_max = max(len(r[2]) for r in batch)
    bge_buf = np.zeros((B, L_max, BGE_DIM), dtype=np.float32)
    pad = np.ones((B, L_max), dtype=bool)
    nom_buf = np.zeros((B, M_max), dtype=np.int64)
    nom_mask = np.zeros((B, M_max), dtype=bool)
    gold = np.zeros((B, M_max, M_max), dtype=np.float32)

    for b, (_, words, nom_idx, gold_cid, _, _) in enumerate(batch):
        L = len(words)
        ids = np.fromiter((word2id[w] for w in words), dtype=np.int64, count=L)
        bge_buf[b, :L] = matrix[ids]
        pad[b, :L] = False
        k = len(nom_idx)
        nom_buf[b, :k] = nom_idx
        nom_mask[b, :k] = True
        same = (gold_cid[:, None] == gold_cid[None, :]) & (gold_cid[:, None] >= 0)
        gold[b, :k, :k] = same.astype(np.float32)

    return (
        torch.from_numpy(bge_buf).to(device),
        torch.from_numpy(pad).to(device),
        torch.from_numpy(nom_buf).to(device),
        torch.from_numpy(nom_mask).to(device),
        torch.from_numpy(gold).to(device),
    )


def run_epoch(model, batches, word2id, matrix, optimizer, device) -> float:
    train = optimizer is not None
    total, n = 0.0, 0
    for batch in tqdm(batches, desc="train" if train else "val"):
        bge, pad, nom_idx, nom_mask, gold = collate_batch(batch, word2id, matrix, device)
        if train:
            optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(bge, pad, nom_idx, nom_mask)
            loss = pair_bce_loss(logits, gold, nom_mask)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


Cluster = frozenset


def _gold_clusters(nom_idx: np.ndarray, gold_cid: np.ndarray) -> list:
    groups: dict[int, list[int]] = {}
    for k, cid in enumerate(gold_cid):
        if cid >= 0:
            groups.setdefault(int(cid), []).append(int(nom_idx[k]))
    return [frozenset(m) for m in groups.values() if len(m) >= 2]


def _links(clusters: list) -> set:
    pairs = set()
    for c in clusters:
        members = sorted(c)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def _muc(pred: list, gold: list) -> tuple:
    def score(a, b):
        bm = set().union(*b) if b else set()
        num = den = 0
        for c in a:
            if len(c) < 2:
                continue
            den += len(c) - 1
            num += len(c) - (sum(1 for bc in b if c & bc) + len(c - bm))
        return num, den
    pn, pd = score(pred, gold)
    rn, rd = score(gold, pred)
    return pn, pd, rn, rd


def _b3(pred: list, gold: list) -> tuple:
    pm = {m: c for c in pred for m in c}
    gm = {m: c for c in gold for m in c}
    mentions = set(pm) | set(gm)
    if not mentions:
        return 0.0, 0.0, 0
    p = sum(len(pm.get(m, frozenset({m})) & gm.get(m, frozenset({m}))) / len(pm.get(m, frozenset({m}))) for m in mentions)
    r = sum(len(pm.get(m, frozenset({m})) & gm.get(m, frozenset({m}))) / len(gm.get(m, frozenset({m}))) for m in mentions)
    return p, r, len(mentions)


def _ceafe(pred: list, gold: list) -> tuple:
    if not pred or not gold:
        return 0.0, 0.0
    cost = np.array([[2 * len(p & g) / (len(p) + len(g)) for g in gold] for p in pred])
    ri, ci = linear_sum_assignment(-cost)
    s = cost[ri, ci].sum()
    return s / len(pred), s / len(gold)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _collect_predictions(model, val_data, word2id, matrix, device) -> list:
    collected = []
    for row in tqdm(val_data, desc="eval"):
        _, words, nom_idx, gold_cid, _, _ = row
        L = len(words)
        ids = np.fromiter((word2id[w] for w in words), dtype=np.int64, count=L)
        bge = torch.from_numpy(matrix[ids][None]).to(device)
        pad = torch.zeros(1, L, dtype=torch.bool, device=device)
        nom_t = torch.from_numpy(nom_idx[None]).to(device)
        nom_mask = torch.ones(1, len(nom_idx), dtype=torch.bool, device=device)
        with torch.inference_mode():
            logits = model(bge, pad, nom_t, nom_mask).squeeze(0).float().cpu().numpy()
        collected.append((nom_idx, logits, _gold_clusters(nom_idx, gold_cid)))
    return collected


def _score(collected: list, threshold: float) -> dict:
    muc = [0, 0, 0, 0]
    b3p = b3r = 0.0
    b3n = 0
    cep = cer = 0.0
    n = 0
    glink = plink = tp = 0
    for nom_idx, logits, gold in collected:
        pred = decode_clusters(nom_idx, logits, threshold)
        a, b, c, d = _muc(pred, gold)
        muc[0] += a; muc[1] += b; muc[2] += c; muc[3] += d
        bp, br, bn = _b3(pred, gold)
        b3p += bp; b3r += br; b3n += bn
        cp, cr = _ceafe(pred, gold)
        cep += cp; cer += cr; n += 1
        gl, pl = _links(gold), _links(pred)
        glink += len(gl); plink += len(pl); tp += len(gl & pl)
    muc_f = _f1(muc[0] / max(muc[1], 1), muc[2] / max(muc[3], 1))
    b3_f = _f1(b3p / max(b3n, 1), b3r / max(b3n, 1))
    ce_f = _f1(cep / max(n, 1), cer / max(n, 1))
    return {
        "link_p": tp / max(plink, 1), "link_r": tp / max(glink, 1), "link_f": _f1(tp / max(plink, 1), tp / max(glink, 1)),
        "MUC": muc_f, "B3": b3_f, "CEAFe": ce_f, "CoNLL": (muc_f + b3_f + ce_f) / 3,
    }


def run_eval(model, val_data, word2id, matrix, device) -> None:
    collected = _collect_predictions(model, val_data, word2id, matrix, device)
    print("\n=== threshold sweep ===")
    print(f"{'thr':>5} {'linkP':>6} {'linkR':>6} {'linkF':>6} {'MUC':>6} {'B3':>6} {'CEAFe':>6} {'CoNLL':>6}")
    results = []
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        m = _score(collected, thr)
        m["threshold"] = thr
        results.append(m)
        print(f"{thr:>5.1f} {m['link_p']:>6.3f} {m['link_r']:>6.3f} {m['link_f']:>6.3f} {m['MUC']:>6.3f} {m['B3']:>6.3f} {m['CEAFe']:>6.3f} {m['CoNLL']:>6.3f}")
    with (MODELS_DIR / "stage1_eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def train_stage1(
    learning_rate: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 6,
    n_folds: int = 4,
    fold: int = 0,
    resume: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print("\nBuilding Stage 1 data...")
    data = build_stage1_data()
    word2id, matrix = build_bge_cache(data)
    print(f"Total rows: {len(data)}")

    train_data, val_data = kfold_split(data, n_folds, fold)
    print(f"Fold {fold}/{n_folds}: {len(train_data)} train, {len(val_data)} val")

    print(f"\nTraining on {device}")
    model = NominalCorefScorer().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    train_batches = make_batches(train_data)
    val_batches = make_batches(val_data)
    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(CACHE_DIR / "tensorboard" / f"stage1_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
    best_val, patience_ctr, best_epoch = float("inf"), 0, 0
    disk_best = float("inf")
    ckpt_path = MODELS_DIR / CKPT_NAME
    if resume and ckpt_path.exists():
        # Warm-start from saved weights, then train a fresh full cycle from epoch 0 (fresh
        # optimizer + LR schedule). The on-disk best is overwritten only on a new all-time low,
        # so a warm restart can never regress the saved model.
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        disk_best = ckpt.get("best_val_loss", float("inf"))
        print(f"Warm-started from saved model (best_val_loss {disk_best:.6f}); training fresh from epoch 0")

    for epoch in range(max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        random.shuffle(train_batches)
        model.train()
        tr_loss = run_epoch(model, train_batches, word2id, matrix, optimizer, device)
        print(f"Loss: {tr_loss:.6f}")
        model.eval()
        with torch.inference_mode():
            val_loss = run_epoch(model, val_batches, word2id, matrix, None, device)
        print(f"Val loss: {val_loss:.6f}")
        scheduler.step()
        tb.add_scalar("loss/train", tr_loss, epoch + 1)
        tb.add_scalar("loss/val", val_loss, epoch + 1)

        if val_loss < best_val - 1e-4:
            best_val, patience_ctr, best_epoch = val_loss, 0, epoch
            if val_loss < disk_best - 1e-4:
                disk_best = val_loss
                torch.save({"epoch": epoch, "model": model.state_dict(), "best_val_loss": disk_best}, ckpt_path)
                print(f"✓ Best model saved (all-time best {disk_best:.6f})")
            else:
                print(f"Improved this run to {val_loss:.6f} (all-time best {disk_best:.6f}; not overwriting)")
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"\n⊘ Early stopping. Best this run epoch {best_epoch + 1}, val_loss {best_val:.6f}")
                break

    print("\n✓ Training complete")
    tb.close()
    ckpt = torch.load(MODELS_DIR / CKPT_NAME, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    run_eval(model, val_data, word2id, matrix, device)


if __name__ == "__main__":
    train_stage1()
