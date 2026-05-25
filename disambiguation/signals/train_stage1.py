import pickle
import random
import torch
import torch.optim as optim
import numpy as np
import spacy
import spacy.tokens
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk

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
        if data and len(data[0]) != 8:
            print("Cache format outdated — rebuilding...")
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

                mentions_by_sent: dict[int, dict[int, int]] = {}
                cluster_counts_by_sent: dict[int, dict[int, int]] = {}
                for cluster_id, cluster in enumerate(doc_clusters):
                    for m_sent_idx, m_start, m_end in cluster:
                        token_map = mentions_by_sent.setdefault(m_sent_idx, {})
                        for m_tok_idx in range(m_start, m_end):
                            token_map[m_tok_idx] = cluster_id
                        counts = cluster_counts_by_sent.setdefault(m_sent_idx, {})
                        counts[cluster_id] = counts.get(cluster_id, 0) + 1

                for sent_idx, sent_len in enumerate(sls):
                    n_sents += 1
                    counts = cluster_counts_by_sent.get(sent_idx, {})
                    if not any(c >= 2 for c in counts.values()):
                        continue

                    mention_map = mentions_by_sent.get(sent_idx, {})
                    sent = sents[sent_idx]
                    cat, cont, edges, etypes, nominal_positions = build_sentence_graph(sent, sent_len)

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
                    # key = (ds_name, split_name, doc_id, sent_idx) for spaCy doc lookup
                    key = (ds_name, split_name, doc_id, sent_idx)
                    rows.append((key, cat, cont, edges, etypes, nom_idx, gold_ante))

            n_sents_total += n_sents
            n_filtered_total += n_filtered
            rate = n_filtered * 100.0 / max(n_sents, 1)
            print(f"  {ds_name}/{split_name}: {n_filtered}/{n_sents} ({rate:.1f}%)")

    print(f"\nTotal Stage 1 data: {n_filtered_total}/{n_sents_total} ({n_filtered_total*100.0/max(n_sents_total, 1):.1f}%)")

    data = [(key, cat, cont, edges, etypes, ni, ga) for key, cat, cont, edges, etypes, ni, ga in rows]

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

    # training strips key, val keeps key for evaluation lookup
    return [data[i][1:] for i in train_idx], [data[i] for i in val_idx]


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

    cat_buf = np.zeros((B, l_max, 6), dtype=np.int64)
    cont_buf = np.zeros((B, l_max, 12), dtype=np.float32)
    pad_mask = np.ones((B, l_max), dtype=bool)
    nom_idx_buf = np.zeros((B, n_max), dtype=np.int64)
    nom_mask_buf = np.zeros((B, n_max), dtype=bool)
    gold_buf = np.zeros((B, n_max, n_max + 1), dtype=np.float32)
    edge_list = []
    lengths = []

    for b, (cat, cont, edges, etypes, ni, ga) in enumerate(batch):
        L = cat.shape[0]
        cat_buf[b, :L] = cat
        cont_buf[b, :L] = cont
        pad_mask[b, :L] = False
        k = len(ni)
        nom_idx_buf[b, :k] = ni
        nom_mask_buf[b, :k] = True
        gold_buf[b, :k, :k] = ga[:, :k]
        gold_buf[b, :k, n_max] = ga[:, -1]
        edge_list.append((edges, etypes))
        lengths.append(L)

    cat_t = torch.from_numpy(cat_buf).to(device)
    cont_t = torch.from_numpy(cont_buf).to(device)
    pad_t = torch.from_numpy(pad_mask).to(device)
    nom_t = torch.from_numpy(nom_idx_buf).to(device)
    nom_mask_t = torch.from_numpy(nom_mask_buf).to(device)
    gold_t = torch.from_numpy(gold_buf).to(device)
    attn_bias = model._build_batch_attn_bias(edge_list, lengths, l_max, device)

    return cat_t, cont_t, pad_t, nom_t, nom_mask_t, gold_t, attn_bias


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
        cat, cont, pad_mask, nom_idx, nom_mask, gold_ante, attn_bias = collate_batch(batch, device, model)

        if train:
            optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            scores = model(cat, cont, pad_mask, nom_idx, nom_mask, attn_bias)
            loss = mention_ranking_loss(scores, gold_ante, nom_mask)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_stage1(
    learning_rate: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 6,
    n_folds: int = 5,
    fold: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print("\nBuilding Stage 1 training data...")
    data = build_stage1_data()
    print(f"Total rows: {len(data)}")

    train_data, val_data = kfold_split(data, n_folds, fold)
    print(f"Fold {fold}/{n_folds}: {len(train_data)} train, {len(val_data)} val")
    val_data_stripped = [row[1:] for row in val_data]

    print(f"\nTraining Stage 1 on {device}")
    model = DepGraphTransformer(d_model=256, n_heads=8, n_layers=4).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\n=== Model Complexity ===")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {model_bytes / 1024:.1f} KB")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_batches = make_batches(train_data)
    val_batches = make_batches(val_data_stripped)
    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
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
        with torch.no_grad():
            avg_val_loss = run_epoch(model, val_batches, None, device)
        print(f"Val loss: {avg_val_loss:.6f}")

        scheduler.step()

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


if __name__ == "__main__":
    train_stage1()
