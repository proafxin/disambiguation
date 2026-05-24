import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
import spacy.tokens
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

from disambiguation.signals.stage1_intrasentence import MiniTransformer, extract_raw_attributes
from disambiguation.signals.train_full import _clusters, _sent_lens, _compute_depth

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
DATA_DIR = CACHE_DIR.parent / "data"
SPACY_TRF_DIR = DATA_DIR / "spacy_trf"
MODELS_DIR = CACHE_DIR / "models"
EMBEDDINGS_CACHE = DATA_DIR / "stage1_embeddings.pkl"
FEATURES_CACHE = DATA_DIR / "stage1_features.pkl"
BGE_MODEL = "BAAI/bge-small-en-v1.5"

DATASET_CONFIG = [
    ("preco", ["train", "validation"]),
    ("litbank", ["train", "validation", "test"]),
    ("corefud", ["train", "validation"]),
    ("conll2012", ["train", "validation", "test"]),
]


def build_stage1_data():
    if FEATURES_CACHE.exists():
        with FEATURES_CACHE.open("rb") as f:
            train_data, val_data = pickle.load(f)
        print(f"Loaded cached features: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data

    embeddings = {}
    if EMBEDDINGS_CACHE.exists():
        with EMBEDDINGS_CACHE.open("rb") as f:
            embeddings = pickle.load(f)

    vocab = spacy.blank("en").vocab
    rows = []
    missing_keys = []
    missing_texts = []
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

                mentions_by_sent = {}
                cluster_counts_by_sent = {}
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
                    nominal_positions = []
                    for token_idx in range(sent_len):
                        tok = sent[token_idx]
                        if tok.pos_ in {"PRON", "NOUN", "PROPN"}:
                            nominal_positions.append(token_idx)

                    if len(nominal_positions) < 2:
                        continue

                    n_filtered += 1

                    token_attrs = extract_raw_attributes(sent, [sent_len])
                    n_nom = len(nominal_positions)
                    pair_labels = np.zeros((n_nom, n_nom), dtype=np.float32)
                    for a, ta in enumerate(nominal_positions):
                        for b, tb in enumerate(nominal_positions):
                            ca = mention_map.get(ta)
                            cb = mention_map.get(tb)
                            if ca is not None and cb is not None and ca == cb:
                                pair_labels[a, b] = 1.0

                    nom_idx = np.array(nominal_positions, dtype=np.int64)
                    key = (ds_name, split_name, doc_id, sent_idx)
                    if key not in embeddings:
                        missing_keys.append(key)
                        missing_texts.append(sent.text)
                    rows.append((is_train, key, token_attrs, nom_idx, pair_labels))

            n_sents_total += n_sents
            n_filtered_total += n_filtered
            rate = n_filtered * 100.0 / max(n_sents, 1)
            print(f"  {ds_name}/{split_name}: {n_filtered}/{n_sents} ({rate:.1f}%)")

    print(f"\nTotal Stage 1 data: {n_filtered_total}/{n_sents_total} ({n_filtered_total*100.0/max(n_sents_total, 1):.1f}%)")

    if missing_keys:
        print(f"\nEncoding {len(missing_keys)} embeddings with BGE...")
        model = SentenceTransformer(BGE_MODEL)
        vecs = model.encode(
            missing_texts,
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        for key, vec in zip(missing_keys, vecs, strict=True):
            embeddings[key] = vec
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with EMBEDDINGS_CACHE.open("wb") as f:
            pickle.dump(embeddings, f)
        print(f"Cached {len(embeddings)} embeddings to {EMBEDDINGS_CACHE.name}")

    train_data = []
    val_data = []
    for is_train, key, token_attrs, nom_idx, pair_labels in rows:
        sent_emb_reshaped = embeddings[key].reshape(12, 32)
        feature_matrix = np.vstack([token_attrs, sent_emb_reshaped])
        (train_data if is_train else val_data).append((feature_matrix, nom_idx, pair_labels))

    with FEATURES_CACHE.open("wb") as f:
        pickle.dump((train_data, val_data), f)
    print(f"Cached features to {FEATURES_CACHE.name}")

    return train_data, val_data


def make_batches(data, max_pairs=2_000_000, max_rows=2048):
    order = sorted(range(len(data)), key=lambda i: data[i][0].shape[0])
    batches = []
    cur = []
    cur_nmax = 0
    for i in order:
        n_nom = len(data[i][1])
        nmax = max(cur_nmax, n_nom)
        if cur and ((len(cur) + 1) * nmax * nmax > max_pairs or len(cur) + 1 > max_rows):
            batches.append([data[j] for j in cur])
            cur = []
            cur_nmax = 0
        cur.append(i)
        cur_nmax = max(cur_nmax, n_nom)
    if cur:
        batches.append([data[j] for j in cur])
    return batches


def collate_batch(batch, device):
    n = len(batch)
    l_max = max(fm.shape[0] - 12 for fm, _, _ in batch)
    n_max = max(len(ni) for _, ni, _ in batch)
    seq = l_max + 12

    feats = np.zeros((n, seq, 32), dtype=np.float32)
    pad_mask = np.ones((n, seq), dtype=bool)
    nom_idx = np.zeros((n, n_max), dtype=np.int64)
    nom_mask = np.zeros((n, n_max), dtype=bool)
    labels = np.zeros((n, n_max, n_max), dtype=np.float32)

    for b, (fm, ni, lab) in enumerate(batch):
        li = fm.shape[0] - 12
        feats[b, :li] = fm[:li]
        feats[b, l_max:seq] = fm[li:li + 12]
        pad_mask[b, :li] = False
        pad_mask[b, l_max:seq] = False
        k = len(ni)
        nom_idx[b, :k] = ni
        nom_mask[b, :k] = True
        labels[b, :k, :k] = lab

    return (
        torch.from_numpy(feats).to(device),
        torch.from_numpy(nom_idx).to(device),
        torch.from_numpy(pad_mask).to(device),
        torch.from_numpy(nom_mask).to(device),
        torch.from_numpy(labels).to(device),
    )


def run_epoch(model, batches, loss_fn, optimizer, device):
    train = optimizer is not None
    total_loss_sum = 0.0
    total_pairs = 0
    for batch in tqdm(batches, desc="train" if train else "val"):
        feats, nom_idx, pad_mask, nom_mask, labels = collate_batch(batch, device)
        valid = nom_mask.unsqueeze(2) & nom_mask.unsqueeze(1)
        eye = torch.eye(valid.shape[1], device=device, dtype=torch.bool)
        pair_mask = (valid & ~eye).float()

        if train:
            optimizer.zero_grad()
        logits = model(feats, nom_idx, attn_pad_mask=pad_mask)
        masked = loss_fn(logits, labels) * pair_mask
        loss = masked.sum() / pair_mask.sum()

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        total_loss_sum += masked.sum().item()
        total_pairs += pair_mask.sum().item()

    return total_loss_sum / max(total_pairs, 1)


def train_stage1(
    learning_rate: float = 1e-3,
    max_epochs: int = 10,
    patience: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(f"\nBuilding Stage 1 training data...")
    train_data, val_data = build_stage1_data()
    print(f"Train rows: {len(train_data)}, Val rows: {len(val_data)}")

    print(f"\nTraining Stage 1 on {device}")
    model = MiniTransformer(n_heads=4, n_layers=1).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\n=== Model Complexity ===")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {model_bytes / 1024:.1f} KB")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    train_batches = make_batches(train_data)
    val_batches = make_batches(val_data)
    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        random.shuffle(train_batches)

        model.train()
        avg_loss = run_epoch(model, train_batches, loss_fn, optimizer, device)
        print(f"Loss: {avg_loss:.6f}")

        model.eval()
        with torch.no_grad():
            avg_val_loss = run_epoch(model, val_batches, loss_fn, None, device)
        print(f"Val loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            model_path = MODELS_DIR / "stage1_mini_transformer.pt"
            torch.save(model.state_dict(), model_path)
            print(f"✓ Best model saved")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n⊘ Early stopping at epoch {epoch + 1}")
                print(f"Best: epoch {best_epoch + 1}, val_loss: {best_val_loss:.6f}")
                break

    print(f"\n✓ Training complete")


if __name__ == "__main__":
    train_stage1()
