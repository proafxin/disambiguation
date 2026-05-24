import pickle
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
    ("preco", ["train"]),
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
                for cluster_id, cluster in enumerate(doc_clusters):
                    for m_sent_idx, m_start, m_end in cluster:
                        token_map = mentions_by_sent.setdefault(m_sent_idx, {})
                        for m_tok_idx in range(m_start, m_end):
                            token_map[m_tok_idx] = cluster_id

                for sent_idx, sent_len in enumerate(sls):
                    n_sents += 1

                    mention_map = mentions_by_sent.get(sent_idx, {})
                    if len(mention_map) < 2:
                        continue

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
                    pair_labels = np.zeros((sent_len, sent_len), dtype=np.float32)
                    for i in nominal_positions:
                        for j in nominal_positions:
                            ci = mention_map.get(i)
                            cj = mention_map.get(j)
                            if ci is not None and cj is not None and ci == cj:
                                pair_labels[i, j] = 1.0

                    key = (ds_name, split_name, doc_id, sent_idx)
                    if key not in embeddings:
                        missing_keys.append(key)
                        missing_texts.append(sent.text)
                    rows.append((is_train, key, token_attrs, pair_labels))

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
    for is_train, key, token_attrs, pair_labels in rows:
        sent_emb_reshaped = embeddings[key].reshape(12, 32)
        feature_matrix = np.vstack([token_attrs, sent_emb_reshaped])
        (train_data if is_train else val_data).append((feature_matrix, pair_labels))

    with FEATURES_CACHE.open("wb") as f:
        pickle.dump((train_data, val_data), f)
    print(f"Cached features to {FEATURES_CACHE.name}")

    return train_data, val_data


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
    loss_fn = nn.BCEWithLogitsLoss()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        model.train()
        total_loss = 0.0
        n_rows = 0

        for feature_matrix, pair_labels in train_data:
            feature_t = torch.from_numpy(feature_matrix).float().unsqueeze(0).to(device)
            labels_t = torch.from_numpy(pair_labels).float().to(device)

            optimizer.zero_grad()
            pair_scores = model(feature_t).squeeze(0)
            loss = loss_fn(pair_scores, labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            total_loss += loss.item()
            n_rows += 1

        if n_rows == 0:
            print("No training data!")
            break

        avg_loss = total_loss / n_rows
        print(f"Loss: {avg_loss:.6f} ({n_rows} rows)")

        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for feature_matrix, pair_labels in val_data:
                feature_t = torch.from_numpy(feature_matrix).float().unsqueeze(0).to(device)
                labels_t = torch.from_numpy(pair_labels).float().to(device)

                pair_scores = model(feature_t).squeeze(0)
                loss = loss_fn(pair_scores, labels_t)
                val_loss += loss.item()
                n_val += 1

        if n_val == 0:
            print("No validation data!")
            break

        avg_val_loss = val_loss / n_val
        print(f"Val loss: {avg_val_loss:.6f} ({n_val} rows)")

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
