import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
import spacy.tokens
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk

from disambiguation.signals.stage1_intrasentence import MiniTransformer, extract_raw_attributes
from disambiguation.signals.train_full import _clusters, _sent_lens, _compute_depth

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
DATA_DIR = CACHE_DIR.parent / "data"
SPACY_TRF_DIR = DATA_DIR / "spacy_trf"
MODELS_DIR = CACHE_DIR / "models"
EMBEDDINGS_DIR = CACHE_DIR / "bge_embeddings"

DATASET_CONFIG = [
    ("preco", ["train"]),
    ("litbank", ["train", "validation", "test"]),
    ("corefud", ["train", "validation"]),
    ("conll2012", ["train", "validation", "test"]),
]


def generate_stage1_training_data():
    vocab = spacy.blank("en").vocab
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
            n_sents = 0
            n_filtered = 0

            for doc_id, (sample, full_doc) in enumerate(tqdm(
                zip(split_ds, doc_iter, strict=True),
                desc=f"Stage 1 {ds_name}/{split_name}",
                total=len(split_ds),
            )):
                doc_clusters = _clusters(ds_name, sample)
                sls = _sent_lens(ds_name, sample)
                sents = list(full_doc.sents)

                for sent_idx, sent_len in enumerate(sls):
                    n_sents += 1
                    sent = sents[sent_idx]

                    mention_map = {}
                    nominal_positions = []

                    for token_idx in range(sent_len):
                        tok = sent[token_idx]
                        if tok.pos_ in {"PRON", "NOUN", "PROPN"}:
                            nominal_positions.append(token_idx)

                    for cluster_id, cluster in enumerate(doc_clusters):
                        for mention in cluster:
                            m_sent_idx, m_start, m_end = mention
                            if m_sent_idx == sent_idx:
                                for m_tok_idx in range(m_start, m_end):
                                    mention_map[m_tok_idx] = cluster_id

                    if len(mention_map) < 2 or len(nominal_positions) < 2:
                        continue

                    n_filtered += 1
                    token_attrs = extract_raw_attributes(sent, [sent_len])

                    emb_path = EMBEDDINGS_DIR / f"{ds_name}_{split_name}_{doc_id}_{sent_idx}.npy"
                    if emb_path.exists():
                        sent_embedding = np.load(emb_path)
                    else:
                        sent_embedding = np.zeros(384, dtype=np.float32)

                    sent_emb_reshaped = sent_embedding.reshape(12, 32).astype(np.float32)
                    feature_matrix = np.vstack([token_attrs, sent_emb_reshaped])

                    n_nominals = len(nominal_positions)
                    pair_labels = np.zeros((n_nominals, n_nominals), dtype=np.float32)

                    for i, nom_i in enumerate(nominal_positions):
                        for j, nom_j in enumerate(nominal_positions):
                            cluster_i = mention_map.get(nom_i)
                            cluster_j = mention_map.get(nom_j)
                            if (cluster_i is not None and cluster_j is not None and
                                cluster_i == cluster_j):
                                pair_labels[i, j] = 1.0

                    yield feature_matrix, pair_labels

            n_sents_total += n_sents
            n_filtered_total += n_filtered
            rate = n_filtered * 100.0 / max(n_sents, 1)
            print(f"  {ds_name}/{split_name}: {n_filtered}/{n_sents} ({rate:.1f}%)")

    print(f"\nTotal Stage 1 data: {n_filtered_total}/{n_sents_total} ({n_filtered_total*100.0/max(n_sents_total, 1):.1f}%)")


def train_stage1(
    learning_rate: float = 1e-3,
    max_epochs: int = 10,
    patience: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(f"Training Stage 1 on {device}")

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

    first_epoch = True

    for epoch in range(max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        model.train()
        total_loss = 0.0
        n_examples = 0
        token_counts = []
        nominal_counts = []

        for feature_matrix, pair_labels in generate_stage1_training_data():
            feature_t = torch.from_numpy(feature_matrix).float().unsqueeze(0).to(device)
            labels_t = torch.from_numpy(pair_labels).float().to(device)

            n_tokens = feature_matrix.shape[0] - 12
            n_nominals = pair_labels.shape[0]
            token_counts.append(n_tokens)
            nominal_counts.append(n_nominals)

            optimizer.zero_grad()
            pair_scores = model(feature_t).squeeze(0)

            if pair_scores.shape != labels_t.shape:
                continue

            loss = loss_fn(pair_scores, labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            total_loss += loss.item()
            n_examples += 1

        if n_examples == 0:
            print("No training examples!")
            break

        avg_loss = total_loss / n_examples

        if first_epoch and token_counts:
            import numpy as np
            token_counts = np.array(token_counts)
            nominal_counts = np.array(nominal_counts)

            avg_tokens = token_counts.mean()
            avg_nominals = nominal_counts.mean()

            transformer_flops = 2 * avg_tokens * avg_tokens * 32 * 4
            pair_head_flops = 2 * avg_nominals * avg_nominals * 64 * 32
            total_flops_per_ex = transformer_flops + pair_head_flops
            total_flops = n_examples * max_epochs * total_flops_per_ex

            print(f"\n=== Complexity (actual data) ===")
            print(f"Examples: {n_examples}")
            print(f"Tokens: min={token_counts.min()}, max={token_counts.max()}, avg={avg_tokens:.1f}")
            print(f"Nominals: min={nominal_counts.min()}, max={nominal_counts.max()}, avg={avg_nominals:.1f}")
            print(f"FLOPs/example: {total_flops_per_ex/1e6:.1f}M")
            print(f"Total FLOPs ({max_epochs} epochs): {total_flops/1e9:.1f}B")

            if device == "cuda":
                est_sec = total_flops / (100 * 1e12)
                print(f"Est. time (100 TFLOPS): {est_sec/3600:.1f} hours")

            first_epoch = False

        print(f"Loss: {avg_loss:.6f} ({n_examples} examples)")

        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for feature_matrix, pair_labels in generate_stage1_training_data():
                feature_t = torch.from_numpy(feature_matrix).float().unsqueeze(0).to(device)
                labels_t = torch.from_numpy(pair_labels).float().to(device)

                pair_scores = model(feature_t).squeeze(0)

                if pair_scores.shape != labels_t.shape:
                    continue

                loss = loss_fn(pair_scores, labels_t)
                val_loss += loss.item()
                n_val += 1

        if n_val == 0:
            print("No validation examples!")
            break

        avg_val_loss = val_loss / n_val
        print(f"Val loss: {avg_val_loss:.6f} ({n_val} examples)")

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
