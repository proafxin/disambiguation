import json
import random

import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2TokenizerFast

from disambiguation.parsing.loader import load_parser
from disambiguation.signals.extraction import parse_document
from disambiguation.signals.pairs import MentionPair, build_training_pairs
from disambiguation.signals.scorer import LinkScorer, compute_pair_features, pair_features_to_tensor


class PairDataset(Dataset):
    def __init__(self, features: list[Tensor], labels: list[int]) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def prepare_dataset(
    num_docs: int = 50,
    context_window: int = 5,
) -> tuple[list[Tensor], list[int]]:
    parser = load_parser(device="cuda", dtype=torch.float16)
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    config_path = hf_hub_download(repo_id="ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt", filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    ds = load_from_disk("data/preco")

    all_features = []
    all_labels = []

    for doc_idx in range(num_docs):
        sample = ds["train"][doc_idx]
        parsed_doc = parse_document(sample["sentences"], parser, tokenizer, config, device="cuda")
        pairs = build_training_pairs(parsed_doc, sample["mention_clusters"], context_window=context_window)

        for pair in pairs:
            features = compute_pair_features(pair.anchor, pair.candidate)
            feature_tensor = pair_features_to_tensor(features)
            all_features.append(feature_tensor)
            all_labels.append(pair.label)

    return all_features, all_labels


def train(
    num_docs: int = 50,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> LinkScorer:
    print("Preparing dataset...")
    features, labels = prepare_dataset(num_docs=num_docs)

    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"Dataset: {len(labels)} pairs ({pos_count} positive, {neg_count} negative)")

    # Compute class weight for imbalanced data
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32, device="cuda")

    dataset = PairDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LinkScorer().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_labels in loader:
            batch_features = batch_features.cuda()
            batch_labels = batch_labels.cuda()

            logits = model(batch_features)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_features.shape[0]
            preds = (logits > 0).float()
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.shape[0]

        avg_loss = total_loss / total
        accuracy = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}: loss={avg_loss:.4f} acc={accuracy:.4f}")

    return model


def evaluate(model: LinkScorer, num_docs: int = 10, start_doc: int = 50) -> None:
    parser = load_parser(device="cuda", dtype=torch.float16)
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    config_path = hf_hub_download(repo_id="ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt", filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    ds = load_from_disk("data/preco")

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    model.eval()
    with torch.no_grad():
        for doc_idx in range(start_doc, start_doc + num_docs):
            sample = ds["train"][doc_idx]
            parsed_doc = parse_document(sample["sentences"], parser, tokenizer, config, device="cuda")
            pairs = build_training_pairs(parsed_doc, sample["mention_clusters"], context_window=5)

            for pair in pairs:
                features = compute_pair_features(pair.anchor, pair.candidate)
                feature_tensor = pair_features_to_tensor(features).unsqueeze(0).cuda()
                logit = model(feature_tensor)
                pred = int(logit.item() > 0)

                if pred == 1 and pair.label == 1:
                    tp += 1
                elif pred == 1 and pair.label == 0:
                    fp += 1
                elif pred == 0 and pair.label == 0:
                    tn += 1
                else:
                    fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

    print(f"\n=== Evaluation on docs {start_doc}-{start_doc + num_docs} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")


if __name__ == "__main__":
    model = train(num_docs=50, epochs=20, batch_size=64)
    evaluate(model, num_docs=10, start_doc=50)
