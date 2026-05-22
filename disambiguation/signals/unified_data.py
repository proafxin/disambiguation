import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_from_disk


@dataclass
class UnifiedDocument:
    doc_id: str
    sentences: list[list[str]]
    # All clusters use [sent_idx, start_token, end_token_exclusive]
    clusters: list[list[list[int]]]
    source: str  # "preco" or "litbank"


def load_preco_docs(num_docs: int | None = None) -> list[UnifiedDocument]:
    ds = load_from_disk("data/preco")
    data = ds["train"]
    n = num_docs if num_docs else data.num_rows
    docs = []
    for i in range(n):
        sample = data[i]
        docs.append(UnifiedDocument(
            doc_id=f"preco_{i}",
            sentences=sample["sentences"],
            clusters=sample["mention_clusters"],  # already exclusive end
            source="preco",
        ))
    return docs


def load_litbank_docs(num_docs: int | None = None) -> list[UnifiedDocument]:
    ds = load_from_disk("data/litbank")
    data = ds["train"]
    n = min(num_docs, data.num_rows) if num_docs else data.num_rows
    docs = []
    for i in range(n):
        sample = data[i]
        # Convert inclusive end to exclusive end
        clusters = []
        for chain in sample["coref_chains"]:
            cluster = []
            for mention in chain:
                si, st, en = mention
                cluster.append([si, st, en + 1])  # inclusive -> exclusive
            clusters.append(cluster)
        docs.append(UnifiedDocument(
            doc_id=f"litbank_{i}",
            sentences=sample["sentences"],
            clusters=clusters,
            source="litbank",
        ))
    return docs


def load_conll2012_docs(num_docs: int | None = None) -> list[UnifiedDocument]:
    ds = load_from_disk("data/conll2012")
    data = ds["train"]
    n = min(num_docs, data.num_rows) if num_docs else data.num_rows
    docs = []
    for i in range(n):
        sample = data[i]
        docs.append(UnifiedDocument(
            doc_id=sample["doc_id"],
            sentences=sample["sentences"],
            clusters=sample["mention_clusters"],
            source="conll2012",
        ))
    return docs


def load_combined(preco_docs: int = 500, litbank_docs: int = 80) -> list[UnifiedDocument]:
    docs = []
    docs.extend(load_preco_docs(preco_docs))
    docs.extend(load_litbank_docs(litbank_docs))
    return docs


def train_test_split_docs(
    docs: list[UnifiedDocument],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[UnifiedDocument], list[UnifiedDocument]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(docs))
    split_point = int(len(docs) * (1 - test_ratio))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    train_docs = [docs[i] for i in train_indices]
    test_docs = [docs[i] for i in test_indices]
    return train_docs, test_docs


if __name__ == "__main__":
    docs = load_combined(preco_docs=100, litbank_docs=80)
    train_docs, test_docs = train_test_split_docs(docs)

    preco_train = sum(1 for d in train_docs if d.source == "preco")
    litbank_train = sum(1 for d in train_docs if d.source == "litbank")
    preco_test = sum(1 for d in test_docs if d.source == "preco")
    litbank_test = sum(1 for d in test_docs if d.source == "litbank")

    print(f"Total docs: {len(docs)}")
    print(f"Train: {len(train_docs)} (preco={preco_train}, litbank={litbank_train})")
    print(f"Test: {len(test_docs)} (preco={preco_test}, litbank={litbank_test})")

    # Stats
    total_clusters = sum(len(d.clusters) for d in docs)
    multi_clusters = sum(sum(1 for c in d.clusters if len(c) >= 2) for d in docs)
    print(f"Total clusters: {total_clusters}, Multi-mention: {multi_clusters}")
