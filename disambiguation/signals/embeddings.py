import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

EMBED_DIM = 384
BATCH_TOKEN_LIMIT = 1000000


def precompute_all_embeddings(
    dataset_path: str = "data/preco",
    split: str = "train",
    model_name: str = "all-MiniLM-L6-v2",
    num_docs: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    ds = load_from_disk(dataset_path)
    data = ds[split]

    if num_docs is None:
        num_docs = data.num_rows

    embedder = SentenceTransformer(model_name)

    all_texts = []
    all_token_counts = []
    index_map = []

    for doc_idx in range(num_docs):
        sample = data[doc_idx]
        for sent_idx, sent in enumerate(sample["sentences"]):
            all_texts.append(" ".join(sent))
            all_token_counts.append(len(sent))
            index_map.append((doc_idx, sent_idx))

    print(f"Total sentences to embed: {len(all_texts)}")

    all_embeddings = []
    batch = []
    batch_tokens = 0
    processed = 0

    for i, text in enumerate(all_texts):
        tok_count = all_token_counts[i]
        if batch_tokens + tok_count > BATCH_TOKEN_LIMIT and batch:
            embs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(embs.astype(np.float16))
            processed += len(batch)
            if processed % 100000 == 0:
                print(f"  Embedded {processed}/{len(all_texts)} sentences")
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += tok_count

    if batch:
        embs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embs.astype(np.float16))
        processed += len(batch)

    embeddings = np.vstack(all_embeddings)
    print(f"  Embedded {processed}/{len(all_texts)} sentences")
    print(f"Embedding matrix: {embeddings.shape}, {embeddings.nbytes / (1024**2):.1f} MB")

    return embeddings, index_map


class EmbeddingStore:
    def __init__(self, embeddings: np.ndarray, index_map: list[tuple[int, int]]) -> None:
        self.embeddings = embeddings
        self._lookup: dict[tuple[int, int], int] = {}
        for row_idx, (doc_idx, sent_idx) in enumerate(index_map):
            self._lookup[(doc_idx, sent_idx)] = row_idx

    def get_sentence_embedding(self, doc_idx: int, sent_idx: int) -> np.ndarray:
        row = self._lookup[(doc_idx, sent_idx)]
        return self.embeddings[row].astype(np.float32)

    def get_document_embeddings(self, doc_idx: int, num_sentences: int) -> np.ndarray:
        rows = [self._lookup[(doc_idx, si)] for si in range(num_sentences) if (doc_idx, si) in self._lookup]
        return self.embeddings[rows].astype(np.float32)

    @classmethod
    def load(cls, embeddings_path: str = "data/preco_sentence_embeddings.npz", index_path: str = "data/preco_embedding_index.json") -> "EmbeddingStore":
        import json
        data = np.load(embeddings_path)
        embeddings = data["embeddings"]
        with open(index_path) as f:
            index_map = [tuple(x) for x in json.load(f)]
        return cls(embeddings, index_map)


if __name__ == "__main__":
    embeddings, index_map = precompute_all_embeddings(num_docs=100)
    store = EmbeddingStore(embeddings, index_map)
    emb = store.get_sentence_embedding(0, 0)
    print(f"Test lookup (doc=0, sent=0): shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
