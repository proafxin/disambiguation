import json
import time
from pathlib import Path

import numpy as np
import spacy
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

from disambiguation.signals.abstract_features import (
    DEP_IDS,
    GENDER_IDS,
    NUMBER_IDS,
    POS_IDS,
    PRONTYPE_IDS,
    _compute_depth,
)

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
BATCH_TOKEN_LIMIT = 1000000
TOKEN_INFO_DIM = 12


def _token_to_features(token) -> list[float]:
    morph = token.morph
    dep = token.dep_
    return [
        POS_IDS.get(token.pos_, len(POS_IDS)),
        DEP_IDS.get(dep, len(DEP_IDS)),
        GENDER_IDS.get(morph.get("Gender", ["unknown"])[0], 3),
        NUMBER_IDS.get(morph.get("Number", ["unknown"])[0], 2),
        int(morph.get("Person", ["0"])[0]),
        PRONTYPE_IDS.get(morph.get("PronType", ["unknown"])[0], 5),
        int(dep in ("nsubj", "nsubj:pass", "nsubj:outer", "csubj")),
        int(dep in ("obj", "iobj")),
        int(dep == "nmod:poss"),
        _compute_depth(token),
        len(list(token.children)),
        token.i / max(len(token.doc) - 1, 1),
    ]


def precompute_unified() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    preco_ds = load_from_disk("data/preco")["train"]
    litbank_ds = load_from_disk("data/litbank")["train"]

    # Collect all sentences from both datasets with unified doc IDs
    all_sentences = []  # (text, token_count)
    doc_boundaries = []  # (start_sent_idx, end_sent_idx, source, orig_idx)
    clusters_by_doc = []  # unified clusters per doc [sent_idx, start, end_exclusive]

    sent_offset = 0

    # PreCo
    for doc_idx in range(preco_ds.num_rows):
        sample = preco_ds[doc_idx]
        start = sent_offset
        for sent in sample["sentences"]:
            all_sentences.append((" ".join(sent), len(sent)))
            sent_offset += 1
        doc_boundaries.append((start, sent_offset, "preco", doc_idx))
        clusters_by_doc.append(sample["mention_clusters"])

    # LitBank (convert inclusive end to exclusive)
    for doc_idx in range(litbank_ds.num_rows):
        sample = litbank_ds[doc_idx]
        start = sent_offset
        for sent in sample["sentences"]:
            all_sentences.append((" ".join(sent), len(sent)))
            sent_offset += 1
        doc_boundaries.append((start, sent_offset, "litbank", doc_idx))
        clusters = []
        for chain in sample["coref_chains"]:
            cluster = [[m[0], m[1], m[2] + 1] for m in chain]
            clusters.append(cluster)
        clusters_by_doc.append(clusters)

    total_sents = len(all_sentences)
    total_docs = len(doc_boundaries)
    print(f"Combined: {total_docs} docs, {total_sents} sentences")

    # 1. Spacy NLP processing
    print("Step 1: Spacy NLP processing...")
    nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer"])

    all_token_infos = []  # flat list of feature vectors
    sent_offsets = np.zeros(total_sents + 1, dtype=np.int64)
    noun_texts = []
    noun_positions = []  # (global_sent_idx, token_idx)

    texts = [s[0] for s in all_sentences]
    token_counts = [s[1] for s in all_sentences]

    start_time = time.time()
    token_offset = 0
    for global_si, doc in enumerate(nlp.pipe(texts, batch_size=512)):
        target_len = token_counts[global_si]
        for ti, token in enumerate(doc):
            if ti >= target_len:
                break
            features = _token_to_features(token)
            all_token_infos.append(features)
            if features[0] in (POS_IDS["NOUN"], POS_IDS["PROPN"]):
                noun_texts.append(token.text)
                noun_positions.append((global_si, ti))

        # Pad if spacy produced fewer tokens
        produced = min(len(doc), target_len)
        for _ in range(target_len - produced):
            all_token_infos.append([0, 0, 3, 2, 0, 5, 0, 0, 0, 0, 0, 0.0])

        token_offset += target_len
        sent_offsets[global_si + 1] = token_offset

        if (global_si + 1) % 200000 == 0:
            print(f"  Spacy: {global_si + 1}/{total_sents}")

    spacy_time = time.time() - start_time
    print(f"  Spacy done: {spacy_time:.1f}s, {token_offset} tokens")

    # Save token infos as flat fp16 array
    token_data = np.array(all_token_infos, dtype=np.float16)
    np.savez_compressed(CACHE_DIR / "token_infos.npz", data=token_data, offsets=sent_offsets)
    print(f"  Token infos: {token_data.shape}, {token_data.nbytes / (1024**2):.0f} MB")

    # 2. Sentence embeddings
    print("Step 2: Sentence embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    sent_embeddings = []
    batch = []
    batch_tokens = 0
    start_time = time.time()

    for i, (text, tok_count) in enumerate(all_sentences):
        if batch_tokens + tok_count > BATCH_TOKEN_LIMIT and batch:
            embs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            sent_embeddings.append(embs.astype(np.float16))
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += tok_count

    if batch:
        embs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        sent_embeddings.append(embs.astype(np.float16))

    sent_emb_matrix = np.vstack(sent_embeddings)
    embed_time = time.time() - start_time
    np.savez_compressed(CACHE_DIR / "sentence_embeddings.npz", data=sent_emb_matrix)
    print(f"  Sentence embeddings: {sent_emb_matrix.shape}, {embed_time:.1f}s")

    # 3. Noun/PROPN mention embeddings
    print(f"Step 3: Mention embeddings ({len(noun_texts)} nouns)...")
    start_time = time.time()

    noun_embeddings = []
    batch = []
    batch_tokens = 0
    for text in noun_texts:
        tok_count = len(text.split()) + 1
        if batch_tokens + tok_count > BATCH_TOKEN_LIMIT and batch:
            embs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            noun_embeddings.append(embs.astype(np.float16))
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += tok_count

    if batch:
        embs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        noun_embeddings.append(embs.astype(np.float16))

    noun_emb_matrix = np.vstack(noun_embeddings) if noun_embeddings else np.zeros((0, 384), dtype=np.float16)
    noun_time = time.time() - start_time
    np.savez_compressed(CACHE_DIR / "noun_embeddings.npz", data=noun_emb_matrix)
    print(f"  Noun embeddings: {noun_emb_matrix.shape}, {noun_time:.1f}s")

    # 4. Save metadata
    metadata = {
        "doc_boundaries": doc_boundaries,
        "noun_positions": noun_positions,
        "total_docs": total_docs,
        "total_sents": total_sents,
    }
    with open(CACHE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Save clusters
    with open(CACHE_DIR / "clusters.json", "w") as f:
        json.dump(clusters_by_doc, f)

    # Summary
    total_cache_size = sum(f.stat().st_size for f in CACHE_DIR.iterdir()) / (1024**2)
    print(f"\nCache saved: {total_cache_size:.0f} MB total")
    print(f"  {total_docs} docs, {total_sents} sentences, {token_offset} tokens, {len(noun_texts)} noun mentions")


if __name__ == "__main__":
    precompute_unified()
