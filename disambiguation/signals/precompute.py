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
)

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
BATCH_TOKEN_LIMIT = 1000000
TOKEN_INFO_DIM = 15  # extended from 12


def _compute_depth(token) -> int:
    depth = 0
    current = token
    while current.head != current:
        depth += 1
        current = current.head
        if depth > 20:
            break
    return depth


def _token_to_features(token, sent_len: int) -> list[float]:
    morph = token.morph
    dep = token.dep_
    head_idx = token.head.i - (token.i - token.i % sent_len) if token.head != token else -1
    # Clamp head_idx to be relative to sentence start
    token_sent_start = token.i - (token.i % 10000)  # placeholder, we handle below
    return [
        POS_IDS.get(token.pos_, len(POS_IDS)),          # 0: pos
        DEP_IDS.get(dep, len(DEP_IDS)),                 # 1: dep
        GENDER_IDS.get(morph.get("Gender", ["unknown"])[0], 3),  # 2: gender
        NUMBER_IDS.get(morph.get("Number", ["unknown"])[0], 2),  # 3: number
        int(morph.get("Person", ["0"])[0]),              # 4: person
        PRONTYPE_IDS.get(morph.get("PronType", ["unknown"])[0], 5),  # 5: prontype
        int(dep in ("nsubj", "nsubj:pass", "nsubj:outer", "csubj")),  # 6: is_subject
        int(dep in ("obj", "iobj")),                     # 7: is_object
        int(dep == "nmod:poss"),                         # 8: is_possessive
        _compute_depth(token),                           # 9: depth_to_root
        len(list(token.children)),                       # 10: n_children
        token.i / max(len(token.doc) - 1, 1),           # 11: sent_position (relative)
        # NEW fields:
        token.head.i - token.doc[0].i if token.head != token else -1,  # 12: head_token_index (relative to sent)
        POS_IDS.get(token.head.pos_, len(POS_IDS)),     # 13: head_pos
        DEP_IDS.get(token.head.dep_, len(DEP_IDS)),     # 14: head_dep
    ]


def precompute_unified() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    preco_ds = load_from_disk("data/preco")["train"]
    litbank_ds = load_from_disk("data/litbank")["train"]
    corefud_ds = load_from_disk("data/corefud")

    all_sentences = []
    doc_boundaries = []
    clusters_by_doc = []
    sent_offset = 0

    # PreCo
    print("Loading PreCo...")
    for doc_idx in range(preco_ds.num_rows):
        sample = preco_ds[doc_idx]
        start = sent_offset
        for sent in sample["sentences"]:
            all_sentences.append((" ".join(sent), len(sent)))
            sent_offset += 1
        doc_boundaries.append((start, sent_offset, "preco", doc_idx))
        clusters_by_doc.append(sample["mention_clusters"])

    # LitBank
    print("Loading LitBank...")
    for doc_idx in range(litbank_ds.num_rows):
        sample = litbank_ds[doc_idx]
        start = sent_offset
        for sent in sample["sentences"]:
            all_sentences.append((" ".join(sent), len(sent)))
            sent_offset += 1
        doc_boundaries.append((start, sent_offset, "litbank", doc_idx))
        clusters = [[[m[0], m[1], m[2] + 1] for m in chain] for chain in sample["coref_chains"]]
        clusters_by_doc.append(clusters)

    # CorefUD
    print("Loading CorefUD...")
    for split_name in ["train", "validation"]:
        split_data = corefud_ds[split_name]
        for doc_idx in range(len(split_data)):
            sample = split_data[doc_idx]
            start = sent_offset
            sent_id_to_local = {}
            for si, sent in enumerate(sample["sentences"]):
                tokens = [tok["form"] for tok in sent["tokens"]]
                all_sentences.append((" ".join(tokens), len(tokens)))
                sent_id_to_local[sent["sent_id"]] = si
                sent_offset += 1
            doc_boundaries.append((start, sent_offset, "corefud", doc_idx))
            clusters = []
            for entity in sample["coref_entities"]:
                if len(entity) < 2:
                    continue
                cluster = []
                for mention in entity:
                    sid = mention["sent_id"]
                    if sid not in sent_id_to_local:
                        continue
                    si = sent_id_to_local[sid]
                    span = mention["span"]
                    if "-" in span:
                        parts = span.split("-")
                        st = int(parts[0]) - 1
                        en = int(parts[1])
                    else:
                        st = int(span) - 1
                        en = st + 1
                    cluster.append([si, st, en])
                if len(cluster) >= 2:
                    clusters.append(cluster)
            clusters_by_doc.append(clusters)

    total_sents = len(all_sentences)
    total_docs = len(doc_boundaries)
    print(f"Combined: {total_docs} docs, {total_sents} sentences")

    # Step 1: Spacy processing
    print("\nStep 1: Spacy NLP (extracting 15 features per token)...")
    nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer"])

    all_token_infos = []
    sent_offsets_arr = np.zeros(total_sents + 1, dtype=np.int64)
    noun_texts = []
    noun_positions = []

    token_counts = [s[1] for s in all_sentences]
    texts = [s[0] for s in all_sentences]

    start_time = time.time()
    token_offset = 0

    for global_si, doc in enumerate(nlp.pipe(texts, batch_size=512)):
        target_len = token_counts[global_si]
        for ti, token in enumerate(doc):
            if ti >= target_len:
                break
            # Compute head index relative to this sentence (not doc)
            head_rel = token.head.i - doc[0].i if token.head != token else -1
            morph = token.morph
            dep = token.dep_
            features = [
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
                ti / max(target_len - 1, 1),
                head_rel,
                POS_IDS.get(token.head.pos_, len(POS_IDS)),
                DEP_IDS.get(token.head.dep_, len(DEP_IDS)),
            ]
            all_token_infos.append(features)
            if features[0] in (POS_IDS["NOUN"], POS_IDS["PROPN"]):
                noun_texts.append(token.text)
                noun_positions.append((global_si, ti))

        produced = min(len(doc), target_len)
        for _ in range(target_len - produced):
            all_token_infos.append([0, 0, 3, 2, 0, 5, 0, 0, 0, 0, 0, 0.0, -1, 0, 0])

        token_offset += target_len
        sent_offsets_arr[global_si + 1] = token_offset

        if (global_si + 1) % 200000 == 0:
            elapsed = time.time() - start_time
            print(f"  Spacy: {global_si + 1}/{total_sents} ({elapsed:.0f}s)")

    spacy_time = time.time() - start_time
    print(f"  Spacy done: {spacy_time:.1f}s, {token_offset} tokens")

    token_data = np.array(all_token_infos, dtype=np.float32)
    np.savez_compressed(CACHE_DIR / "token_infos.npz", data=token_data, offsets=sent_offsets_arr)
    print(f"  Token infos: {token_data.shape}, {token_data.nbytes / (1024**2):.0f} MB")
    del all_token_infos  # free memory

    # Step 2: Sentence embeddings
    print("\nStep 2: Sentence embeddings...")
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
    print(f"  Sentence embeddings: {sent_emb_matrix.shape}, {time.time() - start_time:.1f}s")
    np.savez_compressed(CACHE_DIR / "sentence_embeddings.npz", data=sent_emb_matrix)
    del sent_embeddings

    # Step 3: Noun embeddings
    print(f"\nStep 3: Mention embeddings ({len(noun_texts)} nouns)...")
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
    print(f"  Noun embeddings: {noun_emb_matrix.shape}, {time.time() - start_time:.1f}s")
    np.savez_compressed(CACHE_DIR / "noun_embeddings.npz", data=noun_emb_matrix)

    # Step 4: Metadata
    metadata = {
        "doc_boundaries": doc_boundaries,
        "noun_positions": noun_positions,
        "total_docs": total_docs,
        "total_sents": total_sents,
        "token_info_dim": TOKEN_INFO_DIM,
    }
    with open(CACHE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)

    with open(CACHE_DIR / "clusters.json", "w") as f:
        json.dump(clusters_by_doc, f)

    # Dataset ranges
    dataset_ranges = {}
    for i, (start, end, source, orig_idx) in enumerate(doc_boundaries):
        if source not in dataset_ranges:
            dataset_ranges[source] = {"start_doc": i, "end_doc": i + 1}
        else:
            dataset_ranges[source]["end_doc"] = i + 1

    with open(CACHE_DIR / "dataset_ranges.json", "w") as f:
        json.dump(dataset_ranges, f, indent=2)

    total_cache = sum(f.stat().st_size for f in CACHE_DIR.iterdir()) / (1024**2)
    print(f"\nCache complete: {total_cache:.0f} MB")
    print(f"  Sources: {dataset_ranges}")
    print(f"  {total_docs} docs, {total_sents} sents, {token_offset} tokens, {len(noun_texts)} nouns")
    print(f"  Token features: {TOKEN_INFO_DIM} per token (pos, dep, gender, number, person, prontype, is_subj, is_obj, is_poss, depth, n_children, sent_pos, head_idx, head_pos, head_dep)")


if __name__ == "__main__":
    precompute_unified()
