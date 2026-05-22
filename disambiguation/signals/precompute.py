import json
import time
from pathlib import Path

import numpy as np
import spacy
from datasets import load_from_disk

from disambiguation.signals.abstract_features import (
    DEP_IDS,
    GENDER_IDS,
    NUMBER_IDS,
    POS_IDS,
    PRONTYPE_IDS,
)

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
TOKEN_INFO_DIM = 15


def _compute_depth(token) -> int:
    depth = 0
    current = token
    while current.head != current:
        depth += 1
        current = current.head
        if depth > 20:
            break
    return depth


def _extract_token_features(token, sent_start: int, sent_len: int) -> list[float]:
    morph = token.morph
    dep = token.dep_
    ti = token.i - sent_start
    head_rel = token.head.i - sent_start if token.head != token else -1
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
        ti / max(sent_len - 1, 1),
        head_rel,
        POS_IDS.get(token.head.pos_, len(POS_IDS)),
        DEP_IDS.get(token.head.dep_, len(DEP_IDS)),
    ]


def _load_conll2012_docs() -> list[tuple[list[list[str]], list]]:
    ds = load_from_disk("data/conll2012")["train"]
    return [(s["sentences"], s["mention_clusters"]) for s in ds]


def _doc_to_text_and_offsets(sentences: list[list[str]]) -> tuple[str, list[int]]:
    offsets = []
    tokens = []
    for sent in sentences:
        offsets.append(len(tokens))
        tokens.extend(sent)
    return " ".join(tokens), offsets


def precompute_unified() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    preco_ds = load_from_disk("data/preco")["train"]
    litbank_ds = load_from_disk("data/litbank")["train"]
    corefud_ds = load_from_disk("data/corefud")

    all_sentences: list[tuple[str, int]] = []
    doc_boundaries = []
    clusters_by_doc = []
    sent_offset = 0

    print("Loading PreCo...")
    for doc_idx in range(preco_ds.num_rows):
        sample = preco_ds[doc_idx]
        start = sent_offset
        for sent in sample["sentences"]:
            all_sentences.append((" ".join(sent), len(sent)))
            sent_offset += 1
        doc_boundaries.append((start, sent_offset, "preco", doc_idx))
        clusters_by_doc.append(sample["mention_clusters"])

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

    import thinc.api
    thinc.api.set_gpu_allocator("pytorch")
    thinc.api.require_gpu(0)
    nlp = spacy.load("en_core_web_trf", disable=["ner", "lemmatizer", "senter"])
    print(f"Pipeline: {nlp.pipe_names}")

    texts = [s[0] for s in all_sentences]
    token_counts = [s[1] for s in all_sentences]

    all_token_infos = []
    sent_offsets_arr = np.zeros(total_sents + 1, dtype=np.int64)
    token_offset = 0
    start_time = time.time()

    def sent_stream():
        for i, text in enumerate(texts):
            yield text, i

    results: dict[int, list] = {}
    for spacy_doc, sent_idx in nlp.pipe(sent_stream(), as_tuples=True, batch_size=4096):
        target_len = token_counts[sent_idx]
        sent_start = spacy_doc[0].i if len(spacy_doc) > 0 else 0
        features = [_extract_token_features(token, sent_start, target_len) for token in spacy_doc[:target_len]]
        while len(features) < target_len:
            features.append([0, 0, 3, 2, 0, 5, 0, 0, 0, 0, 0, 0.0, -1, 0, 0])
        results[sent_idx] = features
        if (sent_idx + 1) % 100000 == 0:
            print(f"  {sent_idx+1}/{total_sents} sents ({time.time()-start_time:.0f}s)")

    for i in range(total_sents):
        sent_offsets_arr[i] = token_offset
        for feat in results[i]:
            all_token_infos.append(feat)
        token_offset += token_counts[i]
    sent_offsets_arr[total_sents] = token_offset

    print(f"  spaCy done: {time.time()-start_time:.1f}s, {token_offset} tokens")

    token_data = np.array(all_token_infos, dtype=np.float32)
    np.savez_compressed(CACHE_DIR / "token_infos.npz", data=token_data, offsets=sent_offsets_arr)
    print(f"  token_infos: {token_data.shape}, {token_data.nbytes/(1024**2):.0f} MB")

    metadata = {
        "doc_boundaries": doc_boundaries,
        "total_docs": total_docs,
        "total_sents": total_sents,
        "token_info_dim": TOKEN_INFO_DIM,
    }
    with open(CACHE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)

    with open(CACHE_DIR / "clusters.json", "w") as f:
        json.dump(clusters_by_doc, f)

    dataset_ranges: dict[str, dict] = {}
    for i, (start, end, source, orig_idx) in enumerate(doc_boundaries):
        if source not in dataset_ranges:
            dataset_ranges[source] = {"start_doc": i, "end_doc": i + 1}
        else:
            dataset_ranges[source]["end_doc"] = i + 1
    with open(CACHE_DIR / "dataset_ranges.json", "w") as f:
        json.dump(dataset_ranges, f, indent=2)

    print(f"Cache complete. Sources: {dataset_ranges}")



    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    metadata = json.load(open(CACHE_DIR / "metadata.json"))
    dataset_ranges = json.load(open(CACHE_DIR / "dataset_ranges.json"))

    if "conll2012" in dataset_ranges:
        print("[skip] conll2012 already in cache")
        return

    clusters_by_doc = json.load(open(CACHE_DIR / "clusters.json"))
    existing = np.load(CACHE_DIR / "token_infos.npz")
    existing_data = existing["data"]
    existing_offsets = existing["offsets"]

    print("Loading CoNLL-2012...")
    docs = _load_conll2012_docs()
    total_new_sents = sum(len(sents) for sents, _ in docs)
    print(f"  {len(docs)} docs, {total_new_sents} sentences")

    import thinc.api
    thinc.api.set_gpu_allocator("pytorch")
    thinc.api.require_gpu(0)
    nlp = spacy.load("en_core_web_trf", disable=["ner", "lemmatizer", "senter"])

    all_token_infos = []
    new_sent_offsets = []  # token offset per sentence
    new_doc_boundaries = []
    new_clusters = []

    sent_offset = metadata["total_sents"]
    token_offset = int(existing_offsets[-1])
    doc_offset = metadata["total_docs"]

    start_time = time.time()

    def sent_stream():
        for doc_idx, (sentences, _) in enumerate(docs):
            for sent_idx, sent in enumerate(sentences):
                yield " ".join(sent), (doc_idx, sent_idx)

    doc_sent_lens = [[len(sent) for sent in sents] for sents, _ in docs]
    results: dict[int, dict[int, list]] = {i: {} for i in range(len(docs))}
    processed = 0

    for spacy_doc, (doc_idx, sent_idx) in nlp.pipe(sent_stream(), as_tuples=True, batch_size=4096):
        sent_len = doc_sent_lens[doc_idx][sent_idx]
        results[doc_idx][sent_idx] = [
            _extract_token_features(token, 0, sent_len)
            for token in spacy_doc
        ]
        processed += 1
        if processed % 50000 == 0:
            print(f"  {processed}/{total_new_sents} sents ({time.time() - start_time:.0f}s)")

    for doc_idx, (sentences, clusters) in enumerate(docs):
        doc_start_sent = sent_offset
        for sent_idx, sent in enumerate(sentences):
            sent_len = len(sent)
            new_sent_offsets.append(token_offset)
            for features in results[doc_idx][sent_idx]:
                all_token_infos.append(features)
            token_offset += sent_len
            sent_offset += 1
        new_doc_boundaries.append((doc_start_sent, sent_offset, "conll2012", doc_idx))
        new_clusters.append(clusters)

    print(f"  Spacy done: {time.time() - start_time:.1f}s")

    # Append token infos
    new_data = np.array(all_token_infos, dtype=np.float32)
    merged_data = np.vstack([existing_data, new_data])

    new_offsets = np.array(new_sent_offsets + [token_offset], dtype=np.int64)
    merged_offsets = np.concatenate([existing_offsets, new_offsets[1:]])

    np.savez_compressed(CACHE_DIR / "token_infos.npz", data=merged_data, offsets=merged_offsets)
    print(f"  token_infos: {merged_data.shape}")

    # Update metadata
    metadata["doc_boundaries"].extend(new_doc_boundaries)
    metadata["total_docs"] += len(docs)
    metadata["total_sents"] += total_new_sents
    with open(CACHE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Update clusters
    clusters_by_doc.extend(new_clusters)
    with open(CACHE_DIR / "clusters.json", "w") as f:
        json.dump(clusters_by_doc, f)

    # Update dataset ranges
    dataset_ranges["conll2012"] = {"start_doc": doc_offset, "end_doc": doc_offset + len(docs)}
    with open(CACHE_DIR / "dataset_ranges.json", "w") as f:
        json.dump(dataset_ranges, f, indent=2)

    print(f"  Sources: {dataset_ranges}")


if __name__ == "__main__":
    precompute_unified()
    precompute_conll2012()
