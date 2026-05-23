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


def _make_doc(nlp: spacy.Language, tokens: list[str], sent_lens: list[int]) -> spacy.tokens.Doc:
    sent_starts = []
    for sent_len in sent_lens:
        sent_starts.append(True)
        sent_starts.extend([False] * (sent_len - 1))
    return spacy.tokens.Doc(nlp.vocab, words=tokens, sent_starts=sent_starts)


def _set_transformer_batch_size(nlp: spacy.Language, batch_size: int) -> None:
    trf = nlp.get_pipe("transformer")
    def find_strided(m):
        if m.name == "with_strided_spans":
            return m
        for l in m.layers:
            result = find_strided(l)
            if result:
                return result
        return None
    ws = find_strided(trf.model)
    if ws:
        ws.attrs["batch_size"] = batch_size
        print(f"  with_strided_spans batch_size set to {batch_size}")


def _make_doc(nlp: spacy.Language, tokens: list[str], sent_lens: list[int]) -> spacy.tokens.Doc:
    sent_starts = []
    for sent_len in sent_lens:
        sent_starts.append(True)
        sent_starts.extend([False] * (sent_len - 1))
    return spacy.tokens.Doc(nlp.vocab, words=tokens, sent_starts=sent_starts)


def _extract_features(spacy_docs: list, doc_sent_lens: list[list[int]]) -> np.ndarray:
    rows = []
    for spacy_doc, sent_lens in zip(spacy_docs, doc_sent_lens):
        abs_pos = 0
        for sent_len in sent_lens:
            sent_start = abs_pos
            for _ in range(sent_len):
                rows.append(_extract_token_features(spacy_doc[abs_pos], sent_start, sent_len))
                abs_pos += 1
    return np.array(rows, dtype=np.float32)


def _run_spacy(nlp: spacy.Language, doc_token_lists: list[list[str]], doc_sent_lens: list[list[int]], batch_size: int) -> np.ndarray:
    import torch
    start = time.time()
    spacy_docs = []
    for i, doc in enumerate(nlp.pipe(
        (_make_doc(nlp, doc_token_lists[i], doc_sent_lens[i]) for i in range(len(doc_token_lists))),
        batch_size=batch_size
    )):
        spacy_docs.append(doc)
        if (i + 1) % 256 == 0:
            torch.cuda.empty_cache()
    print(f"  pipeline: {time.time()-start:.1f}s")
    data = _extract_features(spacy_docs, doc_sent_lens)
    print(f"  features: {time.time()-start:.1f}s, {data.shape}")
    return data


def _append_to_cache(data: np.ndarray, sent_lens_flat: list[int]) -> None:
    path = CACHE_DIR / "token_infos.npz"
    if path.exists():
        existing = np.load(path)
        existing_data = existing["data"]
        existing_offsets = existing["offsets"]
        token_offset = int(existing_offsets[-1])
        new_offsets = [token_offset]
        for sl in sent_lens_flat:
            token_offset += sl
            new_offsets.append(token_offset)
        merged_data = np.vstack([existing_data, data])
        merged_offsets = np.concatenate([existing_offsets, np.array(new_offsets[1:], dtype=np.int64)])
    else:
        token_offset = 0
        offsets = [0]
        for sl in sent_lens_flat:
            token_offset += sl
            offsets.append(token_offset)
        merged_data = data
        merged_offsets = np.array(offsets, dtype=np.int64)
    np.savez_compressed(path, data=merged_data, offsets=merged_offsets)


def precompute_unified() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["ner", "lemmatizer", "senter"])
    _set_transformer_batch_size(nlp, 32)
    print(f"Pipeline: {nlp.pipe_names}")
    doc_boundaries = []
    clusters_by_doc = []
    sent_offset = 0

    # PreCo
    print("Processing PreCo...")
    preco_ds = load_from_disk("data/preco")["train"]
    doc_token_lists, doc_sent_lens = [], []
    for doc_idx in range(preco_ds.num_rows):
        sents = preco_ds[doc_idx]["sentences"]
        doc_token_lists.append([tok for sent in sents for tok in sent])
        doc_sent_lens.append([len(sent) for sent in sents])
        start = sent_offset
        sent_offset += len(sents)
        doc_boundaries.append((start, sent_offset, "preco", doc_idx))
        clusters_by_doc.append(preco_ds[doc_idx]["mention_clusters"])
    data = _run_spacy(nlp, doc_token_lists, doc_sent_lens, batch_size=8)
    _append_to_cache(data, [sl for lens in doc_sent_lens for sl in lens])

    # LitBank
    print("Processing LitBank...")
    litbank_ds = load_from_disk("data/litbank")["train"]
    doc_token_lists, doc_sent_lens = [], []
    for doc_idx in range(litbank_ds.num_rows):
        sample = litbank_ds[doc_idx]
        sents = sample["sentences"]
        doc_token_lists.append([tok for sent in sents for tok in sent])
        doc_sent_lens.append([len(sent) for sent in sents])
        start = sent_offset
        sent_offset += len(sents)
        doc_boundaries.append((start, sent_offset, "litbank", doc_idx))
        clusters_by_doc.append([[[m[0], m[1], m[2] + 1] for m in chain] for chain in sample["coref_chains"]])
    data = _run_spacy(nlp, doc_token_lists, doc_sent_lens, batch_size=8)
    _append_to_cache(data, [sl for lens in doc_sent_lens for sl in lens])

    # CorefUD
    print("Processing CorefUD...")
    corefud_ds = load_from_disk("data/corefud")
    doc_token_lists, doc_sent_lens = [], []
    for split_name in ["train", "validation"]:
        for doc_idx, sample in enumerate(corefud_ds[split_name]):
            sent_id_to_local = {}
            sents = []
            for si, sent in enumerate(sample["sentences"]):
                tokens = [tok["form"] for tok in sent["tokens"]]
                sents.append(tokens)
                sent_id_to_local[sent["sent_id"]] = si
            doc_token_lists.append([tok for sent in sents for tok in sent])
            doc_sent_lens.append([len(sent) for sent in sents])
            start = sent_offset
            sent_offset += len(sents)
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
    data = _run_spacy(nlp, doc_token_lists, doc_sent_lens, batch_size=8)
    _append_to_cache(data, [sl for lens in doc_sent_lens for sl in lens])

    total_sents = sent_offset
    total_docs = len(doc_boundaries)
    print(f"Combined: {total_docs} docs, {total_sents} sentences")

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


def precompute_conll2012() -> None:
    metadata = json.load(open(CACHE_DIR / "metadata.json"))
    dataset_ranges = json.load(open(CACHE_DIR / "dataset_ranges.json"))

    if "conll2012" in dataset_ranges:
        print("[skip] conll2012 already in cache")
        return

    clusters_by_doc = json.load(open(CACHE_DIR / "clusters.json"))

    ds = load_from_disk("data/conll2012")["train"]
    doc_token_lists = [[tok for sent in s["sentences"] for tok in sent] for s in ds]
    doc_sent_lens = [[len(sent) for sent in s["sentences"]] for s in ds]
    all_clusters = [s["mention_clusters"] for s in ds]
    total_new_sents = sum(len(sl) for sl in doc_sent_lens)
    print(f"CoNLL-2012: {len(ds)} docs, {total_new_sents} sentences")

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["ner", "lemmatizer", "senter"])
    _set_transformer_batch_size(nlp, 32)
    print(f"Pipeline: {nlp.pipe_names}")

    data = _run_spacy(nlp, doc_token_lists, doc_sent_lens, batch_size=8)
    _append_to_cache(data, [sl for lens in doc_sent_lens for sl in lens])

    sent_offset = metadata["total_sents"]
    doc_offset = metadata["total_docs"]
    new_doc_boundaries = []
    for doc_idx, sent_lens in enumerate(doc_sent_lens):
        doc_start_sent = sent_offset
        sent_offset += len(sent_lens)
        new_doc_boundaries.append((doc_start_sent, sent_offset, "conll2012", doc_idx))

    metadata["doc_boundaries"].extend(new_doc_boundaries)
    metadata["total_docs"] += len(ds)
    metadata["total_sents"] += total_new_sents
    with open(CACHE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)

    clusters_by_doc.extend(all_clusters)
    with open(CACHE_DIR / "clusters.json", "w") as f:
        json.dump(clusters_by_doc, f)

    dataset_ranges["conll2012"] = {"start_doc": doc_offset, "end_doc": doc_offset + len(ds)}
    with open(CACHE_DIR / "dataset_ranges.json", "w") as f:
        json.dump(dataset_ranges, f, indent=2)

    print(f"  Sources: {dataset_ranges}")


if __name__ == "__main__":
    precompute_unified()
    precompute_conll2012()
