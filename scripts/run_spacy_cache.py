import gc
import json
import pickle
import queue
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import spacy
import spacy.tokens
import torch
from datasets import Dataset, load_from_disk
from tqdm import tqdm

import cupy

MAX_TOKENS = 4000
CHECKPOINT_INTERVAL = 2000
CACHE_DIR = Path("data/spacy_trf")
DATA_DIR = Path("data")
NOMINAL_POS = {"PRON", "NOUN", "PROPN"}


def _compute_cosines(
    tok: np.ndarray, is_nominal: np.ndarray, chunk_sent_lens: list[int], base_sent_idx: int
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    off = 0
    for k, sl in enumerate(chunk_sent_lens):
        noms = np.where(is_nominal[off:off + sl])[0] + off
        if len(noms) >= 2:
            v = tok[noms]
            out[base_sent_idx + k] = (v @ v.T).astype(np.float16)
        off += sl
    return out

DATASETS = [
    ("preco", ["train", "validation"]),
    ("litbank", ["train", "validation", "test"]),
    ("corefud", ["train", "validation"]),
    ("conll2012", ["train", "validation", "test"]),
]


def find_strided_spans(model: object) -> object | None:
    if model.name == "with_strided_spans":
        return model
    for layer in model.layers:
        result = find_strided_spans(layer)
        if result:
            return result
    return None


def load_nlp() -> spacy.Language:
    gpu = spacy.prefer_gpu()
    print(f"  GPU: {gpu}")
    nlp = spacy.load("en_core_web_trf", disable=["senter"])
    trf = nlp.get_pipe("transformer")
    ws = find_strided_spans(trf.model)
    if ws:
        ws.attrs["batch_size"] = 32
        print(f"  with_strided_spans batch_size: {ws.attrs['batch_size']}")
    print(f"  pipeline: {nlp.pipe_names}")
    return nlp


def get_sentences(example: dict, ds_name: str) -> list:
    if ds_name == "corefud":
        return [[tok["form"] for tok in sent["tokens"]] for sent in example["sentences"]]
    return example["sentences"]


def split_into_chunks(sents: list, max_tokens: int) -> list[tuple[list, list]]:
    chunks = []
    cur_tokens: list = []
    cur_lens: list = []
    for sent in sents:
        if len(sent) > max_tokens:
            if cur_tokens:
                chunks.append((cur_tokens, cur_lens))
                cur_tokens, cur_lens = [], []
            chunks.append((list(sent), [len(sent)]))
            continue
        if cur_tokens and len(cur_tokens) + len(sent) > max_tokens:
            chunks.append((cur_tokens, cur_lens))
            cur_tokens, cur_lens = [], []
        cur_tokens.extend(sent)
        cur_lens.append(len(sent))
    if cur_tokens:
        chunks.append((cur_tokens, cur_lens))
    return chunks


def make_doc(nlp: spacy.Language, tokens: list, sent_lens: list) -> spacy.tokens.Doc:
    sent_starts = []
    for sl in sent_lens:
        sent_starts.append(True)
        sent_starts.extend([False] * (sl - 1))
    return spacy.tokens.Doc(nlp.vocab, words=tokens, sent_starts=sent_starts)


def build_chunk_meta(ds: Dataset, ds_name: str) -> tuple[list, list]:
    chunk_meta: list[tuple[int, list[int]]] = []
    doc_chunk_counts: list[int] = []
    for ex in ds:
        sents = get_sentences(ex, ds_name)
        chunks = split_into_chunks(sents, MAX_TOKENS)
        doc_chunk_counts.append(len(chunks))
        for _, chunk_sent_lens in chunks:
            chunk_meta.append((len(doc_chunk_counts) - 1, chunk_sent_lens))
    return chunk_meta, doc_chunk_counts


def doc_stream(
    ds: Dataset, ds_name: str, nlp: spacy.Language, chunk_meta: list, start_chunk: int
) -> Iterator[spacy.tokens.Doc]:
    chunk_idx = 0
    for ex in ds:
        sents = get_sentences(ex, ds_name)
        for chunk_tokens, _ in split_into_chunks(sents, MAX_TOKENS):
            if chunk_idx >= start_chunk:
                yield make_doc(nlp, chunk_tokens, chunk_meta[chunk_idx][1])
            chunk_idx += 1


def load_checkpoint(ds_name: str, split: str, ckpt_path: Path) -> tuple[int, int]:
    if ckpt_path.exists():
        with ckpt_path.open(encoding="utf-8") as f:
            ckpt = json.load(f)
        print(f"  resuming from chunk {ckpt['n_done']}")
        return ckpt["n_done"], ckpt.get("n_written_chunks", ckpt["n_done"] // CHECKPOINT_INTERVAL)
    chunk_files = list(CACHE_DIR.glob(f"{ds_name}_{split}_chunk_*.spacy"))
    if chunk_files:
        n_written = len(chunk_files)
        n_done = n_written * CHECKPOINT_INTERVAL
        print(f"  resuming from {n_written} chunk files (n_done={n_done})")
        return n_done, n_written
    return 0, 0


def merge_and_save(ds_name: str, split: str, out_docs: Path, doc_chunk_counts: list) -> None:
    chunk_files = sorted(
        CACHE_DIR.glob(f"{ds_name}_{split}_chunk_*.spacy"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    print(f"  merging {len(chunk_files)} chunks...")
    attrs = ["TAG", "POS", "MORPH", "LEMMA", "DEP", "HEAD", "ENT_TYPE", "ENT_IOB", "SENT_START"]

    blank_nlp = spacy.blank("en")
    all_chunks = []
    all_chunk_meta = []
    for cp in chunk_files:
        chunk_bin = spacy.tokens.DocBin(attrs=attrs, store_user_data=True).from_disk(cp)
        for doc in chunk_bin.get_docs(blank_nlp.vocab):
            all_chunks.append(doc)
            all_chunk_meta.append((doc.user_data["ex_idx"], doc.user_data["sent_lens"]))

    final_bin = spacy.tokens.DocBin(attrs=attrs, store_user_data=True)
    chunk_idx = 0
    for doc_id, n_chunks in enumerate(doc_chunk_counts):
        chunks = all_chunks[chunk_idx:chunk_idx + n_chunks]
        chunk_meta = all_chunk_meta[chunk_idx:chunk_idx + n_chunks]

        if n_chunks == 1:
            full_doc = chunks[0]
        else:
            full_doc = spacy.tokens.Doc.from_docs(chunks)

        combined_sent_lens = []
        for _, sent_lens in chunk_meta:
            combined_sent_lens.extend(sent_lens)

        ex_idx = chunk_meta[0][0]
        full_doc.user_data["sent_lens"] = combined_sent_lens
        full_doc.user_data["ex_idx"] = ex_idx

        final_bin.add(full_doc)
        chunk_idx += n_chunks

    print(f"  writing {out_docs.name}...")
    final_bin.to_disk(out_docs)
    for cp in chunk_files:
        cp.unlink()


def writer_worker(
    q: queue.Queue,
    ds_name: str,
    split: str,
    ckpt_path: Path,
    n_done_offset: int,
    n_written_start: int,
) -> None:
    attrs = ["TAG", "POS", "MORPH", "LEMMA", "DEP", "HEAD", "ENT_TYPE", "ENT_IOB", "SENT_START"]
    doc_bin = spacy.tokens.DocBin(attrs=attrs, store_user_data=True)
    n_written = n_written_start
    n_processed = 0

    while True:
        item = q.get()
        if item is None:
            break
        doc, ex_idx, chunk_sent_lens = item
        doc.user_data["sent_lens"] = chunk_sent_lens
        doc.user_data["ex_idx"] = ex_idx
        doc_bin.add(doc)
        n_processed += 1

        if n_processed % CHECKPOINT_INTERVAL == 0:
            cp = CACHE_DIR / f"{ds_name}_{split}_chunk_{n_written}.spacy"
            tmp = cp.with_suffix(".tmp.spacy")
            doc_bin.to_disk(tmp)
            tmp.rename(cp)
            with ckpt_path.open("w", encoding="utf-8") as f:
                json.dump({"n_done": n_done_offset + n_processed, "n_written_chunks": n_written + 1}, f)
            doc_bin = spacy.tokens.DocBin(attrs=attrs, store_user_data=True)
            n_written += 1

    cp = CACHE_DIR / f"{ds_name}_{split}_chunk_{n_written}.spacy"
    tmp = cp.with_suffix(".tmp.spacy")
    doc_bin.to_disk(tmp)
    tmp.rename(cp)


def process_split(nlp: spacy.Language, ds_name: str, split: str) -> None:
    out_docs = CACHE_DIR / f"{ds_name}_{split}.spacy"
    out_meta = CACHE_DIR / f"{ds_name}_{split}_meta.json"
    ckpt_path = CACHE_DIR / f"{ds_name}_{split}_checkpoint.json"
    cosine_path = CACHE_DIR / f"{ds_name}_{split}_cosine.pkl"

    if out_docs.exists() and out_meta.exists() and cosine_path.exists():
        print(f"skip {ds_name}/{split} — cache + cosine exist")
        return
    if not (DATA_DIR / ds_name).exists():
        print(f"skip {ds_name}/{split} — data dir not found")
        return

    ds_dict = load_from_disk(str(DATA_DIR / ds_name))
    if split not in ds_dict:
        print(f"skip {ds_name}/{split} — split not in dataset")
        return

    ds = ds_dict[split]
    n = len(ds)
    print(f"\n{ds_name}/{split}: {n} examples")

    chunk_meta, doc_chunk_counts = build_chunk_meta(ds, ds_name)
    n_chunks = len(chunk_meta)
    print(f"  {n_chunks} chunks")

    # Per-chunk sentence offset within its example (resume-safe sentence indexing).
    chunk_sent_offsets: list[int] = []
    prev_ex = None
    running = 0
    for c_ex_idx, c_sent_lens in chunk_meta:
        if c_ex_idx != prev_ex:
            running = 0
            prev_ex = c_ex_idx
        chunk_sent_offsets.append(running)
        running += len(c_sent_lens)

    n_done, n_written_chunks = load_checkpoint(ds_name, split, ckpt_path)

    cosine_cache: dict[tuple[int, int], np.ndarray] = {}

    q: queue.Queue = queue.Queue()
    writer = threading.Thread(
        target=writer_worker,
        args=(q, ds_name, split, ckpt_path, n_done, n_written_chunks),
        daemon=True,
    )
    writer.start()

    pipe = nlp.pipe(doc_stream(ds, ds_name, nlp, chunk_meta, n_done), batch_size=128)
    remaining = chunk_meta[n_done:]

    for ci, (doc, (ex_idx, chunk_sent_lens)) in enumerate(tqdm(
        zip(pipe, remaining, strict=True), total=len(remaining), desc=f"{ds_name}/{split}"
    )):
        chunk_idx = n_done + ci
        lhs = doc._.trf_data.last_hidden_layer_state
        raw = lhs.dataXd
        lengths = (cupy.asnumpy(lhs.lengths) if isinstance(lhs.lengths, cupy.ndarray) else np.asarray(lhs.lengths)).astype(np.int64)
        pos_arr = doc.to_array("POS")
        nom_ids = {doc.vocab.strings[p] for p in NOMINAL_POS}
        is_nominal = np.isin(pos_arr, list(nom_ids))
        wp_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        nom_wp_mask = np.zeros(int(raw.shape[0]), dtype=bool)
        for ti in np.where(is_nominal)[0]:
            s, ln = int(wp_starts[ti]), int(lengths[ti])
            if ln > 0:
                nom_wp_mask[s:s + ln] = True
        if isinstance(raw, cupy.ndarray):
            nom_rows = cupy.asnumpy(raw[cupy.asarray(nom_wp_mask)]).astype(np.float32)
        else:
            nom_rows = np.asarray(raw)[nom_wp_mask].astype(np.float32)
        nom_lengths = lengths[is_nominal]
        nom_wp_starts = np.concatenate([[0], np.cumsum(nom_lengths)[:-1]]) if len(nom_lengths) else np.array([], dtype=np.int64)
        tok = np.zeros((len(lengths), nom_rows.shape[1]), dtype=np.float32)
        for ii, ti in enumerate(np.where(is_nominal)[0]):
            ln = int(nom_lengths[ii])
            if ln > 0:
                tok[ti] = nom_rows[int(nom_wp_starts[ii]):int(nom_wp_starts[ii]) + ln].mean(axis=0)
        tok /= np.maximum(np.linalg.norm(tok, axis=1, keepdims=True), 1e-8)
        doc._.trf_data = None
        for sent_idx, cos in _compute_cosines(tok, is_nominal, chunk_sent_lens, chunk_sent_offsets[chunk_idx]).items():
            cosine_cache[(ex_idx, sent_idx)] = cos
        doc.tensor = doc.tensor[:0]
        q.put((doc, ex_idx, chunk_sent_lens))

    q.put(None)
    writer.join()

    with cosine_path.open("wb") as f:
        pickle.dump(cosine_cache, f)
    print(f"  cosine: {len(cosine_cache)} sentences")

    merge_and_save(ds_name, split, out_docs, doc_chunk_counts)

    with out_meta.open("w", encoding="utf-8") as f:
        json.dump({"n_examples": n}, f)

    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"  saved: {n_chunks} chunks, {n} examples")

    del ds, ds_dict, chunk_meta, doc_chunk_counts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def merge_existing_chunks(ds_name: str, split: str) -> None:
    out_docs = CACHE_DIR / f"{ds_name}_{split}.spacy"
    out_meta = CACHE_DIR / f"{ds_name}_{split}_meta.json"

    if out_docs.exists():
        print(f"skip {ds_name}/{split} — merged output already exists")
        return

    chunk_files = sorted(
        CACHE_DIR.glob(f"{ds_name}_{split}_chunk_*.spacy"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not chunk_files:
        print(f"skip {ds_name}/{split} — no chunk files found")
        return

    old_meta_path = CACHE_DIR / f"{ds_name}_{split}_meta.json"
    if old_meta_path.exists():
        with old_meta_path.open() as f:
            meta = json.load(f)
        doc_chunk_counts = meta.get("doc_chunk_counts")
        n_examples = meta.get("n_examples")
    else:
        print(f"skip {ds_name}/{split} — no metadata found")
        return

    if not doc_chunk_counts:
        print(f"skip {ds_name}/{split} — no doc_chunk_counts in metadata")
        return

    print(f"\n{ds_name}/{split}: merging {len(chunk_files)} chunks (no spacy)")
    merge_and_save(ds_name, split, out_docs, doc_chunk_counts)

    with out_meta.open("w") as f:
        json.dump({"n_examples": n_examples}, f)

    print(f"  done")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nlp = load_nlp()
    for ds_name, splits in DATASETS:
        for split in splits:
            process_split(nlp, ds_name, split)
    print("\ndone")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge-only":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for ds_name, splits in DATASETS:
            for split in splits:
                merge_existing_chunks(ds_name, split)
        print("\ndone (merge only)")
    else:
        main()
