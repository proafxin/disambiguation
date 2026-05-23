import gc
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import spacy
import spacy.tokens
import torch
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from disambiguation.signals.abstract_features import (
    DEP_IDS,
    ENT_TYPE_IDS,
    GENDER_IDS,
    NUMBER_IDS,
    POS_IDS,
    PRONTYPE_IDS,
)

MAX_TOKENS = 4000
MAX_DEPTH = 20
N_FEATURES = 16
CHECKPOINT_INTERVAL = 2000
CACHE_DIR = Path("data/spacy_trf")
DATA_DIR = Path("data")

DATASETS = [
    ("preco", ["train"]),
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
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["senter", "lemmatizer"])
    trf = nlp.get_pipe("transformer")
    ws = find_strided_spans(trf.model)
    if ws:
        ws.attrs["batch_size"] = 192
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


def compute_depth(token: spacy.tokens.Token) -> int:
    depth = 0
    cur = token
    while cur.head != cur and depth < MAX_DEPTH:
        cur = cur.head
        depth += 1
    return depth


def fill_doc_features(doc: spacy.tokens.Doc, sent_lens: list, data: np.ndarray, start_pos: int) -> int:
    n_tokens = sum(sent_lens)
    buf = np.empty((n_tokens, N_FEATURES), dtype=np.float32)
    pos = 0
    abs_p = 0
    for sl in sent_lens:
        sent_start = abs_p
        for i in range(abs_p, abs_p + sl):
            tok = doc[i]
            morph = tok.morph.to_dict()
            dep = tok.dep_
            ti = i - sent_start
            head_rel = tok.head.i - sent_start if tok.head != tok else -1
            buf[pos, 0] = POS_IDS.get(tok.pos_, len(POS_IDS))
            buf[pos, 1] = DEP_IDS.get(dep, len(DEP_IDS))
            buf[pos, 2] = GENDER_IDS.get(morph.get("Gender", "unknown"), 3)
            buf[pos, 3] = NUMBER_IDS.get(morph.get("Number", "unknown"), 2)
            buf[pos, 4] = int(morph.get("Person", "0"))
            buf[pos, 5] = PRONTYPE_IDS.get(morph.get("PronType", "unknown"), 5)
            buf[pos, 6] = int(dep in {"nsubj", "nsubj:pass", "nsubj:outer", "csubj"})
            buf[pos, 7] = int(dep in {"obj", "iobj"})
            buf[pos, 8] = int(dep == "nmod:poss")
            buf[pos, 9] = compute_depth(tok)
            buf[pos, 10] = tok.n_lefts + tok.n_rights
            buf[pos, 11] = ti / max(sl - 1, 1)
            buf[pos, 12] = head_rel
            buf[pos, 13] = POS_IDS.get(tok.head.pos_, len(POS_IDS))
            buf[pos, 14] = DEP_IDS.get(tok.head.dep_, len(DEP_IDS))
            buf[pos, 15] = ENT_TYPE_IDS.get(tok.ent_type_, len(ENT_TYPE_IDS))
            pos += 1
        abs_p += sl
    end_pos = int(start_pos) + n_tokens
    data[start_pos:end_pos] = buf
    return end_pos


def save_checkpoint(ckpt_path: Path, ex_fill: list, n_done: int) -> None:
    tmp = ckpt_path.with_name(ckpt_path.stem + ".tmp.npz")
    np.savez(str(tmp), ex_fill=np.array(ex_fill, dtype=np.int64), n_done=np.array([n_done]))
    tmp.replace(ckpt_path)


def build_chunk_meta(ds: Dataset, ds_name: str) -> tuple[list, np.ndarray, int]:
    chunk_meta: list[tuple[int, list[int]]] = []
    doc_token_counts: list[int] = []
    for ex_idx, ex in enumerate(ds):
        sents = get_sentences(ex, ds_name)
        doc_token_counts.append(sum(len(s) for s in sents))
        for _, chunk_sent_lens in split_into_chunks(sents, MAX_TOKENS):
            chunk_meta.append((ex_idx, chunk_sent_lens))
    offsets = np.zeros(len(doc_token_counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(doc_token_counts)
    return chunk_meta, offsets, int(offsets[-1])


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


def process_split(nlp: spacy.Language, ds_name: str, split: str) -> None:
    out_data = CACHE_DIR / f"{ds_name}_{split}_data.npy"
    out_offsets = CACHE_DIR / f"{ds_name}_{split}_offsets.npy"
    tmp_data = CACHE_DIR / f"{ds_name}_{split}_data.tmp.npy"
    ckpt_path = CACHE_DIR / f"{ds_name}_{split}_checkpoint.npz"

    if out_data.exists() and out_offsets.exists():
        print(f"skip {ds_name}/{split} — cache exists")
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

    chunk_meta, offsets, total_tokens = build_chunk_meta(ds, ds_name)
    n_chunks = len(chunk_meta)
    print(f"  {n_chunks} chunks, {total_tokens} tokens")

    if tmp_data.exists() and ckpt_path.exists():
        ckpt = np.load(str(ckpt_path))
        ex_fill = list(ckpt["ex_fill"])
        n_done = int(ckpt["n_done"][0])
        data = np.memmap(str(tmp_data), dtype=np.float32, mode="r+", shape=(total_tokens, N_FEATURES))
        print(f"  resuming from chunk {n_done}/{n_chunks}")
    else:
        if tmp_data.exists():
            tmp_data.unlink()
        data = np.memmap(str(tmp_data), dtype=np.float32, mode="w+", shape=(total_tokens, N_FEATURES))
        ex_fill = [int(offsets[i]) for i in range(n)]
        n_done = 0

    pipe = nlp.pipe(doc_stream(ds, ds_name, nlp, chunk_meta, n_done), batch_size=8)
    remaining = chunk_meta[n_done:]

    for i, (doc, (ex_idx, chunk_sent_lens)) in enumerate(
        tqdm(zip(pipe, remaining, strict=True), total=len(remaining), desc=f"{ds_name}/{split}")
    ):
        ex_fill[ex_idx] = fill_doc_features(doc, chunk_sent_lens, data, ex_fill[ex_idx])
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            data.flush()
            save_checkpoint(ckpt_path, ex_fill, n_done + i + 1)
            gc.collect()

    data.flush()
    del data

    tmp_data.replace(out_data)
    tmp_np = out_offsets.with_name(out_offsets.stem + ".tmp.npy")
    np.save(str(tmp_np), offsets)
    tmp_np.replace(out_offsets)

    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"  saved: {total_tokens} tokens, {n} examples")

    del ds, ds_dict, chunk_meta, offsets, ex_fill
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nlp = load_nlp()
    for ds_name, splits in DATASETS:
        for split in splits:
            process_split(nlp, ds_name, split)
    print("\ndone")


if __name__ == "__main__":
    main()
