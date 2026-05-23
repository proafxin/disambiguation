import gc
import os
import time
from pathlib import Path

import numpy as np
import spacy
import spacy.tokens
import torch
from datasets import load_from_disk
from tqdm import tqdm

from disambiguation.signals.abstract_features import (
    DEP_IDS,
    GENDER_IDS,
    NUMBER_IDS,
    POS_IDS,
    PRONTYPE_IDS,
)

MAX_TOKENS = 4000
N_FEATURES = 15
CHECKPOINT_INTERVAL = 500
CACHE_DIR = Path("cache/spacy_trf")
DATA_DIR = Path("data")

DATASETS = [
    ("preco", ["train", "validation"]),
    ("litbank", ["train", "validation", "test"]),
    ("corefud", ["train", "validation"]),
    ("conll2012", ["train", "validation", "test"]),
]


def find_strided_spans(model):
    if model.name == "with_strided_spans":
        return model
    for layer in model.layers:
        result = find_strided_spans(layer)
        if result:
            return result
    return None


def load_nlp():
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["senter", "lemmatizer"])
    trf = nlp.get_pipe("transformer")
    ws = find_strided_spans(trf.model)
    if ws:
        ws.attrs["batch_size"] = 384
        print(f"  with_strided_spans batch_size: {ws.attrs['batch_size']}")
    print(f"  pipeline: {nlp.pipe_names}")
    return nlp


def get_sentences(example, ds_name):
    if ds_name == "corefud":
        return [[tok["form"] for tok in sent["tokens"]] for sent in example["sentences"]]
    return [list(s) for s in example["sentences"]]


def split_into_chunks(sents, max_tokens):
    chunks = []
    cur_tokens = []
    cur_lens = []
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


def make_doc(nlp, tokens, sent_lens):
    sent_starts = []
    for sl in sent_lens:
        sent_starts.append(True)
        sent_starts.extend([False] * (sl - 1))
    return spacy.tokens.Doc(nlp.vocab, words=tokens, sent_starts=sent_starts)


def compute_depth(token):
    depth = 0
    cur = token
    while cur.head != cur and depth < 20:
        cur = cur.head
        depth += 1
    return depth


def fill_doc_features(doc, sent_lens, data, start_pos):
    pos = int(start_pos)
    abs_p = 0
    for sl in sent_lens:
        sent_start = abs_p
        for i in range(abs_p, abs_p + sl):
            tok = doc[i]
            morph = tok.morph
            dep = tok.dep_
            ti = i - sent_start
            head_rel = tok.head.i - sent_start if tok.head != tok else -1
            data[pos, 0] = POS_IDS.get(tok.pos_, len(POS_IDS))
            data[pos, 1] = DEP_IDS.get(dep, len(DEP_IDS))
            data[pos, 2] = GENDER_IDS.get(morph.get("Gender", ["unknown"])[0], 3)
            data[pos, 3] = NUMBER_IDS.get(morph.get("Number", ["unknown"])[0], 2)
            data[pos, 4] = int(morph.get("Person", ["0"])[0])
            data[pos, 5] = PRONTYPE_IDS.get(morph.get("PronType", ["unknown"])[0], 5)
            data[pos, 6] = int(dep in ("nsubj", "nsubj:pass", "nsubj:outer", "csubj"))
            data[pos, 7] = int(dep in ("obj", "iobj"))
            data[pos, 8] = int(dep == "nmod:poss")
            data[pos, 9] = compute_depth(tok)
            data[pos, 10] = len(list(tok.children))
            data[pos, 11] = ti / max(sl - 1, 1)
            data[pos, 12] = head_rel
            data[pos, 13] = POS_IDS.get(tok.head.pos_, len(POS_IDS))
            data[pos, 14] = DEP_IDS.get(tok.head.dep_, len(DEP_IDS))
            pos += 1
        abs_p += sl
    return pos


def collect_chunks(ds, ds_name):
    chunks = []
    for ex_idx, ex in enumerate(ds):
        sents = get_sentences(ex, ds_name)
        for chunk_tokens, chunk_sent_lens in split_into_chunks(sents, MAX_TOKENS):
            chunks.append((ex_idx, chunk_tokens, chunk_sent_lens))
    return chunks


def docs_generator(chunks, nlp):
    for _, chunk_tokens, chunk_sent_lens in chunks:
        yield make_doc(nlp, chunk_tokens, chunk_sent_lens)


def save_checkpoint(ckpt_path, ex_fill, n_done):
    tmp = ckpt_path.with_name(ckpt_path.stem + ".tmp.npz")
    np.savez(str(tmp), ex_fill=np.array(ex_fill, dtype=np.int64), n_done=np.array([n_done]))
    os.replace(tmp, ckpt_path)


def build_offsets(chunks, n):
    ex_token_counts = [0] * n
    for ex_idx, chunk_tokens, _ in chunks:
        ex_token_counts[ex_idx] += len(chunk_tokens)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        offsets[i + 1] = offsets[i] + ex_token_counts[i]
    return offsets


def process_split(nlp, ds_name, split):
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

    chunks = collect_chunks(ds, ds_name)
    n_chunks = len(chunks)
    offsets = build_offsets(chunks, n)
    total_tokens = int(offsets[-1])
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

    remaining = chunks[n_done:]
    pipe = nlp.pipe(docs_generator(remaining, nlp), batch_size=32)

    for i, ((ex_idx, _, chunk_sent_lens), doc) in enumerate(tqdm(
        zip(remaining, pipe), total=len(remaining), desc=f"{ds_name}/{split}"
    )):
        ex_fill[ex_idx] = fill_doc_features(doc, chunk_sent_lens, data, ex_fill[ex_idx])

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            data.flush()
            save_checkpoint(ckpt_path, ex_fill, n_done + i + 1)
            gc.collect()
            time.sleep(5)

    data.flush()
    del data

    os.replace(tmp_data, out_data)
    tmp_offsets = out_offsets.with_name(out_offsets.stem + ".tmp.npy")
    np.save(str(tmp_offsets), offsets)
    os.replace(tmp_offsets, out_offsets)

    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"  saved: {total_tokens} tokens, {n} examples")

    del ds, ds_dict, chunks, offsets, ex_fill
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nlp = load_nlp()
    for ds_name, splits in DATASETS:
        for split in splits:
            process_split(nlp, ds_name, split)
    print("\ndone")


main()
