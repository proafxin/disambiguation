import gc
import json
from collections.abc import Iterator
from pathlib import Path

import spacy
import spacy.tokens
import torch
from datasets import Dataset, load_from_disk
from tqdm import tqdm

MAX_TOKENS = 4000
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
    gpu = spacy.prefer_gpu()
    print(f"  GPU: {gpu}")
    nlp = spacy.load("en_core_web_trf", disable=["senter", "lemmatizer"])
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
    for p in CACHE_DIR.glob(f"{ds_name}_{split}_chunk_*.spacy"):
        p.unlink()
    return 0, 0


def merge_and_save(ds_name: str, split: str, out_docs: Path) -> None:
    chunk_files = sorted(
        CACHE_DIR.glob(f"{ds_name}_{split}_chunk_*.spacy"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    final_bin = spacy.tokens.DocBin(store_user_data=True)
    for cp in chunk_files:
        final_bin.merge(spacy.tokens.DocBin(store_user_data=True).from_disk(cp))
    final_bin.to_disk(out_docs)
    for cp in chunk_files:
        cp.unlink()


def process_split(nlp: spacy.Language, ds_name: str, split: str) -> None:
    out_docs = CACHE_DIR / f"{ds_name}_{split}.spacy"
    out_meta = CACHE_DIR / f"{ds_name}_{split}_meta.json"
    ckpt_path = CACHE_DIR / f"{ds_name}_{split}_checkpoint.json"

    if out_docs.exists() and out_meta.exists():
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

    chunk_meta, doc_chunk_counts = build_chunk_meta(ds, ds_name)
    n_chunks = len(chunk_meta)
    print(f"  {n_chunks} chunks")

    n_done, n_written_chunks = load_checkpoint(ds_name, split, ckpt_path)

    pipe = nlp.pipe(doc_stream(ds, ds_name, nlp, chunk_meta, n_done), batch_size=8)
    remaining = chunk_meta[n_done:]
    doc_bin = spacy.tokens.DocBin(store_user_data=True)

    for i, (doc, (ex_idx, chunk_sent_lens)) in enumerate(
        tqdm(zip(pipe, remaining, strict=True), total=len(remaining), desc=f"{ds_name}/{split}")
    ):
        doc.user_data["sent_lens"] = chunk_sent_lens
        doc.user_data["ex_idx"] = ex_idx
        doc_bin.add(doc)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            chunk_path = CACHE_DIR / f"{ds_name}_{split}_chunk_{n_written_chunks}.spacy"
            doc_bin.to_disk(chunk_path)
            doc_bin = spacy.tokens.DocBin(store_user_data=True)
            n_written_chunks += 1
            with ckpt_path.open("w", encoding="utf-8") as f:
                json.dump({"n_done": n_done + i + 1, "n_written_chunks": n_written_chunks}, f)
            gc.collect()

    chunk_path = CACHE_DIR / f"{ds_name}_{split}_chunk_{n_written_chunks}.spacy"
    doc_bin.to_disk(chunk_path)
    n_written_chunks += 1

    merge_and_save(ds_name, split, out_docs)

    with out_meta.open("w", encoding="utf-8") as f:
        json.dump({"n_examples": n, "n_chunks": n_chunks, "doc_chunk_counts": doc_chunk_counts}, f)

    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"  saved: {n_chunks} chunks, {n} examples")

    del ds, ds_dict, chunk_meta, doc_chunk_counts, doc_bin
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
