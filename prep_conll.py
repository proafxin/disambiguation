import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

from ontonotes_parser import Ontonotes

RAW_DIR = Path("data/conll2012_raw")
OUT_DIR = "data/conll2012"
SPLIT_OF = {"train": "train", "development": "validation", "test": "test"}


def _split_of(path: Path) -> str | None:
    if "v4" not in path.parts or "english" not in path.parts:
        return None
    for part, split in SPLIT_OF.items():
        if part in path.parts:
            return split
    return None


def _file_record(conll_file: Path) -> dict:
    sentences: list[list[str]] = []
    clusters: dict[int, list[list[int]]] = {}
    document_id = conll_file.stem
    for si, sent in enumerate(Ontonotes().sentence_iterator(str(conll_file))):
        document_id = sent.document_id
        sentences.append(list(sent.words))
        for cid, (a, b) in sent.coref_spans:
            clusters.setdefault(int(cid), []).append([si, int(a), int(b) + 1])
    mention_clusters = [clusters[k] for k in sorted(clusters)]
    return {"doc_id": document_id, "sentences": sentences, "mention_clusters": mention_clusters}


def main() -> None:
    files = sorted(RAW_DIR.rglob("*gold_conll"))
    if not files:
        raise RuntimeError(f"no *gold_conll under {RAW_DIR} - download and extract the CoNLL-2012 zip first")
    buckets: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for p in files:
        split = _split_of(p)
        if split is not None:
            buckets[split].append(_file_record(p))
    d0 = buckets["validation"][0]
    print("SANITY cluster0:", d0["mention_clusters"][0][:2], "| words0:", d0["sentences"][0][:8], file=sys.stderr)
    out = DatasetDict({sp: Dataset.from_list(recs) for sp, recs in buckets.items()})
    out.save_to_disk(OUT_DIR)
    print("saved", OUT_DIR, {sp: len(v) for sp, v in buckets.items()}, file=sys.stderr)


if __name__ == "__main__":
    main()
