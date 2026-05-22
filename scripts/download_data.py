import gzip
import re
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


DATA_DIR = Path(__file__).parent.parent / "data"
CONLL_DIR = Path("/home/masterkenway/Downloads/att-coref-master/data/conll-2012")


def _parse_conll_file(path: Path) -> list[dict]:
    docs = []
    sentences = []
    current_sent = []
    doc_id = None
    open_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    clusters: dict[str, list[list[int]]] = defaultdict(list)
    sent_idx = 0

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#begin document"):
                doc_id = line.split("(")[1].split(")")[0]
                sentences = []
                current_sent = []
                open_spans = defaultdict(list)
                clusters = defaultdict(list)
                sent_idx = 0
            elif line.startswith("#end document"):
                if current_sent:
                    sentences.append(current_sent)
                docs.append({
                    "doc_id": doc_id,
                    "sentences": sentences,
                    "mention_clusters": [v for v in clusters.values() if len(v) >= 2],
                })
            elif line == "":
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = []
                    sent_idx += 1
            else:
                parts = line.split()
                word = parts[3]
                coref_col = parts[-1]
                current_sent.append(word)
                tok_idx = len(current_sent) - 1

                if coref_col != "-":
                    for mention in re.findall(r"[\(\d\|\)]+", coref_col):
                        for part in mention.split("|"):
                            if part.startswith("(") and part.endswith(")"):
                                eid = part[1:-1]
                                clusters[eid].append([sent_idx, tok_idx, tok_idx + 1])
                            elif part.startswith("("):
                                eid = part[1:]
                                open_spans[eid].append((sent_idx, tok_idx))
                            elif part.endswith(")"):
                                eid = part[:-1]
                                if open_spans[eid]:
                                    s_idx, s_tok = open_spans[eid].pop()
                                    clusters[eid].append([s_idx, s_tok, tok_idx + 1])
    return docs


def parse_conll2012() -> None:
    dest = DATA_DIR / "conll2012"
    if dest.exists():
        print(f"[skip] conll2012 already exists at {dest}")
        return
    print("[parse] conll2012 from att-coref-master")
    splits = {
        "train": CONLL_DIR / "all_train.v4_gold_conll.gz",
        "validation": CONLL_DIR / "all_development.v4_gold_conll.gz",
        "test": CONLL_DIR / "all_test.v4_gold_conll.gz",
    }
    ds_dict = {}
    for split, path in splits.items():
        docs = _parse_conll_file(path)
        ds_dict[split] = Dataset.from_list(docs)
        print(f"  {split}: {len(docs)} docs")
    DatasetDict(ds_dict).save_to_disk(str(dest))
    print(f"[done] conll2012 saved to {dest}")


def download_preco() -> None:
    dest = DATA_DIR / "preco"
    if dest.exists():
        print(f"[skip] preco already exists at {dest}")
        return
    print("[download] coref-data/preco_raw")
    ds = load_dataset("coref-data/preco_raw")
    ds.save_to_disk(str(dest))
    print(f"[done] preco: train={ds['train'].num_rows}, val={ds['validation'].num_rows}")


def download_litbank() -> None:
    dest = DATA_DIR / "litbank"
    if dest.exists():
        print(f"[skip] litbank already exists at {dest}")
        return
    print("[download] coref-data/litbank_raw (split_0)")
    ds = load_dataset("coref-data/litbank_raw", "split_0")
    ds.save_to_disk(str(dest))
    print(f"[done] litbank: train={ds['train'].num_rows}, val={ds['validation'].num_rows}, test={ds['test'].num_rows}")


def download_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_preco()
    download_litbank()
    parse_conll2012()


if __name__ == "__main__":
    download_all()
