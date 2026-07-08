import argparse

from datasets import load_dataset

from disambiguation.paths import DATA_DIR

HF_SOURCES = {
    "preco": ("coref-data/preco_raw", None),
    "litbank": ("coref-data/litbank_raw", "split_0"),
    "corefud": ("coref-data/corefud_raw", "en_gum-corefud"),
}


def build(name: str) -> None:
    hf_id, config = HF_SOURCES[name]
    raw = load_dataset(hf_id, config) if config else load_dataset(hf_id)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.save_to_disk(str(DATA_DIR / name))
    print(f"{name} -> {DATA_DIR / name} " + str({k: len(v) for k, v in raw.items()}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=list(HF_SOURCES))
    args = ap.parse_args()
    for step in args.steps:
        build(step)


if __name__ == "__main__":
    main()
