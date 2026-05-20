from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path(__file__).parent.parent / "data"


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


if __name__ == "__main__":
    download_all()
