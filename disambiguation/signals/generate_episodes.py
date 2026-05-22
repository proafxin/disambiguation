import json
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
EPISODES_DIR = CACHE_DIR / "episodes"

# Global cache reference (loaded once per worker via initializer)
_cache = None


def _init_worker():
    global _cache
    from disambiguation.signals.train_full import CachedData
    _cache = CachedData()


def _process_doc(args):
    doc_idx, window_tokens = args
    from disambiguation.signals.train_full import generate_doc_episodes
    feats, labels = generate_doc_episodes(_cache, doc_idx, window_tokens)
    if feats:
        return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int32)
    return None


def generate_dataset_episodes(
    dataset_name: str,
    doc_indices: list[int],
    window_tokens: int,
    num_workers: int = None,
) -> tuple[int, int]:
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    output_path = EPISODES_DIR / f"{dataset_name}_w{window_tokens}.npz"
    if output_path.exists():
        data = np.load(output_path)
        print(f"  {dataset_name} w={window_tokens}: already exists ({data['X'].shape[0]} episodes)")
        return data['X'].shape[0], int(data['y'].sum())

    print(f"  {dataset_name} w={window_tokens}: {len(doc_indices)} docs, {num_workers} workers...")
    start = time.time()

    # Process sequentially but in large batches to show progress
    # (multiprocessing with large shared numpy arrays is problematic)
    from disambiguation.signals.train_full import CachedData, generate_doc_episodes
    global _cache
    if _cache is None:
        _cache = CachedData()

    all_features = []
    all_labels = []
    batch_size = 100

    for i in range(0, len(doc_indices), batch_size):
        batch = doc_indices[i:i + batch_size]
        for doc_idx in batch:
            feats, labels = generate_doc_episodes(_cache, doc_idx, window_tokens)
            all_features.extend(feats)
            all_labels.extend(labels)

        processed = min(i + batch_size, len(doc_indices))
        elapsed = time.time() - start
        if processed > 0 and elapsed > 0:
            rate = processed / elapsed
            remaining = (len(doc_indices) - processed) / rate
            print(f"    {processed}/{len(doc_indices)} docs, {len(all_features)} eps, {rate:.1f} docs/s, ~{remaining:.0f}s left")

    if not all_features:
        return 0, 0

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y)

    elapsed = time.time() - start
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Done: {X.shape[0]} episodes, pos_rate={y.mean():.4f}, {elapsed:.0f}s, {size_mb:.0f} MB")
    return X.shape[0], int(y.sum())


def generate_all() -> None:
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    with open(CACHE_DIR / "dataset_ranges.json") as f:
        ranges = json.load(f)

    window_lengths = [100, 150, 200]

    for window in window_lengths:
        print(f"\n{'='*60}")
        print(f"WINDOW = {window} tokens")
        print(f"{'='*60}")

        # Individual datasets
        for name, r in ranges.items():
            doc_indices = list(range(r["start_doc"], r["end_doc"]))
            generate_dataset_episodes(name, doc_indices, window)

        # Combined
        all_indices = []
        for r in ranges.values():
            all_indices.extend(range(r["start_doc"], r["end_doc"]))
        generate_dataset_episodes("combined", all_indices, window)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_size = sum(f.stat().st_size for f in EPISODES_DIR.iterdir()) / (1024**2)
    print(f"Total episodes on disk: {total_size:.0f} MB")
    for f in sorted(EPISODES_DIR.iterdir()):
        data = np.load(f)
        print(f"  {f.name:30s} {data['X'].shape[0]:>10,} episodes, pos_rate={data['y'].mean():.4f}")


if __name__ == "__main__":
    generate_all()
