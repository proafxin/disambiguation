import concurrent.futures
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from disambiguation.signals.train_full import CachedData, generate_doc_episodes

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
EPISODES_DIR = CACHE_DIR / "episodes"

GULLIVERS_DOC_IDX = 36676

# Set in main process before forking; workers inherit via CoW — never written by workers.
_fork_cache: CachedData | None = None


def episode_dir(dataset_name: str, window_tokens: int) -> Path:
    return EPISODES_DIR / f"{dataset_name}_w{window_tokens}"


def manifest_path(dataset_name: str, window_tokens: int) -> Path:
    return episode_dir(dataset_name, window_tokens) / "manifest.json"


def generate_dataset_episodes(
    cache: CachedData,
    dataset_name: str,
    doc_indices: list[int],
    window_tokens: int,
) -> tuple[int, int]:
    mpath = manifest_path(dataset_name, window_tokens)
    if mpath.exists():
        with open(mpath) as f:
            m = json.load(f)
        print(f"  {dataset_name} w={window_tokens}: already exists "
              f"({m['total_episodes']:,} eps, pos_rate={m['pos_rate']:.4f})")
        return m["total_episodes"], m["total_positive"]

    out_dir = episode_dir(dataset_name, window_tokens)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {dataset_name} w={window_tokens}: {len(doc_indices)} docs...")
    start = time.time()

    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    all_ranks: list[int] = []
    doc_map: dict[int, tuple[int, int]] = {}
    total_pos = 0

    for i, doc_idx in enumerate(doc_indices):
        feats, labels, ranks = generate_doc_episodes(cache, doc_idx, window_tokens)
        if not feats:
            continue
        rs = len(all_X)
        all_X.extend(feats)
        all_y.extend(labels)
        all_ranks.extend(ranks)
        doc_map[doc_idx] = (rs, len(all_X))
        total_pos += sum(labels)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / max(elapsed, 1e-6)
            remaining = (len(doc_indices) - i - 1) / max(rate, 1e-6)
            print(f"    [{dataset_name} w={window_tokens}] "
                  f"{i+1}/{len(doc_indices)} docs, {len(all_X):,} eps, "
                  f"{rate:.1f} docs/s, ~{remaining:.0f}s left")

    total_eps = len(all_X)
    np.save(out_dir / "X.npy", np.array(all_X, dtype=np.float32))
    np.save(out_dir / "y.npy", np.array(all_y, dtype=np.int32))
    np.save(out_dir / "ranks.npy", np.array(all_ranks, dtype=np.int32))

    with open(mpath, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "window_tokens": window_tokens,
            "total_episodes": total_eps,
            "total_positive": total_pos,
            "pos_rate": total_pos / max(total_eps, 1),
            "doc_map": {str(k): list(v) for k, v in doc_map.items()},
        }, f)

    elapsed = time.time() - start
    size_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / (1024 * 1024)
    print(f"    [{dataset_name} w={window_tokens}] Done: {total_eps:,} eps, "
          f"pos_rate={total_pos/max(total_eps,1):.4f}, {elapsed:.0f}s, {size_mb:.0f} MB")
    return total_eps, total_pos


def _task_worker(args: tuple[str, list[int], int]) -> str:
    ds, doc_indices, window = args
    generate_dataset_episodes(_fork_cache, ds, doc_indices, window)
    return f"{ds} w={window}"


def generate_all() -> None:
    global _fork_cache
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    _fork_cache = CachedData()  # writes dataset_ranges.json as a side effect

    with open(CACHE_DIR / "dataset_ranges.json") as f:
        ranges = json.load(f)

    window_lengths = [100, 150, 200]
    tasks: list[tuple[str, list[int], int]] = [
        (name, list(range(r["start_doc"], r["end_doc"])), window)
        for window in window_lengths
        for name, r in ranges.items()
    ]

    n_workers = min(len(tasks), mp.cpu_count())
    print(f"Generating {len(tasks)} tasks with {n_workers} workers...")
    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_task_worker, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                print(f"  {fut.result()} complete")
            except Exception as exc:
                print(f"  {t[0]} w={t[2]} FAILED: {exc}")
                raise

    _fork_cache = None

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for d in sorted(EPISODES_DIR.iterdir()):
        if d.is_dir():
            mp_path = d / "manifest.json"
            if mp_path.exists():
                with open(mp_path) as f:
                    m = json.load(f)
                size_mb = sum(f.stat().st_size for f in d.iterdir()) / (1024 * 1024)
                print(f"  {d.name:40s} {m['total_episodes']:>12,} eps  "
                      f"pos_rate={m['pos_rate']:.4f}  {size_mb:.0f} MB")


if __name__ == "__main__":
    generate_all()
