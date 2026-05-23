import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np

from disambiguation.signals.train_full import CachedData, generate_doc_episodes

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
EPISODES_DIR = CACHE_DIR / "episodes"
BATCH_SIZE = 1000

GULLIVERS_DOC_IDX = 36676


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

    batch_X: list[np.ndarray] = []
    batch_y: list[int] = []
    batch_ranks: list[int] = []
    doc_map: dict[int, tuple[int, int]] = {}
    chunk_paths: list[tuple[Path, Path, Path]] = []
    total_eps = 0
    total_pos = 0

    def flush_batch() -> None:
        nonlocal total_eps
        if not batch_X:
            return
        idx = len(chunk_paths)
        cx = out_dir / f"_chunk_{idx}_X.npy"
        cy = out_dir / f"_chunk_{idx}_y.npy"
        cr = out_dir / f"_chunk_{idx}_ranks.npy"
        np.save(cx, np.array(batch_X, dtype=np.float32))
        np.save(cy, np.array(batch_y, dtype=np.int32))
        np.save(cr, np.array(batch_ranks, dtype=np.int32))
        chunk_paths.append((cx, cy, cr))
        total_eps += len(batch_X)
        batch_X.clear()
        batch_y.clear()
        batch_ranks.clear()

    for i, doc_idx in enumerate(doc_indices):
        feats, labels, ranks = generate_doc_episodes(cache, doc_idx, window_tokens)
        if not feats:
            continue
        rs = total_eps + len(batch_X)
        batch_X.extend(feats)
        batch_y.extend(labels)
        batch_ranks.extend(ranks)
        doc_map[doc_idx] = (rs, total_eps + len(batch_X))
        total_pos += sum(labels)

        if len(batch_X) >= BATCH_SIZE:
            flush_batch()

        if (i + 1) % 1000 == 0 or (i + 1) == len(doc_indices):
            elapsed = time.time() - start
            rate = (i + 1) / max(elapsed, 1e-6)
            remaining = (len(doc_indices) - i - 1) / max(rate, 1e-6)
            print(f"    [{dataset_name} w={window_tokens}] "
                  f"{i+1}/{len(doc_indices)} docs, {total_eps + len(batch_X):,} eps, "
                  f"{rate:.1f} docs/s, ~{remaining:.0f}s left")

    flush_batch()

    if len(chunk_paths) == 1:
        cx, cy, cr = chunk_paths[0]
        cx.rename(out_dir / "X.npy")
        cy.rename(out_dir / "y.npy")
        cr.rename(out_dir / "ranks.npy")
    else:
        np.save(out_dir / "X.npy", np.concatenate([np.load(cx) for cx, _, _ in chunk_paths]))
        np.save(out_dir / "y.npy", np.concatenate([np.load(cy) for _, cy, _ in chunk_paths]))
        np.save(out_dir / "ranks.npy", np.concatenate([np.load(cr) for _, _, cr in chunk_paths]))
        for cx, cy, cr in chunk_paths:
            cx.unlink()
            cy.unlink()
            cr.unlink()

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


def generate_all() -> None:
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    cache = CachedData()

    with open(CACHE_DIR / "dataset_ranges.json") as f:
        ranges = json.load(f)

    window_lengths = [100, 150, 200]

    for name, r in ranges.items():
        doc_indices = list(range(r["start_doc"], r["end_doc"]))
        print(f"\nDataset {name}: {len(doc_indices)} docs, {len(window_lengths)} windows in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(window_lengths)) as executor:
            futures = {
                executor.submit(generate_dataset_episodes, cache, name, doc_indices, w): w
                for w in window_lengths
            }
            for fut in concurrent.futures.as_completed(futures):
                w = futures[fut]
                try:
                    fut.result()
                    print(f"  {name} w={w} complete")
                except Exception as exc:
                    print(f"  {name} w={w} FAILED: {exc}")
                    raise

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
