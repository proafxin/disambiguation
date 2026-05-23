import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np

from disambiguation.signals.train_full import CachedData, generate_doc_episodes

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
EPISODES_DIR = CACHE_DIR / "episodes"

GULLIVERS_DOC_IDX = 36676
HELD_OUT_DOC_INDICES = {36676}

CHUNK_SIZE = 5_000_000


def _chunk_dir(dataset_name: str, window_tokens: int) -> Path:
    return EPISODES_DIR / f"{dataset_name}_w{window_tokens}"


def _manifest_path(dataset_name: str, window_tokens: int) -> Path:
    return _chunk_dir(dataset_name, window_tokens) / "manifest.json"


def generate_dataset_episodes(
    cache: CachedData,
    dataset_name: str,
    doc_indices: list[int],
    window_tokens: int,
    exclude_doc_indices: set | None = None,
) -> tuple[int, int]:
    manifest = _manifest_path(dataset_name, window_tokens)
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        print(f"  {dataset_name} w={window_tokens}: already exists "
              f"({m['total_episodes']:,} eps, {m['num_chunks']} chunks, pos_rate={m['pos_rate']:.4f})")
        return m["total_episodes"], m["total_positive"]

    if exclude_doc_indices:
        doc_indices = [d for d in doc_indices if d not in exclude_doc_indices]

    out_dir = _chunk_dir(dataset_name, window_tokens)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {dataset_name} w={window_tokens}: {len(doc_indices)} docs...")
    start = time.time()

    doc_map: dict[int, tuple[int, int, int]] = {}

    chunk_idx = 0
    chunk_X: list[np.ndarray] = []
    chunk_y: list[int] = []
    chunk_ranks: list[int] = []
    total_eps = 0
    total_pos = 0

    def _flush_chunk() -> None:
        nonlocal chunk_idx, chunk_X, chunk_y, chunk_ranks
        if not chunk_X:
            return
        np.save(out_dir / f"chunk_{chunk_idx:04d}_X.npy", np.array(chunk_X, dtype=np.float32))
        np.save(out_dir / f"chunk_{chunk_idx:04d}_y.npy", np.array(chunk_y, dtype=np.int32))
        np.save(out_dir / f"chunk_{chunk_idx:04d}_ranks.npy", np.array(chunk_ranks, dtype=np.int32))
        chunk_idx += 1
        chunk_X = []
        chunk_y = []
        chunk_ranks = []

    for i, doc_idx in enumerate(doc_indices):
        feats, labels, ranks = generate_doc_episodes(cache, doc_idx, window_tokens)
        if not feats:
            continue

        n = len(feats)
        if chunk_X and len(chunk_X) + n > CHUNK_SIZE:
            _flush_chunk()

        row_start = len(chunk_X)
        chunk_X.extend(feats)
        chunk_y.extend(labels)
        chunk_ranks.extend(ranks)
        doc_map[doc_idx] = (chunk_idx, row_start, row_start + n)
        total_eps += n
        total_pos += sum(labels)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / max(elapsed, 1e-6)
            remaining = (len(doc_indices) - i - 1) / max(rate, 1e-6)
            print(f"    {i+1}/{len(doc_indices)} docs, {total_eps:,} eps, "
                  f"{rate:.1f} docs/s, ~{remaining:.0f}s left")

    _flush_chunk()

    manifest_data = {
        "dataset": dataset_name,
        "window_tokens": window_tokens,
        "total_episodes": total_eps,
        "total_positive": total_pos,
        "pos_rate": total_pos / max(total_eps, 1),
        "num_chunks": chunk_idx,
        "chunk_size": CHUNK_SIZE,
        "doc_map": {str(k): list(v) for k, v in doc_map.items()},
    }
    with open(manifest, "w") as f:
        json.dump(manifest_data, f)

    elapsed = time.time() - start
    size_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / (1024 * 1024)
    print(f"    Saved: {total_eps:,} eps in {chunk_idx} chunks, "
          f"pos_rate={total_pos/max(total_eps,1):.4f}, {elapsed:.0f}s, {size_mb:.0f} MB")
    return total_eps, total_pos


def _generate_window(window: int) -> None:
    with open(CACHE_DIR / "dataset_ranges.json") as f:
        ranges = json.load(f)
    cache = CachedData()
    print(f"\n{'='*60}\nWINDOW = {window} tokens\n{'='*60}")

    # Base episodes (all docs, Gulliver's held out)
    for name, r in ranges.items():
        doc_indices = list(range(r["start_doc"], r["end_doc"]))
        generate_dataset_episodes(cache, name, doc_indices, window,
                                  exclude_doc_indices=HELD_OUT_DOC_INDICES)

    # Gulliver's held-out (always separate)
    generate_dataset_episodes(cache, "gullivers", [GULLIVERS_DOC_IDX], window)


def generate_all() -> None:
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    window_lengths = [100, 150, 200]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_generate_window, w): w for w in window_lengths}
        for fut in concurrent.futures.as_completed(futures):
            w = futures[fut]
            try:
                fut.result()
                print(f"  window={w} complete")
            except Exception as exc:
                print(f"  window={w} FAILED: {exc}")
                raise

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for d in sorted(EPISODES_DIR.iterdir()):
        if d.is_dir():
            mp = d / "manifest.json"
            if mp.exists():
                with open(mp) as f:
                    m = json.load(f)
                size_mb = sum(f.stat().st_size for f in d.iterdir()) / (1024 * 1024)
                print(f"  {d.name:45s} {m['total_episodes']:>12,} eps  "
                      f"pos_rate={m['pos_rate']:.4f}  {size_mb:.0f} MB")


if __name__ == "__main__":
    generate_all()
