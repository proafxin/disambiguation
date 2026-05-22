import json
import time
from pathlib import Path

import numpy as np

from disambiguation.signals.abstract_features import NUM_FEATURES, POS_IDS
from disambiguation.signals.resolution_graph import ResolutionGraph
from disambiguation.signals.train_full import CachedData, build_features_batch

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
EPISODES_DIR = CACHE_DIR / "episodes"

# Gulliver's Travels doc index — always excluded from training
GULLIVERS_DOC_IDX = 36676
# A few other very long chains to keep in test only (top 5 longest in LitBank)
HELD_OUT_DOC_INDICES = {36676}  # can add more if needed


def generate_doc_episodes_filtered(
    cache: CachedData,
    doc_idx: int,
    window_tokens: int,
    max_chain_length: int | None = None,
    min_chain_length: int | None = None,
) -> tuple[list, list]:
    from disambiguation.signals.train_full import generate_doc_episodes

    if max_chain_length is None and min_chain_length is None:
        return generate_doc_episodes(cache, doc_idx, window_tokens)

    # Filter clusters by chain length before generating episodes
    start_gsi, end_gsi, source, orig_idx = cache.doc_boundaries[doc_idx]
    clusters = cache.clusters[doc_idx]

    # Temporarily replace clusters with filtered version
    filtered_clusters = []
    for cluster in clusters:
        chain_len = len(cluster)
        if max_chain_length is not None and chain_len > max_chain_length:
            continue
        if min_chain_length is not None and chain_len < min_chain_length:
            continue
        filtered_clusters.append(cluster)

    if not filtered_clusters:
        return [], []

    # Temporarily patch the cache clusters for this doc
    original_clusters = cache.clusters[doc_idx]
    cache.clusters[doc_idx] = filtered_clusters
    try:
        feats, labels = generate_doc_episodes(cache, doc_idx, window_tokens)
    finally:
        cache.clusters[doc_idx] = original_clusters

    return feats, labels


def generate_dataset_episodes(
    cache: CachedData,
    dataset_name: str,
    doc_indices: list[int],
    window_tokens: int,
    max_chain_length: int | None = None,
    min_chain_length: int | None = None,
    exclude_doc_indices: set | None = None,
) -> tuple[int, int]:
    suffix = ""
    if max_chain_length is not None:
        suffix += f"_maxhop{max_chain_length}"
    if min_chain_length is not None:
        suffix += f"_minhop{min_chain_length}"

    output_path = EPISODES_DIR / f"{dataset_name}_w{window_tokens}{suffix}.npz"
    if output_path.exists():
        data = np.load(output_path)
        n = data["X"].shape[0]
        pos = int(data["y"].sum())
        print(f"  {dataset_name}{suffix} w={window_tokens}: already exists ({n:,} eps, pos_rate={pos/max(n,1):.4f})")
        return n, pos

    # Filter out held-out docs
    if exclude_doc_indices:
        doc_indices = [d for d in doc_indices if d not in exclude_doc_indices]

    print(f"  {dataset_name}{suffix} w={window_tokens}: {len(doc_indices)} docs...")
    start = time.time()

    all_features = []
    all_labels = []
    doc_boundaries = []
    batch_size = 200

    for i in range(0, len(doc_indices), batch_size):
        batch = doc_indices[i:i + batch_size]
        for doc_idx in batch:
            start_ep = len(all_features)
            feats, labels = generate_doc_episodes_filtered(
                cache, doc_idx, window_tokens, max_chain_length, min_chain_length
            )
            all_features.extend(feats)
            all_labels.extend(labels)
            doc_boundaries.append((doc_idx, start_ep, len(all_features)))

        processed = min(i + batch_size, len(doc_indices))
        elapsed = time.time() - start
        rate = processed / max(elapsed, 1e-6)
        remaining = (len(doc_indices) - processed) / max(rate, 1e-6)
        print(f"    {processed}/{len(doc_indices)} docs, {len(all_features):,} eps, {rate:.1f} docs/s, ~{remaining:.0f}s left")

    if not all_features:
        return 0, 0

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    boundaries = np.array(doc_boundaries, dtype=np.int64)

    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, boundaries=boundaries)

    elapsed = time.time() - start
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {X.shape[0]:,} episodes, pos_rate={y.mean():.4f}, {elapsed:.0f}s, {size_mb:.0f} MB")
    return X.shape[0], int(y.sum())


def generate_all() -> None:
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    with open(CACHE_DIR / "dataset_ranges.json") as f:
        ranges = json.load(f)

    cache = CachedData()
    window_lengths = [100, 150, 200]

    for window in window_lengths:
        print(f"\n{'='*60}")
        print(f"WINDOW = {window} tokens")
        print(f"{'='*60}")

        # Standard episodes (all chains, exclude held-out docs)
        for name, r in ranges.items():
            doc_indices = list(range(r["start_doc"], r["end_doc"]))
            generate_dataset_episodes(cache, name, doc_indices, window,
                                      exclude_doc_indices=HELD_OUT_DOC_INDICES)

        # Table 5: low-hop training data (chains ≤10 hops)
        # Only for LitBank (has the long chains worth filtering)
        r = ranges["litbank"]
        litbank_docs = [d for d in range(r["start_doc"], r["end_doc"]) if d not in HELD_OUT_DOC_INDICES]
        generate_dataset_episodes(cache, "litbank", litbank_docs, window, max_chain_length=10)

        # Table 5: high-hop test data (chains >50 hops) — includes held-out docs
        all_litbank = list(range(r["start_doc"], r["end_doc"]))
        generate_dataset_episodes(cache, "litbank_hihop", all_litbank, window, min_chain_length=50)

        # Gulliver's held-out (all chains)
        generate_dataset_episodes(cache, "gullivers", [GULLIVERS_DOC_IDX], window)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_size = sum(f.stat().st_size for f in EPISODES_DIR.iterdir()) / (1024**2)
    print(f"Total on disk: {total_size:.0f} MB")
    for f in sorted(EPISODES_DIR.iterdir()):
        data = np.load(f)
        print(f"  {f.name:40s} {data['X'].shape[0]:>10,} episodes  pos_rate={data['y'].mean():.4f}")


if __name__ == "__main__":
    generate_all()
