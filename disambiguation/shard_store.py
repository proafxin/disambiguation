import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np


class ShardStore:
    # Packs per-doc arrays into ~target_bytes shards on disk and serves them through a
    # bounded LRU, so RAM never holds more than `max_resident` shards at once. Designed
    # for block-shuffled iteration: process one shard's docs before moving to the next,
    # so the LRU rarely evicts mid-epoch. One store per (channel-set, window).
    def __init__(self, root: Path, max_resident: int = 6):
        self.root = Path(root)
        self.max_resident = max_resident
        self.doc_to_shard: dict[int, int] = {}
        self.shard_docs: dict[int, list[int]] = {}
        self._cache: "OrderedDict[int, dict]" = OrderedDict()
        man = self.root / "manifest.pkl"
        if man.exists():
            with man.open("rb") as f:
                self.doc_to_shard, self.shard_docs = pickle.load(f)

    def is_built(self, n_docs: int) -> bool:
        return len(self.doc_to_shard) == n_docs and (self.root / "manifest.pkl").exists()

    def write(self, arrays, target_bytes: float = 2e9) -> None:
        # arrays: iterable of (doc_idx, {name: np.ndarray}); contiguous doc order assumed.
        self.root.mkdir(parents=True, exist_ok=True)
        self.doc_to_shard, self.shard_docs = {}, {}
        shard_id, buf, buf_bytes = 0, {}, 0
        for doc_idx, d in arrays:
            buf[doc_idx] = {k: v for k, v in d.items()}
            buf_bytes += sum(v.nbytes for v in d.values())
            self.doc_to_shard[doc_idx] = shard_id
            self.shard_docs.setdefault(shard_id, []).append(doc_idx)
            if buf_bytes >= target_bytes:
                self._dump(shard_id, buf)
                shard_id, buf, buf_bytes = shard_id + 1, {}, 0
        if buf:
            self._dump(shard_id, buf)
        with (self.root / "manifest.pkl").open("wb") as f:
            pickle.dump((self.doc_to_shard, self.shard_docs), f)

    def _dump(self, shard_id: int, buf: dict) -> None:
        with (self.root / f"shard_{shard_id:04d}.pkl").open("wb") as f:
            pickle.dump(buf, f)

    def _load(self, shard_id: int) -> dict:
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]
        with (self.root / f"shard_{shard_id:04d}.pkl").open("rb") as f:
            shard = pickle.load(f)
        self._cache[shard_id] = shard
        self._cache.move_to_end(shard_id)
        while len(self._cache) > self.max_resident:
            self._cache.popitem(last=False)
        return shard

    def get(self, doc_idx: int) -> dict:
        return self._load(self.doc_to_shard[doc_idx])[doc_idx]

    def shard_order(self, rng=None) -> list[int]:
        ids = list(self.shard_docs.keys())
        if rng is not None:
            rng.shuffle(ids)
        return ids
