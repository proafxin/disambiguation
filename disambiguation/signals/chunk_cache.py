from collections import OrderedDict
from pathlib import Path

import numpy as np

MAX_CACHED_CHUNKS = 64


class ChunkCache:
    def __init__(self) -> None:
        self._cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def load(self, chunk_path_X: Path, chunk_path_y: Path) -> tuple[np.ndarray, np.ndarray]:
        key = (str(chunk_path_X), str(chunk_path_y))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        X = np.load(chunk_path_X)
        y = np.load(chunk_path_y)
        self._cache[key] = (X, y)
        self._cache.move_to_end(key)
        if len(self._cache) > MAX_CACHED_CHUNKS:
            self._cache.popitem(last=False)
        return X, y
