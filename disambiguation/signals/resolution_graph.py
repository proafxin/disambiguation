from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ResolutionGraph:
    mention_to_cluster: dict = field(default_factory=dict)
    cluster_to_canonical: dict = field(default_factory=dict)
    cluster_members: dict = field(default_factory=dict)
    mention_confidence: dict = field(default_factory=dict)
    _next_cluster_id: int = 0
    _cache_valid: bool = False
    _cached_cluster_ids: Optional[np.ndarray] = None
    _cached_confidences: Optional[np.ndarray] = None

    def _new_cluster(self, mention: tuple, is_propn: bool = False) -> int:
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        self.mention_to_cluster[mention] = cid
        self.cluster_members[cid] = {mention}
        self.cluster_to_canonical[cid] = mention if is_propn else None
        return cid

    def add_mention(self, mention: tuple, is_propn: bool = False) -> None:
        if mention not in self.mention_to_cluster:
            self._new_cluster(mention, is_propn)
            self._cache_valid = False

    def link(self, mention_a: tuple, mention_b: tuple, confidence: float, is_b_propn: bool = False) -> None:
        if mention_a not in self.mention_to_cluster:
            self._new_cluster(mention_a)
        if mention_b not in self.mention_to_cluster:
            self._new_cluster(mention_b, is_b_propn)

        cid_a = self.mention_to_cluster[mention_a]
        cid_b = self.mention_to_cluster[mention_b]

        if cid_a == cid_b:
            return

        members_a = self.cluster_members[cid_a]
        members_b = self.cluster_members[cid_b]
        canonical_a = self.cluster_to_canonical[cid_a]
        canonical_b = self.cluster_to_canonical[cid_b]

        if canonical_b is not None and canonical_a is None:
            keep, drop = cid_b, cid_a
        elif canonical_a is not None:
            keep, drop = cid_a, cid_b
        elif len(members_a) >= len(members_b):
            keep, drop = cid_a, cid_b
        else:
            keep, drop = cid_b, cid_a

        for m in self.cluster_members[drop]:
            self.mention_to_cluster[m] = keep
            self.cluster_members[keep].add(m)

        if self.cluster_to_canonical[drop] is not None and self.cluster_to_canonical[keep] is None:
            self.cluster_to_canonical[keep] = self.cluster_to_canonical[drop]

        del self.cluster_members[drop]
        del self.cluster_to_canonical[drop]

        self.mention_confidence[mention_a] = max(self.mention_confidence.get(mention_a, 0.0), confidence)
        self.mention_confidence[mention_b] = max(self.mention_confidence.get(mention_b, 0.0), confidence)
        self._cache_valid = False

    def get_canonical(self, mention: tuple) -> tuple | None:
        cid = self.mention_to_cluster.get(mention)
        if cid is None:
            return None
        return self.cluster_to_canonical.get(cid)

    def is_resolved(self, mention: tuple) -> bool:
        return self.get_canonical(mention) is not None

    def get_cluster_id(self, mention: tuple) -> int:
        return self.mention_to_cluster.get(mention, -1)

    def same_cluster(self, mention_a: tuple, mention_b: tuple) -> bool:
        cid_a = self.mention_to_cluster.get(mention_a, -1)
        cid_b = self.mention_to_cluster.get(mention_b, -1)
        return cid_a >= 0 and cid_a == cid_b

    def get_confidence(self, mention: tuple) -> float:
        return self.mention_confidence.get(mention, 0.0)

    def build_abs_arrays(
        self,
        sent_offsets: np.ndarray,
        doc_start_abs: int,
        doc_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._cache_valid and self._cached_cluster_ids is not None:
            return self._cached_cluster_ids, self._cached_confidences
        cluster_ids = np.full(doc_len, -1, dtype=np.int32)
        confidences = np.zeros(doc_len, dtype=np.float32)
        for (gsi, ti), cid in self.mention_to_cluster.items():
            abs_pos = int(sent_offsets[gsi]) + ti - doc_start_abs
            if 0 <= abs_pos < doc_len:
                cluster_ids[abs_pos] = cid
                confidences[abs_pos] = self.mention_confidence.get((gsi, ti), 0.0)
        self._cached_cluster_ids = cluster_ids
        self._cached_confidences = confidences
        self._cache_valid = True
        return cluster_ids, confidences
