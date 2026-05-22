from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResolutionGraph:
    # Maps (gsi, ti) -> cluster_id
    mention_to_cluster: dict = field(default_factory=dict)
    # Maps cluster_id -> canonical (gsi, ti) if known, else None
    cluster_to_canonical: dict = field(default_factory=dict)
    # Maps cluster_id -> set of (gsi, ti)
    cluster_members: dict = field(default_factory=dict)
    # Maps (gsi, ti) -> confidence of best link established
    mention_confidence: dict = field(default_factory=dict)
    _next_cluster_id: int = 0

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

    def link(self, mention_a: tuple, mention_b: tuple, confidence: float, is_b_propn: bool = False) -> None:
        # Ensure both are in graph
        if mention_a not in self.mention_to_cluster:
            self._new_cluster(mention_a)
        if mention_b not in self.mention_to_cluster:
            self._new_cluster(mention_b, is_b_propn)

        cid_a = self.mention_to_cluster[mention_a]
        cid_b = self.mention_to_cluster[mention_b]

        if cid_a == cid_b:
            return

        # Merge smaller cluster into larger
        members_a = self.cluster_members[cid_a]
        members_b = self.cluster_members[cid_b]
        canonical_a = self.cluster_to_canonical[cid_a]
        canonical_b = self.cluster_to_canonical[cid_b]

        # Keep the cluster with a known canonical, or the larger one
        if canonical_b is not None and canonical_a is None:
            keep, drop = cid_b, cid_a
        elif canonical_a is not None:
            keep, drop = cid_a, cid_b
        elif len(members_a) >= len(members_b):
            keep, drop = cid_a, cid_b
        else:
            keep, drop = cid_b, cid_a

        # Merge
        for m in self.cluster_members[drop]:
            self.mention_to_cluster[m] = keep
            self.cluster_members[keep].add(m)

        # Propagate canonical
        if self.cluster_to_canonical[drop] is not None and self.cluster_to_canonical[keep] is None:
            self.cluster_to_canonical[keep] = self.cluster_to_canonical[drop]

        del self.cluster_members[drop]
        del self.cluster_to_canonical[drop]

        # Update confidence
        self.mention_confidence[mention_a] = max(self.mention_confidence.get(mention_a, 0.0), confidence)
        self.mention_confidence[mention_b] = max(self.mention_confidence.get(mention_b, 0.0), confidence)

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
