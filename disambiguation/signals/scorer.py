from dataclasses import dataclass

import torch
from torch import Tensor, nn

from disambiguation.signals.extraction import MentionSignals


@dataclass
class PairFeatures:
    context_overlap: float
    dep_tree_overlap: float
    same_dep_rel: float
    same_pos: float
    sentence_distance: float
    mention_text_overlap: float
    backward_ctx_overlap: float
    forward_ctx_overlap: float


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def compute_pair_features(anchor: MentionSignals, candidate: MentionSignals) -> PairFeatures:
    # Context overlap (combined forward + backward)
    anchor_ctx = set(anchor.backward_context + anchor.forward_context)
    candidate_ctx = set(candidate.backward_context + candidate.forward_context)
    context_overlap = _jaccard(anchor_ctx, candidate_ctx)

    # Dep tree token overlap
    dep_tree_overlap = _jaccard(set(anchor.dep_tree_tokens), set(candidate.dep_tree_tokens))

    # Structural feature matches
    same_dep_rel = 1.0 if anchor.dep_rel == candidate.dep_rel else 0.0
    same_pos = 1.0 if anchor.pos == candidate.pos else 0.0

    # Sentence distance (normalized)
    sentence_distance = abs(anchor.sent_idx - candidate.sent_idx)

    # Mention text overlap (for named entity aliasing)
    anchor_words = set(anchor.mention_text.lower().split())
    candidate_words = set(candidate.mention_text.lower().split())
    mention_text_overlap = _jaccard(anchor_words, candidate_words)

    # Directional context overlaps
    backward_ctx_overlap = _jaccard(set(anchor.backward_context), set(candidate.backward_context))
    forward_ctx_overlap = _jaccard(set(anchor.forward_context), set(candidate.forward_context))

    return PairFeatures(
        context_overlap=context_overlap,
        dep_tree_overlap=dep_tree_overlap,
        same_dep_rel=same_dep_rel,
        same_pos=same_pos,
        sentence_distance=sentence_distance,
        mention_text_overlap=mention_text_overlap,
        backward_ctx_overlap=backward_ctx_overlap,
        forward_ctx_overlap=forward_ctx_overlap,
    )


def pair_features_to_tensor(features: PairFeatures) -> Tensor:
    return torch.tensor([
        features.context_overlap,
        features.dep_tree_overlap,
        features.same_dep_rel,
        features.same_pos,
        1.0 / (1.0 + features.sentence_distance),  # proximity score
        features.mention_text_overlap,
        features.backward_ctx_overlap,
        features.forward_ctx_overlap,
    ], dtype=torch.float32)


FEATURE_DIM = 8


class LinkScorer(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)
