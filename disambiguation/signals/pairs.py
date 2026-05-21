import random
from dataclasses import dataclass

from disambiguation.signals.extraction import MentionSignals, ParsedDocument, extract_mention_signals


@dataclass
class MentionPair:
    anchor: MentionSignals
    candidate: MentionSignals
    label: int  # 1 = same cluster, 0 = different cluster
    sentence_distance: int


def extract_all_mentions(
    parsed_doc: ParsedDocument,
    clusters: list[list[list[int]]],
    context_window: int = 5,
) -> list[MentionSignals]:
    mentions = []
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            sent_idx, start, end = mention
            if sent_idx >= len(parsed_doc.sentences):
                continue
            if end > len(parsed_doc.sentences[sent_idx].tokens):
                continue
            signals = extract_mention_signals(
                parsed_doc, sent_idx, start, end,
                context_window=context_window,
                cluster_id=cluster_id,
            )
            mentions.append(signals)
    return mentions


def build_positive_pairs(mentions: list[MentionSignals]) -> list[MentionPair]:
    # Group mentions by cluster
    clusters: dict[int, list[MentionSignals]] = {}
    for m in mentions:
        if m.cluster_id not in clusters:
            clusters[m.cluster_id] = []
        clusters[m.cluster_id].append(m)

    pairs = []
    for cluster_mentions in clusters.values():
        if len(cluster_mentions) < 2:
            continue
        # Sort by document order
        sorted_mentions = sorted(cluster_mentions, key=lambda m: (m.sent_idx, m.start_token))
        # Create adjacent pairs (the actual hops)
        for i in range(len(sorted_mentions) - 1):
            anchor = sorted_mentions[i]
            candidate = sorted_mentions[i + 1]
            pairs.append(MentionPair(
                anchor=anchor,
                candidate=candidate,
                label=1,
                sentence_distance=abs(candidate.sent_idx - anchor.sent_idx),
            ))
    return pairs


def build_hard_negative_pairs(
    mentions: list[MentionSignals],
    max_negatives_per_mention: int = 3,
    max_sentence_distance: int = 5,
) -> list[MentionPair]:
    # Hard negatives: mentions that are close but belong to different clusters
    pairs = []
    for i, anchor in enumerate(mentions):
        negatives_found = 0
        candidates = []
        for j, candidate in enumerate(mentions):
            if i == j:
                continue
            if candidate.cluster_id == anchor.cluster_id:
                continue
            dist = abs(candidate.sent_idx - anchor.sent_idx)
            if dist <= max_sentence_distance:
                candidates.append((dist, candidate))

        # Sort by distance (closest first = hardest negatives)
        candidates.sort(key=lambda x: x[0])
        for dist, candidate in candidates[:max_negatives_per_mention]:
            pairs.append(MentionPair(
                anchor=anchor,
                candidate=candidate,
                label=0,
                sentence_distance=dist,
            ))
            negatives_found += 1

    return pairs


def build_training_pairs(
    parsed_doc: ParsedDocument,
    clusters: list[list[list[int]]],
    context_window: int = 5,
    max_negatives_per_mention: int = 3,
    max_neg_sentence_distance: int = 5,
) -> list[MentionPair]:
    mentions = extract_all_mentions(parsed_doc, clusters, context_window)
    positives = build_positive_pairs(mentions)
    negatives = build_hard_negative_pairs(mentions, max_negatives_per_mention, max_neg_sentence_distance)
    all_pairs = positives + negatives
    random.shuffle(all_pairs)
    return all_pairs
