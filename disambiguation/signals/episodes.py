import json
import random
from dataclasses import dataclass

import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from transformers import DebertaV2TokenizerFast

from disambiguation.parsing.loader import load_parser
from disambiguation.signals.extraction import ParsedDocument, parse_document
from disambiguation.signals.state import (
    Candidate,
    State,
    build_initial_state,
    extract_state_features,
    get_absolute_token_position,
    is_terminal,
    make_move,
)


MAX_HOPS = 10


@dataclass
class Episode:
    state_features: dict
    candidate_features: dict
    reward: float
    hop_number: int


def _get_cluster_for_mention(
    sent_idx: int,
    start_token: int,
    end_token: int,
    clusters: list[list[list[int]]],
) -> int:
    for ci, cluster in enumerate(clusters):
        for mention in cluster:
            if mention[0] == sent_idx and mention[1] == start_token and mention[2] == end_token:
                return ci
    return -1


def _get_canonical_positions(
    cluster_id: int,
    clusters: list[list[list[int]]],
    parsed_doc: ParsedDocument,
) -> set[int]:
    if cluster_id < 0:
        return set()
    sentence_lengths = [len(s.tokens) for s in parsed_doc.sentences]
    positions = set()
    for mention in clusters[cluster_id]:
        sent_idx, start, end = mention
        if sent_idx < len(parsed_doc.sentences):
            pos = get_absolute_token_position(sent_idx, start, sentence_lengths)
            positions.add(pos)
    return positions


def _is_correct_move(candidate: Candidate, correct_positions: set[int]) -> bool:
    return candidate.token_position in correct_positions


def extract_candidate_features(state: State, candidate: Candidate, embedder=None) -> dict:
    current = state.current
    cand = candidate.mention

    features = {
        "cand_pos": cand.pos,
        "cand_dep_rel": cand.dep_rel,
        "cand_dep_tree": cand.dep_tree_tokens,
        "cand_forward_ctx": cand.forward_context,
        "cand_backward_ctx": cand.backward_context,
        "cand_mention_text": cand.mention_text,
        "cand_token_position": candidate.token_position,
        "cand_sentence_idx": candidate.sentence_idx,
        "distance_from_current": abs(candidate.token_position - state.current_token_position),
        "same_dep_rel_as_current": cand.dep_rel == current.dep_rel,
        "same_dep_rel_as_origin": cand.dep_rel == state.origin.dep_rel,
        "same_pos_as_current": cand.pos == current.pos,
        "dep_tree_overlap_with_current": len(set(cand.dep_tree_tokens) & set(current.dep_tree_tokens)),
        "dep_tree_overlap_with_origin": len(set(cand.dep_tree_tokens) & set(state.origin.dep_tree_tokens)),
        "context_overlap_with_current": len(
            set(cand.forward_context + cand.backward_context)
            & set(current.forward_context + current.backward_context)
        ),
        "context_overlap_with_origin": len(
            set(cand.forward_context + cand.backward_context)
            & set(state.origin.forward_context + state.origin.backward_context)
        ),
        "embed_sim_mention_to_origin": 0.0,
        "embed_sim_context_to_origin": 0.0,
        "embed_sim_mention_to_current": 0.0,
        "embed_sim_context_to_current": 0.0,
    }

    if embedder is not None:
        origin_mention = state.origin.mention_text
        current_mention = current.mention_text
        cand_mention = cand.mention_text
        origin_ctx = " ".join(state.origin.backward_context + state.origin.forward_context)
        current_ctx = " ".join(current.backward_context + current.forward_context)
        cand_ctx = " ".join(cand.backward_context + cand.forward_context)

        embs = embedder.encode(
            [origin_mention, current_mention, cand_mention, origin_ctx, current_ctx, cand_ctx],
            convert_to_numpy=True,
        )
        from sklearn.metrics.pairwise import cosine_similarity
        features["embed_sim_mention_to_origin"] = float(cosine_similarity([embs[2]], [embs[0]])[0, 0])
        features["embed_sim_context_to_origin"] = float(cosine_similarity([embs[5]], [embs[3]])[0, 0])
        features["embed_sim_mention_to_current"] = float(cosine_similarity([embs[2]], [embs[1]])[0, 0])
        features["embed_sim_context_to_current"] = float(cosine_similarity([embs[5]], [embs[4]])[0, 0])

    return features


def run_episode(
    parsed_doc: ParsedDocument,
    sent_idx: int,
    start_token: int,
    end_token: int,
    clusters: list[list[list[int]]],
    window_tokens: int = 200,
    embedder=None,
) -> list[Episode]:
    cluster_id = _get_cluster_for_mention(sent_idx, start_token, end_token, clusters)
    if cluster_id < 0:
        return []

    correct_positions = _get_canonical_positions(cluster_id, clusters, parsed_doc)
    state = build_initial_state(parsed_doc, sent_idx, start_token, end_token, window_tokens)

    # If already terminal (starting at a named entity), no episode needed
    if is_terminal(state) and state.current_token_position in correct_positions:
        return []

    episodes = []
    for hop in range(MAX_HOPS):
        if not state.candidates:
            break

        state_feats = extract_state_features(state)

        # Evaluate each candidate
        has_correct = False
        for i, candidate in enumerate(state.candidates):
            cand_feats = extract_candidate_features(state, candidate, embedder=embedder)
            correct = _is_correct_move(candidate, correct_positions)

            if correct:
                reward = 1.0
                has_correct = True
            else:
                reward = -1.0

            episodes.append(Episode(
                state_features=state_feats,
                candidate_features=cand_feats,
                reward=reward,
                hop_number=hop,
            ))

        # Make the correct move (teacher forcing during training)
        correct_candidates = [
            i for i, c in enumerate(state.candidates)
            if _is_correct_move(c, correct_positions)
        ]

        if not correct_candidates:
            break

        # Pick the nearest correct candidate
        best_idx = min(
            correct_candidates,
            key=lambda i: abs(state.candidates[i].token_position - state.current_token_position),
        )
        state = make_move(state, best_idx, parsed_doc, window_tokens)

        if is_terminal(state):
            break

    return episodes


def collect_training_data(
    num_docs: int = 20,
    window_tokens: int = 200,
    use_embeddings: bool = True,
) -> list[Episode]:
    parser = load_parser(device="cuda", dtype=torch.float16)
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    config_path = hf_hub_download(
        repo_id="ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt", filename="config.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    embedder = None
    if use_embeddings:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    ds = load_from_disk("data/preco")
    all_episodes = []

    for doc_idx in range(num_docs):
        sample = ds["train"][doc_idx]
        parsed_doc = parse_document(sample["sentences"], parser, tokenizer, config, device="cuda")

        # For each multi-mention cluster, run episodes from non-canonical mentions
        for cluster in sample["mention_clusters"]:
            if len(cluster) < 2:
                continue

            for mention in cluster:
                sent_idx, start, end = mention
                if sent_idx >= len(parsed_doc.sentences):
                    continue
                if end > len(parsed_doc.sentences[sent_idx].tokens):
                    continue

                # Skip if this mention is already a PROPN (likely canonical)
                tok = parsed_doc.sentences[sent_idx].tokens[start]
                if tok.pos == "PROPN":
                    continue

                episodes = run_episode(parsed_doc, sent_idx, start, end, sample["mention_clusters"], window_tokens, embedder=embedder)
                all_episodes.extend(episodes)

        if (doc_idx + 1) % 5 == 0:
            print(f"  Processed {doc_idx + 1}/{num_docs} docs, {len(all_episodes)} episodes so far")

    return all_episodes


def main() -> None:
    print("Collecting training episodes...")
    episodes = collect_training_data(num_docs=10, window_tokens=200)

    positive = sum(1 for e in episodes if e.reward > 0)
    negative = sum(1 for e in episodes if e.reward < 0)

    print(f"\nTotal episodes: {len(episodes)}")
    print(f"Positive (correct moves): {positive}")
    print(f"Negative (wrong moves): {negative}")
    print(f"Ratio: 1:{negative // max(positive, 1)}")

    # Show reward distribution
    rewards = [e.reward for e in episodes]
    print(f"\nReward range: [{min(rewards):.1f}, {max(rewards):.1f}]")

    # Show hop distribution for positive episodes
    hop_counts = [e.hop_number for e in episodes if e.reward > 0]
    if hop_counts:
        from collections import Counter
        hops = Counter(hop_counts)
        print("\nPositive episodes by hop:")
        for h in sorted(hops.keys()):
            print(f"  Hop {h}: {hops[h]}")

    # Show sample episode
    print("\nSample positive episode:")
    for e in episodes:
        if e.reward > 0:
            print(f"  State: origin_pos={e.state_features['origin_pos']}, current_pos={e.state_features['current_pos']}")
            print(f"  Candidate: pos={e.candidate_features['cand_pos']}, dep={e.candidate_features['cand_dep_rel']}")
            print(f"  same_dep_as_origin={e.candidate_features['same_dep_rel_as_origin']}")
            print(f"  dep_tree_overlap_origin={e.candidate_features['dep_tree_overlap_with_origin']}")
            print(f"  context_overlap_origin={e.candidate_features['context_overlap_with_origin']}")
            print(f"  distance={e.candidate_features['distance_from_current']}")
            print(f"  reward={e.reward}")
            break


if __name__ == "__main__":
    main()
