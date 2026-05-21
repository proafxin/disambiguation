import json
import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from transformers import DebertaV2TokenizerFast

from disambiguation.parsing.loader import load_parser
from disambiguation.signals.episodes import Episode, collect_training_data
from disambiguation.signals.evaluation import (
    episode_to_feature_vector,
    print_feature_importance,
    save_model,
    train_evaluation_function,
)
from disambiguation.signals.extraction import parse_document
from disambiguation.signals.state import (
    build_initial_state,
    extract_state_features,
    is_terminal,
    make_move,
)


def evaluate_search(model, num_docs: int = 5, start_doc: int = 10) -> None:
    parser = load_parser(device="cuda", dtype=torch.float16)
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    config_path = hf_hub_download(
        repo_id="ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt", filename="config.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    ds = load_from_disk("data/preco")

    correct_resolutions = 0
    wrong_resolutions = 0
    no_resolution = 0

    for doc_idx in range(start_doc, start_doc + num_docs):
        sample = ds["train"][doc_idx]
        parsed_doc = parse_document(sample["sentences"], parser, tokenizer, config, device="cuda")
        sentence_lengths = [len(s.tokens) for s in parsed_doc.sentences]

        for cluster in sample["mention_clusters"]:
            if len(cluster) < 2:
                continue

            # Find canonical entity (first PROPN in cluster)
            canonical_pos = None
            for mention in cluster:
                sent_idx, start, end = mention
                if sent_idx >= len(parsed_doc.sentences):
                    continue
                if end > len(parsed_doc.sentences[sent_idx].tokens):
                    continue
                tok = parsed_doc.sentences[sent_idx].tokens[start]
                if tok.pos == "PROPN":
                    from disambiguation.signals.state import get_absolute_token_position
                    canonical_pos = get_absolute_token_position(sent_idx, start, sentence_lengths)
                    break

            if canonical_pos is None:
                continue

            # Try to resolve non-PROPN mentions
            for mention in cluster:
                sent_idx, start, end = mention
                if sent_idx >= len(parsed_doc.sentences):
                    continue
                if end > len(parsed_doc.sentences[sent_idx].tokens):
                    continue
                tok = parsed_doc.sentences[sent_idx].tokens[start]
                if tok.pos == "PROPN":
                    continue

                # Run search using evaluation function
                state = build_initial_state(parsed_doc, sent_idx, start, end, window_tokens=200)
                resolved = False

                for hop in range(10):
                    if is_terminal(state):
                        if state.current_token_position == canonical_pos:
                            correct_resolutions += 1
                        else:
                            wrong_resolutions += 1
                        resolved = True
                        break

                    if not state.candidates:
                        break

                    # Score each candidate using the evaluation function
                    best_score = float("-inf")
                    best_idx = 0
                    state_feats = extract_state_features(state)

                    for i, cand in enumerate(state.candidates):
                        from disambiguation.signals.episodes import Episode
                        ep = Episode(
                            state_features=state_feats,
                            candidate_features={
                                "cand_pos": cand.mention.pos,
                                "cand_dep_rel": cand.mention.dep_rel,
                                "cand_dep_tree": cand.mention.dep_tree_tokens,
                                "cand_forward_ctx": cand.mention.forward_context,
                                "cand_backward_ctx": cand.mention.backward_context,
                                "cand_mention_text": cand.mention.mention_text,
                                "cand_token_position": cand.token_position,
                                "cand_sentence_idx": cand.sentence_idx,
                                "distance_from_current": abs(cand.token_position - state.current_token_position),
                                "same_dep_rel_as_current": cand.mention.dep_rel == state.current.dep_rel,
                                "same_dep_rel_as_origin": cand.mention.dep_rel == state.origin.dep_rel,
                                "same_pos_as_current": cand.mention.pos == state.current.pos,
                                "dep_tree_overlap_with_current": len(
                                    set(cand.mention.dep_tree_tokens) & set(state.current.dep_tree_tokens)
                                ),
                                "dep_tree_overlap_with_origin": len(
                                    set(cand.mention.dep_tree_tokens) & set(state.origin.dep_tree_tokens)
                                ),
                                "context_overlap_with_current": len(
                                    set(cand.mention.forward_context + cand.mention.backward_context)
                                    & set(state.current.forward_context + state.current.backward_context)
                                ),
                                "context_overlap_with_origin": len(
                                    set(cand.mention.forward_context + cand.mention.backward_context)
                                    & set(state.origin.forward_context + state.origin.backward_context)
                                ),
                            },
                            reward=0,
                            hop_number=hop,
                        )
                        vec = episode_to_feature_vector(ep).reshape(1, -1)
                        score = model.predict_proba(vec)[0, 1]
                        if score > best_score:
                            best_score = score
                            best_idx = i

                    state = make_move(state, best_idx, parsed_doc, window_tokens=200)

                if not resolved:
                    no_resolution += 1

    total = correct_resolutions + wrong_resolutions + no_resolution
    print(f"\n=== Search Evaluation (docs {start_doc}-{start_doc + num_docs}) ===")
    print(f"Correct: {correct_resolutions} ({correct_resolutions / max(total, 1) * 100:.1f}%)")
    print(f"Wrong: {wrong_resolutions} ({wrong_resolutions / max(total, 1) * 100:.1f}%)")
    print(f"No resolution: {no_resolution} ({no_resolution / max(total, 1) * 100:.1f}%)")
    print(f"Total attempts: {total}")


def main() -> None:
    # Collect episodes
    print("=== Collecting training episodes ===")
    episodes = collect_training_data(num_docs=10, window_tokens=200)
    print(f"Total episodes: {len(episodes)}")

    # Convert to feature vectors
    print("\n=== Training evaluation function ===")
    X = np.array([episode_to_feature_vector(e) for e in episodes])
    y = np.array([1 if e.reward > 0 else 0 for e in episodes], dtype=np.int32)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Positive rate: {y_train.mean():.4f}")

    # Train
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    # Ranking quality: for each state, does the correct candidate rank highest?
    from sklearn.metrics import average_precision_score
    y_prob = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, y_prob)
    print(f"Average precision: {ap:.4f}")

    # Feature importance — what did the system discover?
    print()
    print_feature_importance(model)

    # Save
    save_model(model)
    print("\nModel saved.")

    # Run search evaluation on unseen documents
    print("\n=== Running search on unseen documents ===")
    evaluate_search(model, num_docs=3, start_doc=10)


if __name__ == "__main__":
    main()
