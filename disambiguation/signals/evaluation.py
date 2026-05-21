import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from disambiguation.signals.episodes import Episode

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "eval_function.pkl"

# Feature encoding maps
POS_MAP = {"PROPN": 0, "NOUN": 1, "PRON": 2, "ADJ": 3, "VERB": 4, "DET": 5, "ADP": 6, "AUX": 7, "X": 8}
DEP_MAP = {
    "nsubj": 0, "obj": 1, "obl": 2, "nmod": 3, "nmod:poss": 4, "appos": 5,
    "conj": 6, "compound": 7, "flat": 8, "det": 9, "amod": 10, "root": 11,
    "nsubj:pass": 12, "obl:agent": 13, "iobj": 14, "ccomp": 15, "xcomp": 16,
    "acl": 17, "acl:relcl": 18, "advcl": 19, "expl": 20, "cop": 21,
}


def _encode_pos(pos: str) -> int:
    return POS_MAP.get(pos, len(POS_MAP))


def _encode_dep(dep: str) -> int:
    return DEP_MAP.get(dep, len(DEP_MAP))


def episode_to_feature_vector(episode: Episode) -> np.ndarray:
    sf = episode.state_features
    cf = episode.candidate_features

    features = [
        _encode_pos(sf["origin_pos"]),
        _encode_dep(sf["origin_dep_rel"]),
        _encode_pos(sf["current_pos"]),
        _encode_dep(sf["current_dep_rel"]),
        sf["hop_count"],
        sf["num_candidates"],
        sf["num_propn_visible"],
        sf["num_noun_visible"],
        sf["num_pron_visible"],
        sf["nearest_propn_distance"],
        _encode_pos(cf["cand_pos"]),
        _encode_dep(cf["cand_dep_rel"]),
        cf["distance_from_current"],
        int(cf["same_dep_rel_as_current"]),
        int(cf["same_dep_rel_as_origin"]),
        int(cf["same_pos_as_current"]),
        cf["dep_tree_overlap_with_current"],
        cf["dep_tree_overlap_with_origin"],
        cf["context_overlap_with_current"],
        cf["context_overlap_with_origin"],
        cf.get("embed_sim_mention_to_origin", 0.0),
        cf.get("embed_sim_context_to_origin", 0.0),
        cf.get("embed_sim_mention_to_current", 0.0),
        cf.get("embed_sim_context_to_current", 0.0),
    ]
    return np.array(features, dtype=np.float32)


def train_evaluation_function(episodes: list[Episode], n_estimators: int = 300, max_depth: int = 6) -> GradientBoostingClassifier:
    X = np.array([episode_to_feature_vector(e) for e in episodes])
    y = np.array([1 if e.reward > 0 else 0 for e in episodes], dtype=np.int32)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X, y)
    return model


def save_model(model: GradientBoostingClassifier) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def load_model() -> GradientBoostingClassifier:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def score_candidate(model: GradientBoostingClassifier, episode: Episode) -> float:
    X = episode_to_feature_vector(episode).reshape(1, -1)
    return model.predict_proba(X)[0, 1]


FEATURE_NAMES = [
    "origin_pos", "origin_dep", "current_pos", "current_dep",
    "hop_count", "num_candidates", "num_propn_visible", "num_noun_visible",
    "num_pron_visible", "nearest_propn_distance", "cand_pos", "cand_dep",
    "distance_from_current", "same_dep_as_current", "same_dep_as_origin",
    "same_pos_as_current", "dep_tree_overlap_current", "dep_tree_overlap_origin",
    "context_overlap_current", "context_overlap_origin",
    "embed_sim_mention_origin", "embed_sim_context_origin",
    "embed_sim_mention_current", "embed_sim_context_current",
]


def print_feature_importance(model: GradientBoostingClassifier) -> None:
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("Feature importance:")
    for i in sorted_idx:
        if importances[i] > 0.01:
            print(f"  {FEATURE_NAMES[i]:30s} {importances[i]:.4f}")
