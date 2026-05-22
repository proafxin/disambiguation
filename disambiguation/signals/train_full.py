import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

from disambiguation.signals.abstract_features import (
    POS_IDS,
    NUM_FEATURES,
    FEATURE_NAMES,
)

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
MAX_HOPS = 512
TOP_K = 20


class CachedData:
    def __init__(self) -> None:
        print("Loading cache...")
        start = time.time()

        ti = np.load(CACHE_DIR / "token_infos.npz")
        self.token_data = ti["data"]  # [total_tokens, 12] fp16
        self.sent_offsets = ti["offsets"]  # [total_sents + 1]

        se = np.load(CACHE_DIR / "sentence_embeddings.npz")
        self.sent_embs = se["data"]  # [total_sents, 384] fp16

        ne = np.load(CACHE_DIR / "noun_embeddings.npz")
        self.noun_embs = ne["data"]  # [total_nouns, 384] fp16

        with open(CACHE_DIR / "metadata.json") as f:
            meta = json.load(f)
        self.doc_boundaries = meta["doc_boundaries"]  # [(start_sent, end_sent, source, orig_idx)]
        self.noun_positions = meta["noun_positions"]  # [(global_sent_idx, token_idx)]

        with open(CACHE_DIR / "clusters.json") as f:
            self.clusters = json.load(f)  # per doc

        # Build noun lookup: (global_sent_idx, token_idx) -> noun_emb_idx
        self.noun_lookup = {}
        for i, (gsi, ti) in enumerate(self.noun_positions):
            self.noun_lookup[(gsi, ti)] = i

        elapsed = time.time() - start
        ram = (self.token_data.nbytes + self.sent_embs.nbytes + self.noun_embs.nbytes) / (1024**3)
        print(f"  Loaded in {elapsed:.1f}s, {ram:.2f} GB RAM")
        print(f"  {len(self.doc_boundaries)} docs, {len(self.sent_offsets)-1} sents, {len(self.noun_positions)} nouns")

    def get_token_info(self, global_sent_idx: int, token_idx: int) -> np.ndarray:
        start = int(self.sent_offsets[global_sent_idx])
        return self.token_data[start + token_idx]

    def get_sent_length(self, global_sent_idx: int) -> int:
        return int(self.sent_offsets[global_sent_idx + 1] - self.sent_offsets[global_sent_idx])

    def get_noun_emb(self, global_sent_idx: int, token_idx: int) -> np.ndarray | None:
        idx = self.noun_lookup.get((global_sent_idx, token_idx))
        if idx is None:
            return None
        return self.noun_embs[idx].astype(np.float32)


def build_features_from_cache(
    cache: CachedData,
    origin_gsi: int, origin_ti: int,
    current_gsi: int, current_ti: int,
    cand_gsi: int, cand_ti: int,
    hop_count: int,
    resolved_gender: int,
    resolved_number: int,
    chain_deps: list,
    chain_pos: list,
    chain_noun_embs: list,
    rank: int,
    num_cands: int,
    num_gender_match: int,
    num_propn_cands: int,
) -> np.ndarray:
    o = cache.get_token_info(origin_gsi, origin_ti).astype(np.float32)
    cur = cache.get_token_info(current_gsi, current_ti).astype(np.float32)
    c = cache.get_token_info(cand_gsi, cand_ti).astype(np.float32)

    # Positions
    o_abs = int(cache.sent_offsets[origin_gsi]) + origin_ti
    cur_abs = int(cache.sent_offsets[current_gsi]) + current_ti
    c_abs = int(cache.sent_offsets[cand_gsi]) + cand_ti

    # Agreement
    gender_match_o = int(o[2] == 3 or c[2] == 3 or o[2] == c[2])
    number_match_o = int(o[3] == 2 or c[3] == 2 or o[3] == c[3])
    gender_match_cur = int(cur[2] == 3 or c[2] == 3 or cur[2] == c[2])
    number_match_cur = int(cur[3] == 2 or c[3] == 2 or cur[3] == c[3])
    resolved_gender_match = int(resolved_gender == 3 or c[2] == 3 or resolved_gender == c[2])
    resolved_number_match = int(resolved_number == 2 or c[3] == 2 or resolved_number == c[3])

    # Chain consistency
    dep_consistent = sum(1 for d in chain_deps if d == c[1]) / max(len(chain_deps), 1)
    pos_consistent = sum(1 for p in chain_pos if p == c[0]) / max(len(chain_pos), 1)
    is_propn_terminal = int(c[0] == POS_IDS["PROPN"] and o[0] == POS_IDS["PRON"])

    # Noun embedding similarity to chain
    noun_sim = 0.0
    cand_noun_emb = cache.get_noun_emb(cand_gsi, cand_ti)
    if cand_noun_emb is not None and chain_noun_embs:
        cand_norm = np.linalg.norm(cand_noun_emb) + 1e-8
        for chain_emb in chain_noun_embs:
            s = float(np.dot(cand_noun_emb, chain_emb) / (cand_norm * (np.linalg.norm(chain_emb) + 1e-8)))
            if s > noun_sim:
                noun_sim = s

    return np.array([
        o[0], o[1], o[2], o[3], o[6], o[8],  # origin: pos, dep, gender, number, is_subj, is_poss
        cur[0], cur[1], cur[2], cur[3], cur[6], cur[8],  # current
        c[0], c[1], c[2], c[3], c[4], c[6], c[7], c[8],  # cand: pos,dep,gender,number,person,is_subj,is_obj,is_poss
        int(c[1] == cur[1]), int(c[1] == o[1]), int(c[0] == cur[0]), int(c[6] == o[6]),  # structural
        c[9], int(cand_gsi == current_gsi),  # depth_to_root, same_sentence
        abs(c_abs - cur_abs), abs(cand_gsi - current_gsi), hop_count,  # distance
        gender_match_o, number_match_o, gender_match_cur, number_match_cur,  # agreement
        resolved_gender_match, resolved_number_match,
        dep_consistent, pos_consistent, is_propn_terminal,  # chain
        rank, num_cands, num_gender_match, num_propn_cands,  # competition
        noun_sim,  # noun embedding
    ], dtype=np.float32)


def generate_doc_episodes(cache: CachedData, doc_idx: int, window_tokens: int = 150) -> tuple[list[np.ndarray], list[int]]:
    start_gsi, end_gsi, source, orig_idx = cache.doc_boundaries[doc_idx]
    clusters = cache.clusters[doc_idx]

    features_list = []
    labels_list = []

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        # Build correct positions set (global_sent_idx, token_idx)
        correct_positions = set()
        for mention in cluster:
            si, st, en = mention
            gsi = start_gsi + si
            if gsi < end_gsi:
                correct_positions.add((gsi, st))

        for mention in cluster:
            si, st, en = mention
            origin_gsi = start_gsi + si
            if origin_gsi >= end_gsi:
                continue
            if st >= cache.get_sent_length(origin_gsi):
                continue

            origin_info = cache.get_token_info(origin_gsi, st)
            if origin_info[0] == POS_IDS["PROPN"]:
                continue

            # Initialize chain
            current_gsi = origin_gsi
            current_ti = st
            visited = {(origin_gsi, st)}
            resolved_gender = int(origin_info[2]) if origin_info[2] != 3 else 3
            resolved_number = int(origin_info[3]) if origin_info[3] != 2 else 2
            chain_deps = [int(origin_info[1])]
            chain_pos = [int(origin_info[0])]
            chain_noun_embs = []

            origin_noun = cache.get_noun_emb(origin_gsi, st)
            if origin_noun is not None:
                chain_noun_embs.append(origin_noun)

            for hop in range(MAX_HOPS):
                cur_abs = int(cache.sent_offsets[current_gsi]) + current_ti

                # Gather candidates in window
                candidates = []
                for gsi in range(max(start_gsi, current_gsi - 20), min(end_gsi, current_gsi + 20)):
                    sent_len = cache.get_sent_length(gsi)
                    for ti in range(sent_len):
                        if (gsi, ti) in visited:
                            continue
                        info = cache.get_token_info(gsi, ti)
                        if info[0] not in (POS_IDS["NOUN"], POS_IDS["PROPN"], POS_IDS["PRON"]):
                            continue
                        abs_pos = int(cache.sent_offsets[gsi]) + ti
                        if abs(abs_pos - cur_abs) <= window_tokens:
                            candidates.append((gsi, ti, abs_pos))

                if not candidates:
                    break

                # Rank by sentence embedding similarity to origin (filter)
                origin_emb = cache.sent_embs[origin_gsi].astype(np.float32)
                cand_gsis = [c[0] for c in candidates]
                cand_embs = cache.sent_embs[cand_gsis].astype(np.float32)
                sims = cosine_similarity([origin_emb], cand_embs)[0]
                top_indices = np.argsort(-sims)[:TOP_K]

                # Competition features
                num_propn = sum(1 for gsi, ti, _ in candidates if cache.get_token_info(gsi, ti)[0] == POS_IDS["PROPN"])
                num_gm = sum(
                    1 for gsi, ti, _ in candidates
                    if resolved_gender == 3 or cache.get_token_info(gsi, ti)[2] == 3 or cache.get_token_info(gsi, ti)[2] == resolved_gender
                )

                # Build features for top-K
                for rank, idx in enumerate(top_indices):
                    cgsi, cti, _ = candidates[idx]
                    is_correct = (cgsi, cti) in correct_positions

                    feat = build_features_from_cache(
                        cache, origin_gsi, st, current_gsi, current_ti, cgsi, cti,
                        hop, resolved_gender, resolved_number,
                        chain_deps, chain_pos, chain_noun_embs,
                        rank, len(candidates), num_gm, num_propn,
                    )
                    features_list.append(feat)
                    labels_list.append(1 if is_correct else 0)

                # Teacher forcing
                correct_cands = [
                    (idx, candidates[idx][2])
                    for idx, (gsi, ti, _) in enumerate(candidates)
                    if (gsi, ti) in correct_positions
                ]
                if not correct_cands:
                    break

                best_idx = min(correct_cands, key=lambda x: abs(x[1] - cur_abs))[0]
                next_gsi, next_ti, _ = candidates[best_idx]

                visited.add((next_gsi, next_ti))
                current_gsi = next_gsi
                current_ti = next_ti

                next_info = cache.get_token_info(next_gsi, next_ti)
                chain_deps.append(int(next_info[1]))
                chain_pos.append(int(next_info[0]))

                next_noun = cache.get_noun_emb(next_gsi, next_ti)
                if next_noun is not None:
                    chain_noun_embs.append(next_noun)

                if resolved_gender == 3 and next_info[2] != 3:
                    resolved_gender = int(next_info[2])
                if resolved_number == 2 and next_info[3] != 2:
                    resolved_number = int(next_info[3])

                if next_info[0] == POS_IDS["PROPN"]:
                    break

    return features_list, labels_list


def train_full(num_train_docs: int = 1000, num_test_docs: int = 200, window_tokens: int = 150) -> None:
    cache = CachedData()

    # Shuffle doc indices
    rng = np.random.default_rng(42)
    all_indices = rng.permutation(len(cache.doc_boundaries))
    train_indices = all_indices[:num_train_docs]
    test_indices = all_indices[num_train_docs:num_train_docs + num_test_docs]

    # Generate train episodes
    print(f"\nGenerating train episodes ({num_train_docs} docs)...")
    start = time.time()
    train_features = []
    train_labels = []

    for i, doc_idx in enumerate(train_indices):
        feats, labels = generate_doc_episodes(cache, int(doc_idx), window_tokens)
        train_features.extend(feats)
        train_labels.extend(labels)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_train_docs} docs, {len(train_features)} episodes")

    X_train = np.array(train_features)
    y_train = np.array(train_labels, dtype=np.int32)
    train_time = time.time() - start
    print(f"  Train: {X_train.shape[0]} episodes, {train_time:.1f}s, pos_rate={y_train.mean():.4f}")

    # Generate test episodes
    print(f"\nGenerating test episodes ({num_test_docs} docs)...")
    test_features = []
    test_labels = []

    for doc_idx in test_indices:
        feats, labels = generate_doc_episodes(cache, int(doc_idx), window_tokens)
        test_features.extend(feats)
        test_labels.extend(labels)

    X_test = np.array(test_features)
    y_test = np.array(test_labels, dtype=np.int32)
    print(f"  Test: {X_test.shape[0]} episodes, pos_rate={y_test.mean():.4f}")

    # Train
    print("\nTraining XGBoost (GPU)...")
    start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.1,
        subsample=0.8, min_child_weight=10, device="cuda",
        tree_method="hist", random_state=42,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training: {train_time:.1f}s")

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    ap = average_precision_score(y_test, y_prob)
    recall = y_test[y_pred == 1].sum() / max(y_test.sum(), 1)
    precision = y_test[y_pred == 1].sum() / max(y_pred.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n=== Results ===")
    print(f"AP={ap:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    # By hop type
    pron_id = POS_IDS["PRON"]
    propn_id = POS_IDS["PROPN"]
    noun_id = POS_IDS["NOUN"]
    for cand_name, cand_id in [("PRON", pron_id), ("NOUN", noun_id), ("PROPN", propn_id)]:
        mask = (X_test[:, 0] == pron_id) & (X_test[:, 12] == cand_id)
        if mask.sum() < 10 or y_test[mask].sum() == 0:
            continue
        sub_y = y_test[mask]
        sub_pred = model.predict(X_test[mask])
        r = sub_y[sub_pred == 1].sum() / max(sub_y.sum(), 1)
        print(f"  PRON->{cand_name}: R={r:.4f} (pos={sub_y.sum()})")

    # Feature importance
    print("\nTop 15 features:")
    importances = model.feature_importances_
    for i in np.argsort(importances)[::-1][:15]:
        print(f"  {FEATURE_NAMES[i]:25s} {importances[i]:.4f}")

    # Save model
    model.save_model(str(CACHE_DIR / "xgb_model.json"))
    print(f"\nModel saved to {CACHE_DIR / 'xgb_model.json'}")


if __name__ == "__main__":
    train_full(num_train_docs=1000, num_test_docs=200, window_tokens=150)
