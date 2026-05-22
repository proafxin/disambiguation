import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

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
        self.token_data = ti["data"].astype(np.float32)  # store as fp32 to avoid repeated astype
        self.sent_offsets = ti["offsets"]

        se = np.load(CACHE_DIR / "sentence_embeddings.npz")
        self.sent_embs = se["data"].astype(np.float32)  # fp32 for fast dot products
        # Pre-normalize sentence embeddings for fast cosine similarity
        norms = np.linalg.norm(self.sent_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.sent_embs_normed = self.sent_embs / norms

        ne = np.load(CACHE_DIR / "noun_embeddings.npz")
        self.noun_embs = ne["data"].astype(np.float32)
        # Pre-normalize noun embeddings
        noun_norms = np.linalg.norm(self.noun_embs, axis=1, keepdims=True)
        noun_norms = np.where(noun_norms == 0, 1.0, noun_norms)
        self.noun_embs_normed = self.noun_embs / noun_norms

        with open(CACHE_DIR / "metadata.json") as f:
            meta = json.load(f)
        self.doc_boundaries = meta["doc_boundaries"]
        self.noun_positions = meta["noun_positions"]

        with open(CACHE_DIR / "clusters.json") as f:
            self.clusters = json.load(f)

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

    def get_noun_emb_idx(self, global_sent_idx: int, token_idx: int) -> int:
        return self.noun_lookup.get((global_sent_idx, token_idx), -1)

    def cosine_sim_batch(self, query_idx: int, cand_indices: np.ndarray) -> np.ndarray:
        # Fast cosine similarity using pre-normalized embeddings
        return self.sent_embs_normed[query_idx] @ self.sent_embs_normed[cand_indices].T

    def noun_sim_to_chain(self, cand_gsi: int, cand_ti: int, chain_noun_indices: list[int]) -> float:
        if not chain_noun_indices:
            return 0.0
        cand_idx = self.noun_lookup.get((cand_gsi, cand_ti), -1)
        if cand_idx < 0:
            return 0.0
        # Vectorized dot product against all chain noun embeddings
        chain_embs = self.noun_embs_normed[chain_noun_indices]  # [N, 384]
        sims = chain_embs @ self.noun_embs_normed[cand_idx]  # [N]
        return float(sims.max())


def build_features_batch(
    cache: CachedData,
    origin_gsi: int, origin_ti: int,
    current_gsi: int, current_ti: int,
    cand_gsis: np.ndarray,
    cand_tis: np.ndarray,
    hop_count: int,
    resolved_gender: int,
    resolved_number: int,
    chain_deps: np.ndarray,
    chain_pos: np.ndarray,
    chain_noun_indices: list[int],
    ranks: np.ndarray,
    num_cands: int,
    num_gender_match: int,
    num_propn_cands: int,
) -> np.ndarray:
    n = len(cand_gsis)
    o = cache.get_token_info(origin_gsi, origin_ti)
    cur = cache.get_token_info(current_gsi, current_ti)

    # Batch fetch all candidate token infos
    cur_abs = int(cache.sent_offsets[current_gsi]) + current_ti
    o_abs = int(cache.sent_offsets[origin_gsi]) + origin_ti

    # Vectorized candidate info lookup
    cand_starts = cache.sent_offsets[cand_gsis]
    cand_abs = cand_starts + cand_tis
    c_all = cache.token_data[cand_starts + cand_tis]  # [n, 15]

    # Agreement (vectorized)
    gender_match_o = ((o[2] == 3) | (c_all[:, 2] == 3) | (o[2] == c_all[:, 2])).astype(np.float32)
    number_match_o = ((o[3] == 2) | (c_all[:, 3] == 2) | (o[3] == c_all[:, 3])).astype(np.float32)
    gender_match_cur = ((cur[2] == 3) | (c_all[:, 2] == 3) | (cur[2] == c_all[:, 2])).astype(np.float32)
    number_match_cur = ((cur[3] == 2) | (c_all[:, 3] == 2) | (cur[3] == c_all[:, 3])).astype(np.float32)
    resolved_gender_match = ((resolved_gender == 3) | (c_all[:, 2] == 3) | (c_all[:, 2] == resolved_gender)).astype(np.float32)
    resolved_number_match = ((resolved_number == 2) | (c_all[:, 3] == 2) | (c_all[:, 3] == resolved_number)).astype(np.float32)

    # Chain consistency (vectorized)
    if len(chain_deps) > 0:
        dep_consistent = (c_all[:, 1:2] == chain_deps).mean(axis=1)
        pos_consistent = (c_all[:, 0:1] == chain_pos).mean(axis=1)
    else:
        dep_consistent = np.zeros(n, dtype=np.float32)
        pos_consistent = np.zeros(n, dtype=np.float32)

    is_propn_terminal = ((c_all[:, 0] == POS_IDS["PROPN"]) & (o[0] == POS_IDS["PRON"])).astype(np.float32)

    # Verb/head association (vectorized)
    both_heads_verb = ((o[13] == POS_IDS["VERB"]) & (c_all[:, 13] == POS_IDS["VERB"])).astype(np.float32)
    same_head_pos_origin = (o[13] == c_all[:, 13]).astype(np.float32)
    same_head_pos_current = (cur[13] == c_all[:, 13]).astype(np.float32)
    same_head_token = ((cand_gsis == current_gsi) & (c_all[:, 12] >= 0) & (cur[12] >= 0) & (c_all[:, 12] == cur[12])).astype(np.float32)
    cand_is_verb_arg = ((c_all[:, 13] == POS_IDS["VERB"]) & ((c_all[:, 6] == 1) | (c_all[:, 7] == 1))).astype(np.float32)
    origin_is_verb_arg = float(o[13] == POS_IDS["VERB"] and (o[6] == 1 or o[7] == 1))

    # Noun similarity (vectorized)
    noun_sims = np.zeros(n, dtype=np.float32)
    if chain_noun_indices:
        chain_embs = cache.noun_embs_normed[chain_noun_indices]  # [C, 384]
        for i in range(n):
            idx = cache.noun_lookup.get((int(cand_gsis[i]), int(cand_tis[i])), -1)
            if idx >= 0:
                noun_sims[i] = float((chain_embs @ cache.noun_embs_normed[idx]).max())

    # Stack all features: [n, 49]
    features = np.column_stack([
        np.full(n, o[0]), np.full(n, o[1]), np.full(n, o[2]), np.full(n, o[3]),
        np.full(n, o[6]), np.full(n, o[8]),
        np.full(n, cur[0]), np.full(n, cur[1]), np.full(n, cur[2]), np.full(n, cur[3]),
        np.full(n, cur[6]), np.full(n, cur[8]),
        c_all[:, 0], c_all[:, 1], c_all[:, 2], c_all[:, 3], c_all[:, 4], c_all[:, 6], c_all[:, 7], c_all[:, 8],
        (c_all[:, 1] == cur[1]).astype(np.float32),
        (c_all[:, 1] == o[1]).astype(np.float32),
        (c_all[:, 0] == cur[0]).astype(np.float32),
        (c_all[:, 6] == o[6]).astype(np.float32),
        c_all[:, 9],
        (cand_gsis == current_gsi).astype(np.float32),
        np.abs(cand_abs - cur_abs).astype(np.float32),
        np.abs(cand_gsis - current_gsi).astype(np.float32),
        np.full(n, hop_count, dtype=np.float32),
        gender_match_o, number_match_o, gender_match_cur, number_match_cur,
        resolved_gender_match, resolved_number_match,
        dep_consistent, pos_consistent, is_propn_terminal,
        both_heads_verb, same_head_pos_origin, same_head_pos_current,
        same_head_token, cand_is_verb_arg, np.full(n, origin_is_verb_arg),
        ranks.astype(np.float32),
        np.full(n, num_cands, dtype=np.float32),
        np.full(n, num_gender_match, dtype=np.float32),
        np.full(n, num_propn_cands, dtype=np.float32),
        noun_sims,
    ])
    return features.astype(np.float32)


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
            chain_deps = np.array([int(origin_info[1])], dtype=np.float32)
            chain_pos = np.array([int(origin_info[0])], dtype=np.float32)
            chain_noun_indices = []

            origin_noun_idx = cache.noun_lookup.get((origin_gsi, st), -1)
            if origin_noun_idx >= 0:
                chain_noun_indices.append(origin_noun_idx)

            for hop in range(MAX_HOPS):
                cur_abs = int(cache.sent_offsets[current_gsi]) + current_ti

                # Vectorized candidate gathering
                cur_abs = int(cache.sent_offsets[current_gsi]) + current_ti
                # Sentence range to search
                gsi_lo = max(start_gsi, current_gsi - 20)
                gsi_hi = min(end_gsi, current_gsi + 20)

                # Get all token absolute positions in range
                range_start = int(cache.sent_offsets[gsi_lo])
                range_end = int(cache.sent_offsets[gsi_hi])
                if range_end <= range_start:
                    break

                # Vectorized: get POS for all tokens in range
                tokens_in_range = cache.token_data[range_start:range_end, 0]  # POS column
                abs_positions = np.arange(range_start, range_end)

                # Filter: nominal POS
                nominal_mask = np.isin(tokens_in_range, [POS_IDS["NOUN"], POS_IDS["PROPN"], POS_IDS["PRON"]])
                # Filter: within window
                window_mask = np.abs(abs_positions - cur_abs) <= window_tokens
                # Filter: not visited
                valid_mask = nominal_mask & window_mask

                valid_abs = abs_positions[valid_mask]
                if len(valid_abs) == 0:
                    break

                # Convert abs positions back to (gsi, ti)
                # Use sent_offsets to find which sentence each token belongs to
                # sent_offsets[gsi] <= abs_pos < sent_offsets[gsi+1]
                gsi_range = np.arange(gsi_lo, gsi_hi)
                sent_starts = cache.sent_offsets[gsi_range]
                sent_ends = cache.sent_offsets[gsi_range + 1]

                # Vectorized abs_pos -> (gsi, ti) conversion
                gsi_indices = np.searchsorted(sent_ends, valid_abs, side='right')
                valid_filter = gsi_indices < len(gsi_range)
                valid_abs = valid_abs[valid_filter]
                gsi_indices = gsi_indices[valid_filter]

                gsis = gsi_range[gsi_indices]
                tis = valid_abs - cache.sent_offsets[gsis]

                candidates = [
                    (int(gsi), int(ti), int(ap))
                    for gsi, ti, ap in zip(gsis, tis, valid_abs)
                    if (int(gsi), int(ti)) not in visited
                ]

                if not candidates:
                    break

                # Rank by sentence embedding similarity to origin (filter) - vectorized
                cand_gsis_arr = np.array([c[0] for c in candidates])
                sims = cache.cosine_sim_batch(origin_gsi, cand_gsis_arr)
                top_indices = np.argsort(-sims)[:TOP_K]

                # Competition features (vectorized)
                all_gsis = np.array([c[0] for c in candidates])
                all_tis = np.array([c[1] for c in candidates])
                all_starts = cache.sent_offsets[all_gsis]
                all_pos = cache.token_data[all_starts + all_tis, 0]
                all_gender = cache.token_data[all_starts + all_tis, 2]

                num_propn = int(np.sum(all_pos == POS_IDS["PROPN"]))
                num_gm = int(np.sum(
                    (all_gender == 3) | (all_gender == resolved_gender) | (resolved_gender == 3)
                ))

                # Build features for ALL top-K candidates in one vectorized batch
                top_cand_gsis = np.array([candidates[idx][0] for idx in top_indices])
                top_cand_tis = np.array([candidates[idx][1] for idx in top_indices])
                top_ranks = np.arange(len(top_indices))

                batch_features = build_features_batch(
                    cache, origin_gsi, st, current_gsi, current_ti,
                    top_cand_gsis, top_cand_tis,
                    hop, resolved_gender, resolved_number,
                    chain_deps, chain_pos, chain_noun_indices,
                    top_ranks, len(candidates), num_gm, num_propn,
                )

                for i, idx in enumerate(top_indices):
                    cgsi, cti, _ = candidates[idx]
                    is_correct = (cgsi, cti) in correct_positions
                    features_list.append(batch_features[i])
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
                chain_deps = np.append(chain_deps, next_info[1])
                chain_pos = np.append(chain_pos, next_info[0])

                next_noun_idx = cache.noun_lookup.get((next_gsi, next_ti), -1)
                if next_noun_idx >= 0:
                    chain_noun_indices.append(next_noun_idx)

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
