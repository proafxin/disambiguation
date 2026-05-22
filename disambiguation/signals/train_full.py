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
from disambiguation.signals.resolution_graph import ResolutionGraph

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
MAX_HOPS = 512
TOP_K = 20
# Confidence threshold to commit a link to the resolution graph
COMMIT_THRESHOLD = 0.7


class CachedData:
    def __init__(self) -> None:
        print("Loading cache...")
        start = time.time()

        ti = np.load(CACHE_DIR / "token_infos.npz")
        self.token_data = ti["data"].astype(np.float32)
        self.sent_offsets = ti["offsets"]

        se = np.load(CACHE_DIR / "sentence_embeddings.npz")
        self.sent_embs = se["data"].astype(np.float32)
        norms = np.linalg.norm(self.sent_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.sent_embs_normed = self.sent_embs / norms

        ne = np.load(CACHE_DIR / "noun_embeddings.npz")
        self.noun_embs = ne["data"].astype(np.float32)
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
        return self.token_data[int(self.sent_offsets[global_sent_idx]) + token_idx]

    def get_sent_length(self, global_sent_idx: int) -> int:
        return int(self.sent_offsets[global_sent_idx + 1] - self.sent_offsets[global_sent_idx])

    def cosine_sim_batch(self, query_idx: int, cand_indices: np.ndarray) -> np.ndarray:
        return self.sent_embs_normed[query_idx] @ self.sent_embs_normed[cand_indices].T


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
    graph: ResolutionGraph,
    origin_key: tuple,
) -> np.ndarray:
    n = len(cand_gsis)
    o = cache.get_token_info(origin_gsi, origin_ti)
    cur = cache.get_token_info(current_gsi, current_ti)

    cur_abs = int(cache.sent_offsets[current_gsi]) + current_ti
    cand_starts = cache.sent_offsets[cand_gsis]
    cand_abs = cand_starts + cand_tis
    c_all = cache.token_data[cand_starts + cand_tis]

    # Agreement (vectorized)
    gmo = ((o[2] == 3) | (c_all[:, 2] == 3) | (o[2] == c_all[:, 2])).astype(np.float32)
    nmo = ((o[3] == 2) | (c_all[:, 3] == 2) | (o[3] == c_all[:, 3])).astype(np.float32)
    gmc = ((cur[2] == 3) | (c_all[:, 2] == 3) | (cur[2] == c_all[:, 2])).astype(np.float32)
    nmc = ((cur[3] == 2) | (c_all[:, 3] == 2) | (cur[3] == c_all[:, 3])).astype(np.float32)
    rgm = ((resolved_gender == 3) | (c_all[:, 2] == 3) | (c_all[:, 2] == resolved_gender)).astype(np.float32)
    rnm = ((resolved_number == 2) | (c_all[:, 3] == 2) | (c_all[:, 3] == resolved_number)).astype(np.float32)

    # Chain consistency (vectorized)
    if len(chain_deps) > 0:
        dep_con = (c_all[:, 1:2] == chain_deps).mean(axis=1).astype(np.float32)
        pos_con = (c_all[:, 0:1] == chain_pos).mean(axis=1).astype(np.float32)
    else:
        dep_con = np.zeros(n, dtype=np.float32)
        pos_con = np.zeros(n, dtype=np.float32)

    ipt = ((c_all[:, 0] == POS_IDS["PROPN"]) & (o[0] == POS_IDS["PRON"])).astype(np.float32)

    # Verb/head association (vectorized)
    verb_id = POS_IDS["VERB"]
    bhv = ((o[13] == verb_id) & (c_all[:, 13] == verb_id)).astype(np.float32)
    shpo = (o[13] == c_all[:, 13]).astype(np.float32)
    shpc = (cur[13] == c_all[:, 13]).astype(np.float32)
    sht = ((cand_gsis == current_gsi) & (c_all[:, 12] >= 0) & (cur[12] >= 0) & (c_all[:, 12] == cur[12])).astype(np.float32)
    cva = ((c_all[:, 13] == verb_id) & ((c_all[:, 6] == 1) | (c_all[:, 7] == 1))).astype(np.float32)
    ova = float(o[13] == verb_id and (o[6] == 1 or o[7] == 1))

    # Noun similarity (vectorized)
    noun_sims = np.zeros(n, dtype=np.float32)
    if chain_noun_indices:
        chain_embs = cache.noun_embs_normed[chain_noun_indices]
        for i in range(n):
            idx = cache.noun_lookup.get((int(cand_gsis[i]), int(cand_tis[i])), -1)
            if idx >= 0:
                noun_sims[i] = float((chain_embs @ cache.noun_embs_normed[idx]).max())

    # Graph signals (3 new features)
    # For each candidate: is it already resolved? confidence? same cluster as origin?
    graph_resolved = np.zeros(n, dtype=np.float32)
    graph_confidence = np.zeros(n, dtype=np.float32)
    graph_same_cluster = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ckey = (int(cand_gsis[i]), int(cand_tis[i]))
        graph_resolved[i] = float(graph.is_resolved(ckey))
        graph_confidence[i] = graph.get_confidence(ckey)
        graph_same_cluster[i] = float(graph.same_cluster(origin_key, ckey))

    features = np.empty((n, NUM_FEATURES), dtype=np.float32)
    features[:, 0] = o[0]; features[:, 1] = o[1]; features[:, 2] = o[2]; features[:, 3] = o[3]
    features[:, 4] = o[6]; features[:, 5] = o[8]
    features[:, 6] = cur[0]; features[:, 7] = cur[1]; features[:, 8] = cur[2]; features[:, 9] = cur[3]
    features[:, 10] = cur[6]; features[:, 11] = cur[8]
    features[:, 12] = c_all[:, 0]; features[:, 13] = c_all[:, 1]
    features[:, 14] = c_all[:, 2]; features[:, 15] = c_all[:, 3]
    features[:, 16] = c_all[:, 4]; features[:, 17] = c_all[:, 6]
    features[:, 18] = c_all[:, 7]; features[:, 19] = c_all[:, 8]
    features[:, 20] = (c_all[:, 1] == cur[1]).astype(np.float32)
    features[:, 21] = (c_all[:, 1] == o[1]).astype(np.float32)
    features[:, 22] = (c_all[:, 0] == cur[0]).astype(np.float32)
    features[:, 23] = (c_all[:, 6] == o[6]).astype(np.float32)
    features[:, 24] = c_all[:, 9]
    features[:, 25] = (cand_gsis == current_gsi).astype(np.float32)
    features[:, 26] = np.abs(cand_abs - cur_abs).astype(np.float32)
    features[:, 27] = np.abs(cand_gsis - current_gsi).astype(np.float32)
    features[:, 28] = hop_count
    features[:, 29] = gmo; features[:, 30] = nmo
    features[:, 31] = gmc; features[:, 32] = nmc
    features[:, 33] = rgm; features[:, 34] = rnm
    features[:, 35] = dep_con; features[:, 36] = pos_con; features[:, 37] = ipt
    features[:, 38] = bhv; features[:, 39] = shpo; features[:, 40] = shpc
    features[:, 41] = sht; features[:, 42] = cva; features[:, 43] = ova
    features[:, 44] = ranks.astype(np.float32)
    features[:, 45] = num_cands; features[:, 46] = num_gender_match
    features[:, 47] = num_propn_cands; features[:, 48] = noun_sims
    # Graph signals
    features[:, 49] = graph_resolved
    features[:, 50] = graph_confidence
    features[:, 51] = graph_same_cluster
    return features


def generate_doc_episodes(cache: CachedData, doc_idx: int, window_tokens: int = 150) -> tuple[list[np.ndarray], list[int]]:
    start_gsi, end_gsi, source, orig_idx = cache.doc_boundaries[doc_idx]
    clusters = cache.clusters[doc_idx]

    features_list = []
    labels_list = []

    # One resolution graph per document — shared across all mentions
    graph = ResolutionGraph()

    # Pre-register all PROPN mentions as potential canonical entities
    for cluster in clusters:
        for mention in cluster:
            si, st, en = mention
            gsi = start_gsi + si
            if gsi < end_gsi and st < cache.get_sent_length(gsi):
                info = cache.get_token_info(gsi, st)
                if info[0] == POS_IDS["PROPN"]:
                    graph.add_mention((gsi, st), is_propn=True)

    for cluster in clusters:
        if len(cluster) < 2:
            continue

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
                gsi_lo = max(start_gsi, current_gsi - 20)
                gsi_hi = min(end_gsi, current_gsi + 20)

                range_start = int(cache.sent_offsets[gsi_lo])
                range_end = int(cache.sent_offsets[gsi_hi])
                if range_end <= range_start:
                    break

                tokens_in_range = cache.token_data[range_start:range_end, 0]
                abs_positions = np.arange(range_start, range_end)

                nominal_mask = np.isin(tokens_in_range, [POS_IDS["NOUN"], POS_IDS["PROPN"], POS_IDS["PRON"]])
                window_mask = np.abs(abs_positions - cur_abs) <= window_tokens
                valid_abs = abs_positions[nominal_mask & window_mask]

                if len(valid_abs) == 0:
                    break

                gsi_range = np.arange(gsi_lo, gsi_hi)
                sent_ends = cache.sent_offsets[gsi_range + 1]

                gsi_indices = np.searchsorted(sent_ends, valid_abs, side="right")
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

                cand_gsis_arr = np.array([c[0] for c in candidates])
                sims = cache.cosine_sim_batch(origin_gsi, cand_gsis_arr)
                top_indices = np.argsort(-sims)[:TOP_K]

                all_gsis = np.array([c[0] for c in candidates])
                all_tis = np.array([c[1] for c in candidates])
                all_starts = cache.sent_offsets[all_gsis]
                all_pos = cache.token_data[all_starts + all_tis, 0]
                all_gender = cache.token_data[all_starts + all_tis, 2]

                num_propn = int(np.sum(all_pos == POS_IDS["PROPN"]))
                num_gm = int(np.sum(
                    (all_gender == 3) | (all_gender == resolved_gender) | (resolved_gender == 3)
                ))

                top_cand_gsis = np.array([candidates[idx][0] for idx in top_indices])
                top_cand_tis = np.array([candidates[idx][1] for idx in top_indices])
                top_ranks = np.arange(len(top_indices))

                batch_features = build_features_batch(
                    cache, origin_gsi, st, current_gsi, current_ti,
                    top_cand_gsis, top_cand_tis,
                    hop, resolved_gender, resolved_number,
                    chain_deps, chain_pos, chain_noun_indices,
                    top_ranks, len(candidates), num_gm, num_propn,
                    graph, (origin_gsi, st),
                )

                for i, idx in enumerate(top_indices):
                    cgsi, cti, _ = candidates[idx]
                    is_correct = (cgsi, cti) in correct_positions
                    features_list.append(batch_features[i])
                    labels_list.append(1 if is_correct else 0)

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

                # Commit this link to the resolution graph
                next_is_propn = bool(next_info[0] == POS_IDS["PROPN"])
                graph.link(
                    (origin_gsi, st), (next_gsi, next_ti),
                    confidence=1.0,
                    is_b_propn=next_is_propn,
                )

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

    rng = np.random.default_rng(42)
    all_indices = rng.permutation(len(cache.doc_boundaries))
    train_indices = all_indices[:num_train_docs]
    test_indices = all_indices[num_train_docs:num_train_docs + num_test_docs]

    print(f"\nGenerating train episodes ({num_train_docs} docs)...")
    start = time.time()
    train_features, train_labels = [], []

    for i, doc_idx in enumerate(train_indices):
        feats, labels = generate_doc_episodes(cache, int(doc_idx), window_tokens)
        train_features.extend(feats)
        train_labels.extend(labels)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_train_docs} docs, {len(train_features)} episodes")

    X_train = np.array(train_features)
    y_train = np.array(train_labels, dtype=np.int32)
    print(f"  Train: {X_train.shape[0]} episodes, {time.time()-start:.1f}s, pos_rate={y_train.mean():.4f}")

    print(f"\nGenerating test episodes ({num_test_docs} docs)...")
    test_features, test_labels = [], []
    for doc_idx in test_indices:
        feats, labels = generate_doc_episodes(cache, int(doc_idx), window_tokens)
        test_features.extend(feats)
        test_labels.extend(labels)

    X_test = np.array(test_features)
    y_test = np.array(test_labels, dtype=np.int32)
    print(f"  Test: {X_test.shape[0]} episodes, pos_rate={y_test.mean():.4f}")

    print("\nTraining XGBoost (GPU)...")
    start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.1,
        subsample=0.8, min_child_weight=10, device="cuda",
        tree_method="hist", random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"  Training: {time.time()-start:.1f}s")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    ap = average_precision_score(y_test, y_prob)
    recall = y_test[y_pred == 1].sum() / max(y_test.sum(), 1)
    precision = y_test[y_pred == 1].sum() / max(y_pred.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n=== Results ===")
    print(f"AP={ap:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

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

    print("\nTop 15 features:")
    importances = model.feature_importances_
    for i in np.argsort(importances)[::-1][:15]:
        print(f"  {FEATURE_NAMES[i]:25s} {importances[i]:.4f}")

    model.save_model(str(CACHE_DIR / "xgb_model.json"))
    print(f"\nModel saved to {CACHE_DIR / 'xgb_model.json'}")


if __name__ == "__main__":
    train_full(num_train_docs=1000, num_test_docs=200, window_tokens=150)
