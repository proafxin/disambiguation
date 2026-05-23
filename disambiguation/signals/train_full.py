import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from datasets import load_from_disk
from sklearn.metrics import average_precision_score

from disambiguation.signals.abstract_features import (
    POS_IDS,
    NUM_FEATURES,
    FEATURE_NAMES,
)
from disambiguation.signals.resolution_graph import ResolutionGraph

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
SPACY_TRF_DIR = CACHE_DIR / "spacy_trf"
DATA_DIR = CACHE_DIR.parent / "data"
MAX_HOPS = 512
TOP_K = 20
NEG_SAMPLES = 4
N_TOKEN_FEATURES = 15

DATASET_CONFIG = [
    ("preco", ["train"]),
    ("litbank", ["train", "validation", "test"]),
    ("corefud", ["train", "validation"]),
]


def _sent_lens(ds_name: str, sample: dict) -> list[int]:
    if ds_name == "corefud":
        return [len(sent["tokens"]) for sent in sample["sentences"]]
    return [len(s) for s in sample["sentences"]]


def _clusters(ds_name: str, sample: dict) -> list[list[list[int]]]:
    if ds_name == "preco":
        return sample["mention_clusters"]
    if ds_name == "litbank":
        return [[[m[0], m[1], m[2] + 1] for m in chain] for chain in sample["coref_chains"]]
    # corefud
    sent_id_to_local: dict[str, int] = {}
    for si, sent in enumerate(sample["sentences"]):
        sent_id_to_local[sent["sent_id"]] = si
    result = []
    for entity in sample.get("coref_entities", []):
        if len(entity) < 2:
            continue
        cluster = []
        for mention in entity:
            sid = mention["sent_id"]
            if sid not in sent_id_to_local:
                continue
            si = sent_id_to_local[sid]
            span = mention["span"]
            if "-" in span:
                parts = span.split("-")
                st, en = int(parts[0]) - 1, int(parts[1])
            else:
                st = int(span) - 1
                en = st + 1
            cluster.append([si, st, en])
        if len(cluster) >= 2:
            result.append(cluster)
    return result


class CachedData:
    def __init__(self) -> None:
        print("Loading cache from spacy_trf...")
        start = time.time()

        all_parts: list[np.ndarray] = []
        sent_offsets_list: list[int] = []
        doc_boundaries: list[tuple] = []
        clusters_by_doc: list = []
        dataset_ranges: dict = {}

        global_sent_idx = 0
        global_doc_idx = 0
        cumulative_tokens = 0

        for ds_name, splits in DATASET_CONFIG:
            ds_dict = load_from_disk(str(DATA_DIR / ds_name))
            dataset_ranges[ds_name] = {"start_doc": global_doc_idx}

            for split_name in splits:
                if split_name not in ds_dict:
                    continue
                split_ds = ds_dict[split_name]

                tok_offsets = np.load(SPACY_TRF_DIR / f"{ds_name}_{split_name}_offsets.npy")
                n_tokens = int(tok_offsets[-1])
                tok_data = np.memmap(
                    SPACY_TRF_DIR / f"{ds_name}_{split_name}_data.npy",
                    dtype=np.float32, mode="r", shape=(n_tokens, N_TOKEN_FEATURES),
                )
                all_parts.append(np.array(tok_data))

                for doc_i, sample in enumerate(split_ds):
                    doc_tok_start = cumulative_tokens + int(tok_offsets[doc_i])
                    sls = _sent_lens(ds_name, sample)

                    start_gsi = global_sent_idx
                    running = doc_tok_start
                    for sl in sls:
                        sent_offsets_list.append(running)
                        running += sl
                        global_sent_idx += 1
                    end_gsi = global_sent_idx

                    doc_boundaries.append((start_gsi, end_gsi, ds_name, doc_i))
                    clusters_by_doc.append(_clusters(ds_name, sample))
                    global_doc_idx += 1

                cumulative_tokens += n_tokens
                print(f"  {ds_name}/{split_name}: {len(split_ds)} docs loaded")

            dataset_ranges[ds_name]["end_doc"] = global_doc_idx

        sent_offsets_list.append(cumulative_tokens)

        self.token_data = np.concatenate(all_parts, axis=0)
        self.sent_offsets = np.array(sent_offsets_list, dtype=np.int64)
        self.doc_boundaries = doc_boundaries
        self.clusters = clusters_by_doc

        with open(CACHE_DIR / "dataset_ranges.json", "w") as f:
            json.dump(dataset_ranges, f, indent=2)

        elapsed = time.time() - start
        ram = self.token_data.nbytes / (1024 ** 3)
        print(f"  Loaded in {elapsed:.1f}s, {ram:.2f} GB RAM")
        print(f"  {len(doc_boundaries)} docs, {global_sent_idx} sents, {cumulative_tokens} tokens")

    def get_token_info(self, global_sent_idx: int, token_idx: int) -> np.ndarray:
        return self.token_data[int(self.sent_offsets[global_sent_idx]) + token_idx]

    def get_sent_length(self, global_sent_idx: int) -> int:
        return int(self.sent_offsets[global_sent_idx + 1] - self.sent_offsets[global_sent_idx])



def build_doc_arrays(cache: CachedData, start_gsi: int, end_gsi: int) -> dict:
    doc_start_abs = int(cache.sent_offsets[start_gsi])
    doc_end_abs = int(cache.sent_offsets[end_gsi])
    doc_len = doc_end_abs - doc_start_abs

    pos_slice = cache.token_data[doc_start_abs:doc_end_abs, 0].astype(np.int32)

    # Per-sentence propn counts and quote masks (indexed by gsi - start_gsi)
    n_sents = end_gsi - start_gsi
    sent_propn_counts = np.zeros(n_sents, dtype=np.int32)
    sent_quote_masks = []
    PUNCT_ID = POS_IDS.get("PUNCT", 13)
    for i, gsi in enumerate(range(start_gsi, end_gsi)):
        sent_len = cache.get_sent_length(gsi)
        sent_abs = int(cache.sent_offsets[gsi])
        sent_pos = cache.token_data[sent_abs:sent_abs + sent_len, 0].astype(np.int32)
        sent_propn_counts[i] = int(np.sum(sent_pos == POS_IDS["PROPN"]))
        sent_quote_masks.append((sent_pos == PUNCT_ID))

    # POS cumulative counts for prior_same_pos (indexed by abs_pos - doc_start_abs)
    pos_cumcounts = {}
    for pos_id in np.unique(pos_slice):
        mask = pos_slice == pos_id
        pos_cumcounts[int(pos_id)] = np.cumsum(mask)

    # PROPN salience arrays (indexed by abs_pos - doc_start_abs)
    propn_type_first = np.full(doc_len, -1, dtype=np.int64)   # first abs_pos of this type
    propn_type_freq = np.zeros(doc_len, dtype=np.int32)
    fp_first: dict[tuple, int] = {}
    fp_freq: dict[tuple, int] = {}
    for gsi in range(start_gsi, end_gsi):
        sent_len = cache.get_sent_length(gsi)
        for ti in range(sent_len):
            info = cache.get_token_info(gsi, ti)
            if int(info[0]) != POS_IDS["PROPN"]:
                continue
            fp = (int(info[0]), int(info[1]), int(info[2]), int(info[3]))
            abs_pos = int(cache.sent_offsets[gsi]) + ti
            doc_pos = abs_pos - doc_start_abs
            fp_freq[fp] = fp_freq.get(fp, 0) + 1
            if fp not in fp_first:
                fp_first[fp] = abs_pos
            propn_type_first[doc_pos] = fp_first[fp]
            propn_type_freq[doc_pos] = fp_freq[fp]

    return {
        "doc_start_abs": doc_start_abs,
        "doc_len": doc_len,
        "pos_cumcounts": pos_cumcounts,
        "sent_propn_counts": sent_propn_counts,
        "sent_quote_masks": sent_quote_masks,
        "propn_type_first": propn_type_first,
        "propn_type_freq": propn_type_freq,
    }


def _structural_candidates(
    cache: CachedData,
    cur_abs: int,
    gsi_lo: int,
    gsi_hi: int,
    start_gsi: int,
    end_gsi: int,
    window_tokens: int,
    visited_abs: np.ndarray,
    doc_start_abs: int,
    doc_len: int,
    resolved_gender: int,
    resolved_number: int,
    graph_cluster_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    range_start = int(cache.sent_offsets[gsi_lo])
    range_end = int(cache.sent_offsets[gsi_hi])
    if range_end <= range_start:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    abs_positions = np.arange(range_start, range_end, dtype=np.int64)
    tokens_in_range = cache.token_data[range_start:range_end, 0]
    gender_in_range = cache.token_data[range_start:range_end, 2]
    number_in_range = cache.token_data[range_start:range_end, 3]

    nominal_mask = (
        (tokens_in_range == POS_IDS["PROPN"])
        | (tokens_in_range == POS_IDS["NOUN"])
        | (tokens_in_range == POS_IDS["PRON"])
    )
    window_mask = np.abs(abs_positions - cur_abs) <= window_tokens

    doc_rel = abs_positions - doc_start_abs
    valid_doc_rel = (doc_rel >= 0) & (doc_rel < doc_len)
    visited_mask = np.zeros(len(abs_positions), dtype=bool)
    valid_idx = np.where(valid_doc_rel)[0]
    visited_mask[valid_idx] = visited_abs[doc_rel[valid_idx]]

    # Structural pre-filter: keep if gender/number compatible OR PROPN OR in graph
    gender_ok = (
        (resolved_gender == 3) | (gender_in_range == 3) | (gender_in_range == resolved_gender)
    )
    number_ok = (
        (resolved_number == 2) | (number_in_range == 2) | (number_in_range == resolved_number)
    )
    morph_ok = gender_ok & number_ok
    is_propn = tokens_in_range == POS_IDS["PROPN"]
    in_graph = np.zeros(len(abs_positions), dtype=bool)
    in_graph[valid_idx] = graph_cluster_ids[doc_rel[valid_idx]] >= 0
    structural_ok = morph_ok | is_propn | in_graph

    valid_mask = nominal_mask & window_mask & ~visited_mask & structural_ok
    valid_abs_arr = abs_positions[valid_mask]

    if len(valid_abs_arr) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    gsi_range = np.arange(gsi_lo, gsi_hi)
    sent_ends = cache.sent_offsets[gsi_range + 1]
    gsi_indices = np.searchsorted(sent_ends, valid_abs_arr, side="right")
    valid_filter = gsi_indices < len(gsi_range)
    valid_abs_arr = valid_abs_arr[valid_filter]
    gsi_indices = gsi_indices[valid_filter]
    cand_gsis = gsi_range[gsi_indices]
    cand_tis = valid_abs_arr - cache.sent_offsets[cand_gsis]

    return cand_gsis, cand_tis, valid_abs_arr


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
    num_cands: int,
    num_gender_match: int,
    num_propn_cands: int,
    graph_cluster_ids: np.ndarray,
    graph_confidences: np.ndarray,
    origin_cluster_id: int,
    doc_start_abs: int,
    propn_first_abs: np.ndarray,
    propn_type_first: np.ndarray,
    propn_type_freq: np.ndarray,
    sent_propn_counts: np.ndarray,
    sent_quote_mask: np.ndarray,
    sent_offsets_doc: np.ndarray,
    origin_doc_pos: float,
    prior_same_pos: int,
    chain_progress: float,
    start_gsi: int,
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
    sht = ((cand_gsis == current_gsi) & (c_all[:, 12] >= 0) & (cur[12] >= 0) & (c_all[:, 12] == cur[12])).astype(np.float32)
    cva = ((c_all[:, 13] == verb_id) & ((c_all[:, 6] == 1) | (c_all[:, 7] == 1))).astype(np.float32)
    ova = float(o[13] == verb_id and (o[6] == 1 or o[7] == 1))

    # Graph signals — vectorized via pre-built arrays
    cand_abs_doc = cand_abs - doc_start_abs
    valid_abs = (cand_abs_doc >= 0) & (cand_abs_doc < len(graph_cluster_ids))
    cand_cluster_ids = np.full(n, -1, dtype=np.int32)
    cand_cluster_ids[valid_abs] = graph_cluster_ids[cand_abs_doc[valid_abs]]
    graph_resolved = (cand_cluster_ids >= 0).astype(np.float32)
    graph_confidence = np.zeros(n, dtype=np.float32)
    graph_confidence[valid_abs] = graph_confidences[cand_abs_doc[valid_abs]]
    graph_same_cluster = ((cand_cluster_ids >= 0) & (cand_cluster_ids == origin_cluster_id)).astype(np.float32)

    # PROPN salience — vectorized via pre-built per-token arrays
    propn_first_dist = np.zeros(n, dtype=np.float32)
    propn_freq = np.zeros(n, dtype=np.float32)
    propn_mask = c_all[:, 0] == POS_IDS["PROPN"]
    if propn_mask.any():
        pm_abs = cand_abs[propn_mask] - doc_start_abs
        valid_pm = (pm_abs >= 0) & (pm_abs < len(propn_first_abs))
        pm_abs_v = pm_abs[valid_pm]
        first_abs_vals = propn_type_first[pm_abs_v]
        propn_first_dist_vals = np.abs(cur_abs - (first_abs_vals + doc_start_abs)).astype(np.float32)
        propn_freq_vals = propn_type_freq[pm_abs_v].astype(np.float32)
        idx = np.where(propn_mask)[0][valid_pm]
        propn_first_dist[idx] = propn_first_dist_vals
        propn_freq[idx] = propn_freq_vals

    # Discourse context — vectorized
    cand_token_pos_in_sent = cand_tis.astype(np.float32)
    gsi_local = cand_gsis - start_gsi
    n_sents = len(sent_propn_counts)
    valid_gsi = (gsi_local >= 0) & (gsi_local < n_sents)
    cand_sent_propn_count = np.zeros(n, dtype=np.float32)
    if valid_gsi.any():
        cand_sent_propn_count[valid_gsi] = sent_propn_counts[gsi_local[valid_gsi]]
    # Quote adjacency
    n_qsents = len(sent_quote_mask)
    cand_in_quotes = np.zeros(n, dtype=np.float32)
    for i in range(n):
        gsi_off = int(gsi_local[i])
        if 0 <= gsi_off < n_qsents:
            ti = int(cand_tis[i])
            qmask = sent_quote_mask[gsi_off]
            if len(qmask) > 0:
                cand_in_quotes[i] = float(
                    (ti > 0 and qmask[ti - 1]) or (ti + 1 < len(qmask) and qmask[ti + 1])
                )

    features = np.empty((n, NUM_FEATURES), dtype=np.float32)
    features[:, 0] = o[0]; features[:, 1] = o[1]; features[:, 2] = o[2]; features[:, 3] = o[3]
    features[:, 4] = o[6]
    features[:, 5] = cur[0]; features[:, 6] = cur[1]; features[:, 7] = cur[2]; features[:, 8] = cur[3]
    features[:, 9] = c_all[:, 0]; features[:, 10] = c_all[:, 1]
    features[:, 11] = c_all[:, 2]; features[:, 12] = c_all[:, 3]
    features[:, 13] = c_all[:, 4]; features[:, 14] = c_all[:, 6]
    features[:, 15] = (c_all[:, 0] == cur[0]).astype(np.float32)
    features[:, 16] = c_all[:, 9]
    features[:, 17] = (cand_gsis == current_gsi).astype(np.float32)
    features[:, 18] = np.abs(cand_abs - cur_abs).astype(np.float32)
    features[:, 19] = np.abs(cand_gsis - current_gsi).astype(np.float32)
    features[:, 20] = hop_count
    features[:, 21] = gmo; features[:, 22] = nmo
    features[:, 23] = gmc; features[:, 24] = nmc
    features[:, 25] = rgm; features[:, 26] = rnm
    features[:, 27] = dep_con; features[:, 28] = pos_con; features[:, 29] = ipt
    features[:, 30] = bhv; features[:, 31] = sht; features[:, 32] = cva; features[:, 33] = ova
    features[:, 34] = num_cands; features[:, 35] = num_gender_match; features[:, 36] = num_propn_cands
    features[:, 37] = graph_resolved; features[:, 38] = graph_confidence; features[:, 39] = graph_same_cluster
    features[:, 40] = propn_first_dist; features[:, 41] = propn_freq
    features[:, 42] = cand_sent_propn_count
    features[:, 43] = cand_token_pos_in_sent
    features[:, 44] = float(origin_doc_pos)
    features[:, 45] = chain_progress
    features[:, 46] = float(prior_same_pos)
    features[:, 47] = cand_in_quotes
    features[:, 48] = float(cur[6])
    features[:, 49] = c_all[:, 6] * graph_resolved
    features[:, 50] = c_all[:, 6] * (1.0 - graph_resolved)
    return features


def generate_doc_episodes(
    cache: CachedData,
    doc_idx: int,
    window_tokens: int = 150,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    if rng is None:
        rng = np.random.default_rng(doc_idx)

    start_gsi, end_gsi, source, orig_idx = cache.doc_boundaries[doc_idx]
    clusters = cache.clusters[doc_idx]

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    ranks_list: list[int] = []  # embedding rank for benchmark analysis

    graph = ResolutionGraph()
    doc_arrays = build_doc_arrays(cache, start_gsi, end_gsi)
    doc_start_abs = doc_arrays["doc_start_abs"]
    doc_len = doc_arrays["doc_len"]

    # Pre-register all PROPN mentions
    for cluster in clusters:
        for mention in cluster:
            si, st, en = mention
            gsi = start_gsi + si
            if gsi < end_gsi and st < cache.get_sent_length(gsi):
                if cache.get_token_info(gsi, st)[0] == POS_IDS["PROPN"]:
                    graph.add_mention((gsi, st), is_propn=True)

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        correct_abs = set()
        for mention in cluster:
            si, st, _ = mention
            gsi = start_gsi + si
            if gsi < end_gsi:
                correct_abs.add(int(cache.sent_offsets[gsi]) + st)

        correct_mask = np.zeros(doc_len, dtype=bool)
        for abs_pos in correct_abs:
            doc_rel = abs_pos - doc_start_abs
            if 0 <= doc_rel < doc_len:
                correct_mask[doc_rel] = True

        for mention in cluster:
            si, st, _ = mention
            origin_gsi = start_gsi + si
            if origin_gsi >= end_gsi:
                continue
            if st >= cache.get_sent_length(origin_gsi):
                continue

            origin_info = cache.get_token_info(origin_gsi, st)
            if origin_info[0] == POS_IDS["PROPN"]:
                continue

            # Skip if already resolved: cluster has a PROPN or a graph-resolved canonical
            graph_cluster_ids_pre, _ = graph.build_abs_arrays(
                cache.sent_offsets, doc_start_abs, doc_len
            )
            origin_abs_pre = int(cache.sent_offsets[origin_gsi]) + st
            if origin_abs_pre - doc_start_abs < doc_len:
                if graph_cluster_ids_pre[origin_abs_pre - doc_start_abs] >= 0:
                    continue

            origin_abs = int(cache.sent_offsets[origin_gsi]) + st
            origin_doc_pos = (origin_abs - doc_start_abs) / max(doc_len - 1, 1)
            origin_pos_id = int(origin_info[0])
            cum = doc_arrays["pos_cumcounts"].get(origin_pos_id)
            prior_same_pos = int(cum[origin_abs - doc_start_abs - 1]) if cum is not None and origin_abs - doc_start_abs > 0 else 0

            current_gsi = origin_gsi
            current_ti = st
            visited_abs = np.zeros(doc_len, dtype=bool)
            visited_abs[origin_abs - doc_start_abs] = True

            resolved_gender = int(origin_info[2]) if origin_info[2] != 3 else 3
            resolved_number = int(origin_info[3]) if origin_info[3] != 2 else 2
            chain_deps_list = [int(origin_info[1])]
            chain_pos_list = [int(origin_info[0])]

            for hop in range(MAX_HOPS):
                cur_abs = int(cache.sent_offsets[current_gsi]) + current_ti
                gsi_lo = max(start_gsi, current_gsi - 20)
                gsi_hi = min(end_gsi, current_gsi + 20)

                graph_cluster_ids, graph_confidences = graph.build_abs_arrays(
                    cache.sent_offsets, doc_start_abs, doc_len
                )
                origin_cluster_id = graph_cluster_ids[origin_abs - doc_start_abs] if origin_abs - doc_start_abs < doc_len else -1

                cand_gsis, cand_tis, valid_abs_arr = _structural_candidates(
                    cache, cur_abs, gsi_lo, gsi_hi, start_gsi, end_gsi,
                    window_tokens, visited_abs, doc_start_abs, doc_len,
                    resolved_gender, resolved_number, graph_cluster_ids,
                )

                if len(cand_gsis) == 0:
                    break

                # Take TOP_K closest by token distance to cur_abs
                distances = np.abs(valid_abs_arr - cur_abs)
                top_indices = np.argsort(distances)[:TOP_K]
                top_cand_gsis = cand_gsis[top_indices]
                top_cand_tis = cand_tis[top_indices]
                top_abs = valid_abs_arr[top_indices]

                # Competition features
                all_pos = cache.token_data[valid_abs_arr, 0]
                all_gender = cache.token_data[valid_abs_arr, 2]
                num_propn = int(np.sum(all_pos == POS_IDS["PROPN"]))
                num_gm = int(np.sum(
                    (all_gender == 3) | (all_gender == resolved_gender) | (resolved_gender == 3)
                ))

                chain_progress = float(hop) / max(doc_len - 1, 1)

                chain_deps = np.array(chain_deps_list, dtype=np.float32)
                chain_pos = np.array(chain_pos_list, dtype=np.float32)
                batch_features = build_features_batch(
                    cache, origin_gsi, st, current_gsi, current_ti,
                    top_cand_gsis, top_cand_tis,
                    hop, resolved_gender, resolved_number,
                    chain_deps, chain_pos,
                    len(cand_gsis), num_gm, num_propn,
                    graph_cluster_ids, graph_confidences, origin_cluster_id,
                    doc_start_abs,
                    doc_arrays["propn_type_first"],
                    doc_arrays["propn_type_first"],
                    doc_arrays["propn_type_freq"],
                    doc_arrays["sent_propn_counts"],
                    doc_arrays["sent_quote_masks"],
                    cache.sent_offsets[start_gsi:end_gsi + 1],
                    origin_doc_pos, prior_same_pos, chain_progress,
                    start_gsi,
                )

                # Labels — O(1) lookup; positions are bounds-checked by _structural_candidates
                is_correct = correct_mask[top_abs - doc_start_abs]

                pos_idx = np.where(is_correct)[0]
                neg_idx = np.where(~is_correct)[0]
                if len(neg_idx) > NEG_SAMPLES:
                    neg_idx = rng.choice(neg_idx, size=NEG_SAMPLES, replace=False)
                keep = np.concatenate([pos_idx, neg_idx])
                for i in keep:
                    features_list.append(batch_features[i])
                    labels_list.append(int(is_correct[i]))
                    ranks_list.append(int(i))

                # Teacher forcing: advance to nearest correct candidate in full pool
                correct_in_pool = correct_mask[valid_abs_arr - doc_start_abs]
                if not correct_in_pool.any():
                    break

                correct_pool_abs = valid_abs_arr[correct_in_pool]
                nearest_idx = np.argmin(np.abs(correct_pool_abs - cur_abs))
                next_abs = int(correct_pool_abs[nearest_idx])

                gsi_range = np.arange(gsi_lo, gsi_hi)
                sent_ends = cache.sent_offsets[gsi_range + 1]
                next_gsi_idx = np.searchsorted(sent_ends, next_abs, side="right")
                if next_gsi_idx >= len(gsi_range):
                    break
                next_gsi = int(gsi_range[next_gsi_idx])
                next_ti = next_abs - int(cache.sent_offsets[next_gsi])

                visited_abs[next_abs - doc_start_abs] = True
                current_gsi = next_gsi
                current_ti = next_ti

                next_info = cache.get_token_info(next_gsi, next_ti)
                chain_deps_list.append(int(next_info[1]))
                chain_pos_list.append(int(next_info[0]))

                next_is_propn = bool(next_info[0] == POS_IDS["PROPN"])
                graph.link(
                    (origin_gsi, st), (next_gsi, next_ti),
                    confidence=1.0,
                    is_b_propn=next_is_propn,
                )

                if resolved_gender == 3 and next_info[2] != 3:
                    resolved_gender = int(next_info[2])
                if resolved_number == 2 and next_info[3] != 2:
                    resolved_number = int(next_info[3])

                if next_is_propn:
                    break

    return features_list, labels_list, ranks_list


def train_full(window_tokens: int = 150) -> None:
    cache = CachedData()

    with open(CACHE_DIR / "dataset_ranges.json") as f:
        ranges = json.load(f)

    rng = np.random.default_rng(42)

    GULLIVER_IDX = 36676

    # 2:1 stratified split per dataset, Gulliver held out
    train_indices, test_indices = [], []
    for name, r in ranges.items():
        docs = np.array([i for i in range(r["start_doc"], r["end_doc"]) if i != GULLIVER_IDX])
        shuffled = rng.permutation(docs)
        n_train = int(len(shuffled) * 2 / 3)
        train_indices.extend(shuffled[:n_train])
        test_indices.extend(shuffled[n_train:])
        print(f"  {name}: {n_train} train, {len(shuffled) - n_train} test (from {len(docs)} docs)")

    train_indices = rng.permutation(train_indices)
    test_indices = rng.permutation(test_indices)
    num_train_docs = len(train_indices)
    num_test_docs = len(test_indices)

    print(f"\nGenerating train episodes ({num_train_docs} docs)...")
    start = time.time()
    train_features, train_labels = [], []

    for i, doc_idx in enumerate(train_indices):
        feats, labels, _ = generate_doc_episodes(cache, int(doc_idx), window_tokens)
        train_features.extend(feats)
        train_labels.extend(labels)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{num_train_docs} docs, {len(train_features)} episodes")

    X_train = np.array(train_features)
    y_train = np.array(train_labels, dtype=np.int32)
    print(f"  Train: {X_train.shape[0]} episodes, {time.time()-start:.1f}s, pos_rate={y_train.mean():.4f}")

    print(f"\nGenerating test episodes ({num_test_docs} docs)...")
    test_features, test_labels = [], []
    for doc_idx in test_indices:
        feats, labels, _ = generate_doc_episodes(cache, int(doc_idx), window_tokens)
        test_features.extend(feats)
        test_labels.extend(labels)

    X_test = np.array(test_features)
    y_test = np.array(test_labels, dtype=np.int32)
    print(f"  Test: {X_test.shape[0]} episodes, pos_rate={y_test.mean():.4f}")

    print("\nTraining XGBoost (GPU)...")
    start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=20, learning_rate=0.05,
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
    train_full(window_tokens=150)
