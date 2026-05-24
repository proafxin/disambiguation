import json
import time
from pathlib import Path

import numpy as np
import spacy
import spacy.tokens
from datasets import load_from_disk

from disambiguation.signals.abstract_features import (
    DEP_IDS, ENT_TYPE_IDS, GENDER_IDS, NUMBER_IDS, POS_IDS, PRONTYPE_IDS, NUM_FEATURES,
)
from disambiguation.signals.resolution_graph import ResolutionGraph

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
DATA_DIR = CACHE_DIR.parent / "data"
SPACY_TRF_DIR = DATA_DIR / "spacy_trf"
MAX_HOPS = 512
MAX_DEPTH = 20
TOP_K = 20
NEG_SAMPLES = 4
N_TOKEN_FEATURES = 16

DATASET_CONFIG = [
    ("preco", ["train"]),
    ("litbank", ["train", "validation", "test"]),
    ("corefud", ["train", "validation"]),
    ("conll2012", ["train", "validation", "test"]),
]


def _compute_depth(token: spacy.tokens.Token) -> int:
    depth = 0
    cur = token
    while cur.head != cur and depth < MAX_DEPTH:
        cur = cur.head
        depth += 1
    return depth


def _extract_chunk_features(doc: spacy.tokens.Doc, sent_lens: list[int]) -> np.ndarray:
    n_tokens = sum(sent_lens)
    buf = np.empty((n_tokens, N_TOKEN_FEATURES), dtype=np.float32)
    pos = 0
    abs_p = 0
    for sl in sent_lens:
        sent_start = abs_p
        for i in range(abs_p, abs_p + sl):
            tok = doc[i]
            morph = tok.morph.to_dict()
            dep = tok.dep_
            ti = i - sent_start
            head_rel = tok.head.i - sent_start if tok.head != tok else -1
            buf[pos, 0] = POS_IDS.get(tok.pos_, len(POS_IDS))
            buf[pos, 1] = DEP_IDS.get(dep, len(DEP_IDS))
            buf[pos, 2] = GENDER_IDS.get(morph.get("Gender", "unknown"), 3)
            buf[pos, 3] = NUMBER_IDS.get(morph.get("Number", "unknown"), 2)
            buf[pos, 4] = int(morph.get("Person", "0"))
            buf[pos, 5] = PRONTYPE_IDS.get(morph.get("PronType", "unknown"), 5)
            buf[pos, 6] = int(dep in {"nsubj", "nsubj:pass", "nsubj:outer", "csubj"})
            buf[pos, 7] = int(dep in {"obj", "iobj"})
            buf[pos, 8] = int(dep == "nmod:poss")
            buf[pos, 9] = _compute_depth(tok)
            buf[pos, 10] = tok.n_lefts + tok.n_rights
            buf[pos, 11] = ti / max(sl - 1, 1)
            buf[pos, 12] = head_rel
            buf[pos, 13] = POS_IDS.get(tok.head.pos_, len(POS_IDS))
            buf[pos, 14] = DEP_IDS.get(tok.head.dep_, len(DEP_IDS))
            buf[pos, 15] = ENT_TYPE_IDS.get(tok.ent_type_, len(ENT_TYPE_IDS))
            pos += 1
        abs_p += sl
    return buf


def _sent_lens(ds_name: str, sample: dict) -> list[int]:
    if ds_name == "corefud":
        return [len(sent["tokens"]) for sent in sample["sentences"]]
    return [len(s) for s in sample["sentences"]]


def _clusters(ds_name: str, sample: dict) -> list[list[list[int]]]:
    if ds_name in ("preco", "conll2012"):
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
        print("Loading spacy_trf data...")
        start = time.time()

        all_parts: list[np.ndarray] = []
        sent_offsets_list: list[int] = []
        doc_boundaries: list[tuple] = []
        clusters_by_doc: list = []
        dataset_ranges: dict = {}

        global_sent_idx = 0
        global_doc_idx = 0
        cumulative_tokens = 0
        vocab = spacy.blank("en").vocab

        for ds_name, splits in DATASET_CONFIG:
            ds_dict = load_from_disk(str(DATA_DIR / ds_name))
            dataset_ranges[ds_name] = {"start_doc": global_doc_idx}

            for split_name in splits:
                if split_name not in ds_dict:
                    continue
                split_ds = ds_dict[split_name]

                with (SPACY_TRF_DIR / f"{ds_name}_{split_name}_meta.json").open(encoding="utf-8") as f:
                    meta = json.load(f)
                doc_chunk_counts = meta["doc_chunk_counts"]

                doc_bin = spacy.tokens.DocBin().from_disk(SPACY_TRF_DIR / f"{ds_name}_{split_name}.spacy")
                chunk_iter = iter(doc_bin.get_docs(vocab))

                for doc_i, (sample, n_chunks) in enumerate(zip(split_ds, doc_chunk_counts)):
                    sls = _sent_lens(ds_name, sample)
                    doc_parts: list[np.ndarray] = []
                    for _ in range(n_chunks):
                        chunk = next(chunk_iter)
                        doc_parts.append(_extract_chunk_features(chunk, chunk.user_data["sent_lens"]))

                    start_gsi = global_sent_idx
                    running = cumulative_tokens
                    for sl in sls:
                        sent_offsets_list.append(running)
                        running += sl
                        global_sent_idx += 1

                    doc_token_data = np.concatenate(doc_parts) if len(doc_parts) > 1 else doc_parts[0]
                    all_parts.append(doc_token_data)
                    cumulative_tokens += len(doc_token_data)
                    doc_boundaries.append((start_gsi, global_sent_idx, ds_name, doc_i))
                    clusters_by_doc.append(_clusters(ds_name, sample))
                    global_doc_idx += 1

                print(f"  {ds_name}/{split_name}: {len(split_ds)} docs loaded")

            dataset_ranges[ds_name]["end_doc"] = global_doc_idx

        sent_offsets_list.append(cumulative_tokens)

        self.token_data = np.concatenate(all_parts, axis=0)
        self.sent_offsets = np.array(sent_offsets_list, dtype=np.int64)
        self.doc_boundaries = doc_boundaries
        self.clusters = clusters_by_doc

        with (DATA_DIR / "dataset_ranges.json").open("w", encoding="utf-8") as f:
            json.dump(dataset_ranges, f, indent=2)

        elapsed = time.time() - start
        ram = self.token_data.nbytes / (1024 ** 3)
        print(f"  Loaded in {elapsed:.1f}s, {ram:.2f} GB RAM")
        print(f"  {len(doc_boundaries)} docs, {global_sent_idx} sents, {cumulative_tokens} tokens")

    def get_token_info(self, global_sent_idx: int, token_idx: int) -> np.ndarray:
        return self.token_data[int(self.sent_offsets[global_sent_idx]) + token_idx]

    def get_sent_length(self, global_sent_idx: int) -> int:
        return int(self.sent_offsets[global_sent_idx + 1] - self.sent_offsets[global_sent_idx])



def build_doc_arrays(token_data: np.ndarray, sent_offsets: np.ndarray, start_gsi: int, end_gsi: int) -> dict:
    doc_start_abs = int(sent_offsets[start_gsi])
    doc_end_abs = int(sent_offsets[end_gsi])
    doc_len = doc_end_abs - doc_start_abs

    pos_slice = token_data[doc_start_abs:doc_end_abs, 0].astype(np.int32)
    PROPN_ID = POS_IDS["PROPN"]
    PUNCT_ID = POS_IDS.get("PUNCT", 13)

    # Per-sentence PROPN counts via reduceat (no Python loop over sentences)
    sent_starts_rel = (sent_offsets[start_gsi:end_gsi] - doc_start_abs).astype(np.int64)
    propn_mask_int = (pos_slice == PROPN_ID).astype(np.int32)
    sent_propn_counts = np.add.reduceat(propn_mask_int, sent_starts_rel).astype(np.int32)

    # Flat doc-level PUNCT mask for vectorized quote-adjacency in build_features_batch
    doc_punct_mask = pos_slice == PUNCT_ID

    # POS cumulative counts for prior_same_pos (indexed by abs_pos - doc_start_abs)
    pos_cumcounts = {}
    for pos_id in np.unique(pos_slice):
        pos_cumcounts[int(pos_id)] = np.cumsum(pos_slice == pos_id)

    # PROPN salience — fully vectorized via fingerprint sort (replaces nested Python loop)
    propn_type_first = np.full(doc_len, -1, dtype=np.int64)
    propn_type_freq = np.zeros(doc_len, dtype=np.int32)
    propn_rel = np.where(pos_slice == PROPN_ID)[0]
    if len(propn_rel) > 0:
        pd = token_data[doc_start_abs + propn_rel, :4].astype(np.int32)
        fp = pd[:, 0] * 3720 + pd[:, 1] * 12 + pd[:, 2] * 3 + pd[:, 3]  # 31*4*3=372, *10
        order = np.argsort(fp, stable=True)
        sorted_fp = fp[order]
        sorted_pos = propn_rel[order]
        bounds = np.concatenate([[0], np.where(np.diff(sorted_fp))[0] + 1])
        repeats = np.diff(np.concatenate([bounds, [len(sorted_fp)]]))
        within_idx = np.arange(len(sorted_fp)) - np.repeat(bounds, repeats)
        propn_type_freq[sorted_pos] = within_idx + 1
        propn_type_first[sorted_pos] = doc_start_abs + np.repeat(sorted_pos[bounds], repeats)

    return {
        "doc_start_abs": doc_start_abs,
        "doc_len": doc_len,
        "pos_cumcounts": pos_cumcounts,
        "sent_propn_counts": sent_propn_counts,
        "doc_punct_mask": doc_punct_mask,
        "propn_type_first": propn_type_first,
        "propn_type_freq": propn_type_freq,
    }


def _structural_candidates(
    token_data: np.ndarray,
    sent_offsets: np.ndarray,
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
    range_start = int(sent_offsets[gsi_lo])
    range_end = int(sent_offsets[gsi_hi])
    if range_end <= range_start:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    abs_positions = np.arange(range_start, range_end, dtype=np.int64)
    tokens_in_range = token_data[range_start:range_end, 0]
    gender_in_range = token_data[range_start:range_end, 2]
    number_in_range = token_data[range_start:range_end, 3]

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
    sent_ends = sent_offsets[gsi_range + 1]
    gsi_indices = np.searchsorted(sent_ends, valid_abs_arr, side="right")
    valid_filter = gsi_indices < len(gsi_range)
    valid_abs_arr = valid_abs_arr[valid_filter]
    gsi_indices = gsi_indices[valid_filter]
    cand_gsis = gsi_range[gsi_indices]
    cand_tis = valid_abs_arr - sent_offsets[cand_gsis]

    return cand_gsis, cand_tis, valid_abs_arr


def build_features_batch(
    token_data: np.ndarray,
    sent_offsets: np.ndarray,
    origin_abs: int,
    cur_abs: int,
    current_gsi: int,
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
    doc_len: int,
    propn_type_first: np.ndarray,
    propn_type_freq: np.ndarray,
    sent_propn_counts: np.ndarray,
    doc_punct_mask: np.ndarray,
    sent_offsets_doc: np.ndarray,
    origin_doc_pos: float,
    prior_same_pos: int,
    chain_progress: float,
    start_gsi: int,
) -> np.ndarray:
    n = len(cand_gsis)
    o = token_data[origin_abs]
    cur = token_data[cur_abs]

    cand_abs = sent_offsets[cand_gsis] + cand_tis
    c_all = token_data[cand_abs]

    # Agreement (direct bool→float32, no intermediate variables)
    features = np.empty((n, NUM_FEATURES), dtype=np.float32)
    features[:, 21] = (o[2] == 3) | (c_all[:, 2] == 3) | (o[2] == c_all[:, 2])
    features[:, 22] = (o[3] == 2) | (c_all[:, 3] == 2) | (o[3] == c_all[:, 3])
    features[:, 23] = (cur[2] == 3) | (c_all[:, 2] == 3) | (cur[2] == c_all[:, 2])
    features[:, 24] = (cur[3] == 2) | (c_all[:, 3] == 2) | (cur[3] == c_all[:, 3])
    features[:, 25] = (resolved_gender == 3) | (c_all[:, 2] == 3) | (c_all[:, 2] == resolved_gender)
    features[:, 26] = (resolved_number == 2) | (c_all[:, 3] == 2) | (c_all[:, 3] == resolved_number)

    # Chain consistency
    if len(chain_deps) > 0:
        features[:, 27] = (c_all[:, 1:2] == chain_deps).mean(axis=1)
        features[:, 28] = (c_all[:, 0:1] == chain_pos).mean(axis=1)
    else:
        features[:, 27] = 0.0
        features[:, 28] = 0.0
    features[:, 29] = (c_all[:, 0] == POS_IDS["PROPN"]) & (o[0] == POS_IDS["PRON"])

    # Verb/head association
    verb_id = POS_IDS["VERB"]
    features[:, 30] = (o[13] == verb_id) & (c_all[:, 13] == verb_id)
    features[:, 31] = (cand_gsis == current_gsi) & (c_all[:, 12] >= 0) & (cur[12] >= 0) & (c_all[:, 12] == cur[12])
    features[:, 32] = (c_all[:, 13] == verb_id) & ((c_all[:, 6] == 1) | (c_all[:, 7] == 1))
    features[:, 33] = float(o[13] == verb_id and (o[6] == 1 or o[7] == 1))

    # Graph signals
    cand_abs_doc = cand_abs - doc_start_abs
    valid_abs = (cand_abs_doc >= 0) & (cand_abs_doc < len(graph_cluster_ids))
    cand_cluster_ids = np.full(n, -1, dtype=np.int32)
    cand_cluster_ids[valid_abs] = graph_cluster_ids[cand_abs_doc[valid_abs]]
    graph_resolved = (cand_cluster_ids >= 0).astype(np.float32)
    graph_confidence = np.zeros(n, dtype=np.float32)
    graph_confidence[valid_abs] = graph_confidences[cand_abs_doc[valid_abs]]
    features[:, 37] = graph_resolved
    features[:, 38] = graph_confidence
    features[:, 39] = (cand_cluster_ids >= 0) & (cand_cluster_ids == origin_cluster_id)

    # PROPN salience
    propn_first_dist = np.zeros(n, dtype=np.float32)
    propn_freq = np.zeros(n, dtype=np.float32)
    propn_mask = c_all[:, 0] == POS_IDS["PROPN"]
    if propn_mask.any():
        pm_abs = cand_abs[propn_mask] - doc_start_abs
        valid_pm = (pm_abs >= 0) & (pm_abs < doc_len)
        pm_abs_v = pm_abs[valid_pm]
        first_abs_vals = propn_type_first[pm_abs_v]
        idx = np.where(propn_mask)[0][valid_pm]
        propn_first_dist[idx] = np.abs(cur_abs - first_abs_vals)
        propn_freq[idx] = propn_type_freq[pm_abs_v]

    # Discourse context
    gsi_local = cand_gsis - start_gsi
    n_sents = len(sent_propn_counts)
    valid_gsi = (gsi_local >= 0) & (gsi_local < n_sents)
    cand_sent_propn_count = np.zeros(n, dtype=np.float32)
    if valid_gsi.any():
        cand_sent_propn_count[valid_gsi] = sent_propn_counts[gsi_local[valid_gsi]]

    # Quote adjacency — vectorized via flat doc_punct_mask
    prev_doc = np.maximum(cand_abs_doc - 1, 0)
    next_doc = np.minimum(cand_abs_doc + 1, doc_len - 1)
    cand_in_quotes = (
        (doc_punct_mask[prev_doc] & (cand_abs_doc > 0)) |
        (doc_punct_mask[next_doc] & (cand_abs_doc < doc_len - 1))
    ).astype(np.float32)

    features[:, 0] = o[0]; features[:, 1] = o[1]; features[:, 2] = o[2]; features[:, 3] = o[3]
    features[:, 4] = o[6]
    features[:, 5] = cur[0]; features[:, 6] = cur[1]; features[:, 7] = cur[2]; features[:, 8] = cur[3]
    features[:, 9] = c_all[:, 0]; features[:, 10] = c_all[:, 1]
    features[:, 11] = c_all[:, 2]; features[:, 12] = c_all[:, 3]
    features[:, 13] = c_all[:, 4]; features[:, 14] = c_all[:, 6]
    features[:, 15] = c_all[:, 0] == cur[0]
    features[:, 16] = c_all[:, 9]
    features[:, 17] = cand_gsis == current_gsi
    features[:, 18] = np.abs(cand_abs - cur_abs)
    features[:, 19] = np.abs(cand_gsis - current_gsi)
    features[:, 20] = hop_count
    features[:, 34] = num_cands; features[:, 35] = num_gender_match; features[:, 36] = num_propn_cands
    features[:, 40] = propn_first_dist; features[:, 41] = propn_freq
    features[:, 42] = cand_sent_propn_count
    sent_lens = sent_offsets[cand_gsis + 1] - sent_offsets[cand_gsis]
    features[:, 43] = cand_tis / np.maximum(sent_lens, 1)
    features[:, 44] = float(origin_doc_pos)
    features[:, 45] = chain_progress
    features[:, 46] = float(prior_same_pos)
    features[:, 47] = cand_in_quotes
    features[:, 48] = float(cur[6])
    features[:, 49] = c_all[:, 6] * graph_resolved
    features[:, 50] = c_all[:, 6] * (1.0 - graph_resolved)
    features[:, 51] = o[15]
    features[:, 52] = cur[15]
    features[:, 53] = c_all[:, 15]
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
    doc_arrays = build_doc_arrays(cache.token_data, cache.sent_offsets, start_gsi, end_gsi)
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

            if graph.get_cluster_id((origin_gsi, st)) >= 0:
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
                    cache.token_data, cache.sent_offsets, cur_abs, gsi_lo, gsi_hi, start_gsi, end_gsi,
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

                chain_progress = float(cur_abs - doc_start_abs) / max(doc_len - 1, 1)

                chain_deps = np.array(chain_deps_list, dtype=np.float32)
                chain_pos = np.array(chain_pos_list, dtype=np.float32)
                batch_features = build_features_batch(
                    cache.token_data, cache.sent_offsets, origin_abs, cur_abs, current_gsi,
                    top_cand_gsis, top_cand_tis,
                    hop, resolved_gender, resolved_number,
                    chain_deps, chain_pos,
                    len(cand_gsis), num_gm, num_propn,
                    graph_cluster_ids, graph_confidences, origin_cluster_id,
                    doc_start_abs,
                    doc_arrays["doc_len"],
                    doc_arrays["propn_type_first"],
                    doc_arrays["propn_type_freq"],
                    doc_arrays["sent_propn_counts"],
                    doc_arrays["doc_punct_mask"],
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
                    ranks_list.append(hop)

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


