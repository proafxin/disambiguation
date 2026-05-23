import sys
from pathlib import Path

import numpy as np
import spacy
import xgboost as xgb

from disambiguation.signals.abstract_features import POS_IDS
from disambiguation.signals.resolution_graph import ResolutionGraph
from disambiguation.signals.train_full import (
    MAX_HOPS, TOP_K, build_doc_arrays, build_features_batch, _structural_candidates,
)
from scripts.run_spacy_cache import fill_doc_features, make_doc, split_into_chunks

MODELS_DIR = Path(__file__).parent.parent.parent / "cache" / "models"
N_TOKEN_FEATURES = 16
MAX_TOKENS = 4000


def _sentences_to_arrays(
    nlp: spacy.Language, sentences: list[list[str]]
) -> tuple[np.ndarray, np.ndarray]:
    sent_lens = [len(s) for s in sentences]
    n_tokens = sum(sent_lens)
    token_data = np.empty((n_tokens, N_TOKEN_FEATURES), dtype=np.float32)

    chunk_start = 0
    for chunk_tokens, chunk_sent_lens in split_into_chunks(sentences, MAX_TOKENS):
        doc = make_doc(nlp, chunk_tokens, chunk_sent_lens)
        fill_doc_features(doc, chunk_sent_lens, token_data, chunk_start)
        chunk_start += sum(chunk_sent_lens)

    sent_offsets = np.zeros(len(sentences) + 1, dtype=np.int64)
    sent_offsets[1:] = np.cumsum(sent_lens)
    return token_data, sent_offsets


def _resolve(
    token_data: np.ndarray,
    sent_offsets: np.ndarray,
    tokens: list[str],
    model: xgb.XGBClassifier,
    window_tokens: int,
) -> ResolutionGraph:
    n_sents = len(sent_offsets) - 1
    start_gsi = 0
    end_gsi = n_sents
    doc_start_abs = 0
    doc_len = int(sent_offsets[end_gsi])

    graph = ResolutionGraph()
    doc_arrays = build_doc_arrays(token_data, sent_offsets, start_gsi, end_gsi)

    propn_id = POS_IDS["PROPN"]
    noun_id = POS_IDS["NOUN"]
    pron_id = POS_IDS["PRON"]

    for gsi in range(n_sents):
        sl = int(sent_offsets[gsi + 1] - sent_offsets[gsi])
        for ti in range(sl):
            origin_abs = int(sent_offsets[gsi]) + ti
            origin_info = token_data[origin_abs]
            pos_id = int(origin_info[0])
            if pos_id not in (pron_id, noun_id):
                continue
            if graph.get_cluster_id((gsi, ti)) >= 0:
                continue

            origin_doc_pos = origin_abs / max(doc_len - 1, 1)
            cum = doc_arrays["pos_cumcounts"].get(pos_id)
            prior_same_pos = int(cum[origin_abs - 1]) if cum is not None and origin_abs > 0 else 0

            current_gsi = gsi
            current_ti = ti
            visited_abs = np.zeros(doc_len, dtype=bool)
            visited_abs[origin_abs] = True

            resolved_gender = int(origin_info[2]) if origin_info[2] != 3 else 3
            resolved_number = int(origin_info[3]) if origin_info[3] != 2 else 2
            chain_deps_list = [int(origin_info[1])]
            chain_pos_list = [int(origin_info[0])]

            for hop in range(MAX_HOPS):
                cur_abs = int(sent_offsets[current_gsi]) + current_ti
                gsi_lo = max(start_gsi, current_gsi - 20)
                gsi_hi = min(end_gsi, current_gsi + 20)

                graph_cluster_ids, graph_confidences = graph.build_abs_arrays(
                    sent_offsets, doc_start_abs, doc_len
                )
                origin_cluster_id = graph_cluster_ids[origin_abs] if origin_abs < doc_len else -1

                cand_gsis, cand_tis, valid_abs_arr = _structural_candidates(
                    token_data, sent_offsets, cur_abs, gsi_lo, gsi_hi, start_gsi, end_gsi,
                    window_tokens, visited_abs, doc_start_abs, doc_len,
                    resolved_gender, resolved_number, graph_cluster_ids,
                )
                if len(cand_gsis) == 0:
                    break

                distances = np.abs(valid_abs_arr - cur_abs)
                top_indices = np.argsort(distances)[:TOP_K]
                top_cand_gsis = cand_gsis[top_indices]
                top_cand_tis = cand_tis[top_indices]
                top_abs = valid_abs_arr[top_indices]

                all_pos = token_data[valid_abs_arr, 0]
                all_gender = token_data[valid_abs_arr, 2]
                num_propn = int(np.sum(all_pos == propn_id))
                num_gm = int(np.sum(
                    (all_gender == 3) | (all_gender == resolved_gender) | (resolved_gender == 3)
                ))

                chain_deps = np.array(chain_deps_list, dtype=np.float32)
                chain_pos = np.array(chain_pos_list, dtype=np.float32)
                chain_progress = float(cur_abs) / max(doc_len - 1, 1)

                batch_features = build_features_batch(
                    token_data, sent_offsets, origin_abs, cur_abs, current_gsi,
                    top_cand_gsis, top_cand_tis,
                    hop, resolved_gender, resolved_number,
                    chain_deps, chain_pos,
                    len(cand_gsis), num_gm, num_propn,
                    graph_cluster_ids, graph_confidences, origin_cluster_id,
                    doc_start_abs, doc_len,
                    doc_arrays["propn_type_first"], doc_arrays["propn_type_freq"],
                    doc_arrays["sent_propn_counts"], doc_arrays["doc_punct_mask"],
                    sent_offsets[start_gsi:end_gsi + 1],
                    origin_doc_pos, prior_same_pos, chain_progress, start_gsi,
                )

                scores = model.predict_proba(batch_features)[:, 1]
                best = int(np.argmax(scores))
                next_abs = int(top_abs[best])
                next_gsi = int(top_cand_gsis[best])
                next_ti = int(top_cand_tis[best])

                visited_abs[next_abs] = True
                current_gsi = next_gsi
                current_ti = next_ti

                next_info = token_data[next_abs]
                chain_deps_list.append(int(next_info[1]))
                chain_pos_list.append(int(next_info[0]))

                next_is_propn = bool(next_info[0] == propn_id)
                graph.link(
                    (gsi, ti), (next_gsi, next_ti),
                    confidence=float(scores[best]),
                    is_b_propn=next_is_propn,
                )

                if resolved_gender == 3 and next_info[2] != 3:
                    resolved_gender = int(next_info[2])
                if resolved_number == 2 and next_info[3] != 2:
                    resolved_number = int(next_info[3])

                if next_is_propn:
                    break

    return graph


def disambiguate(
    sentences: list[list[str]],
    model: xgb.XGBClassifier,
    nlp: spacy.Language,
    window_tokens: int = 150,
) -> dict[str, list[str]]:
    flat_tokens = [tok for sent in sentences for tok in sent]
    token_data, sent_offsets = _sentences_to_arrays(nlp, sentences)
    graph = _resolve(token_data, sent_offsets, flat_tokens, model, window_tokens)

    result: dict[str, list[str]] = {}
    for cid, members in graph.cluster_members.items():
        canonical = graph.cluster_to_canonical.get(cid)
        if canonical is None:
            continue
        canon_abs = int(sent_offsets[canonical[0]]) + canonical[1]
        result[flat_tokens[canon_abs]] = [
            flat_tokens[int(sent_offsets[m[0]]) + m[1]]
            for m in sorted(members, key=lambda m: int(sent_offsets[m[0]]) + m[1])
        ]
    return result


def load_model(model_key: str = "t3_unified", window: int = 150) -> xgb.XGBClassifier:
    path = MODELS_DIR / "full" / f"{model_key}_w{window}.ubj"
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def load_nlp() -> spacy.Language:
    spacy.prefer_gpu()
    return spacy.load("en_core_web_trf", disable=["senter", "lemmatizer"])


if __name__ == "__main__":
    raw = sys.stdin.read().strip() if not sys.stdin.isatty() else (
        "Elizabeth Warren announced her campaign .\n"
        "She said she would fight for working families .\n"
        "The senator from Massachusetts has long championed consumer protection ."
    )
    sentences = [line.split() for line in raw.strip().splitlines() if line.strip()]
    nlp = load_nlp()
    model = load_model()
    for canonical, mentions in disambiguate(sentences, model, nlp).items():
        print(f"{canonical}: {mentions}")
