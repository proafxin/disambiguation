import json
from dataclasses import dataclass, field

import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DebertaV2TokenizerFast

from disambiguation.parsing.loader import load_parser
from disambiguation.signals.embeddings import EmbeddingStore
from disambiguation.signals.extraction import ParsedDocument, parse_document


MAX_HOPS = 10


@dataclass
class PrecomputedDoc:
    parsed: ParsedDocument
    sentence_embeddings: np.ndarray  # [num_sentences, embed_dim]
    sentences: list[list[str]]
    clusters: list[list[list[int]]]
    sentence_lengths: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sentence_lengths:
            self.sentence_lengths = [len(s.tokens) for s in self.parsed.sentences]

    def abs_pos(self, sent_idx: int, token_idx: int) -> int:
        return sum(self.sentence_lengths[:sent_idx]) + token_idx


@dataclass
class EpisodeRecord:
    # State features
    origin_pos: int
    origin_dep: int
    origin_sent_idx: int
    current_pos: int
    current_dep: int
    current_sent_idx: int
    hop_count: int
    # Candidate features
    cand_pos: int
    cand_dep: int
    cand_sent_idx: int
    distance_tokens: int
    same_dep_as_current: int
    same_dep_as_origin: int
    same_pos_as_current: int
    dep_tree_overlap_current: int
    dep_tree_overlap_origin: int
    context_overlap_current: int
    context_overlap_origin: int
    # Embedding features (precomputed from sentence embeddings)
    embed_sim_to_origin: float
    embed_sim_to_current: float
    # Competition features
    rank_by_embed_sim_origin: int
    rank_by_embed_sim_current: int
    num_candidates: int
    num_propn_candidates: int
    sim_gap_to_next_best: float
    # Label
    reward: float


POS_MAP = {"PROPN": 0, "NOUN": 1, "PRON": 2, "ADJ": 3, "VERB": 4, "DET": 5, "ADP": 6, "AUX": 7, "X": 8}
DEP_MAP = {
    "nsubj": 0, "obj": 1, "obl": 2, "nmod": 3, "nmod:poss": 4, "appos": 5,
    "conj": 6, "compound": 7, "flat": 8, "det": 9, "amod": 10, "root": 11,
    "nsubj:pass": 12, "obl:agent": 13, "iobj": 14, "ccomp": 15, "xcomp": 16,
    "acl": 17, "acl:relcl": 18, "advcl": 19, "expl": 20, "cop": 21,
}


def encode_pos(pos: str) -> int:
    return POS_MAP.get(pos, len(POS_MAP))


def encode_dep(dep: str) -> int:
    return DEP_MAP.get(dep, len(DEP_MAP))



def precompute_document(
    sample: dict,
    doc_idx: int,
    parser,
    tokenizer,
    config: dict,
    embed_store,
) -> PrecomputedDoc:
    parsed_doc = parse_document(sample["sentences"], parser, tokenizer, config, device="cuda")
    num_sents = len(sample["sentences"])
    sentence_embeddings = embed_store.get_document_embeddings(doc_idx, num_sents)
    return PrecomputedDoc(
        parsed=parsed_doc,
        sentence_embeddings=sentence_embeddings,
        sentences=sample["sentences"],
        clusters=sample["mention_clusters"],
    )


def _get_nominal_candidates(doc: PrecomputedDoc, center_sent: int, window_tokens: int = 200) -> list[tuple[int, int]]:
    center_pos = doc.abs_pos(center_sent, 0)
    candidates = []
    for si, sent in enumerate(doc.parsed.sentences):
        for ti, tok in enumerate(sent.tokens):
            if tok.pos not in ("NOUN", "PROPN", "PRON"):
                continue
            abs_pos = doc.abs_pos(si, ti)
            if abs(abs_pos - center_pos) <= window_tokens:
                candidates.append((si, ti))
    return candidates


def _get_cluster_positions(cluster_id: int, doc: PrecomputedDoc) -> set[tuple[int, int]]:
    if cluster_id < 0:
        return set()
    positions = set()
    for mention in doc.clusters[cluster_id]:
        sent_idx, start, end = mention
        if sent_idx < len(doc.parsed.sentences):
            positions.add((sent_idx, start))
    return positions


def _find_cluster_id(sent_idx: int, start: int, end: int, clusters: list[list[list[int]]]) -> int:
    for ci, cluster in enumerate(clusters):
        for mention in cluster:
            if mention[0] == sent_idx and mention[1] == start and mention[2] == end:
                return ci
    return -1


def generate_episodes_for_mention(
    doc: PrecomputedDoc,
    sent_idx: int,
    start_token: int,
    end_token: int,
    window_tokens: int = 200,
) -> list[EpisodeRecord]:
    cluster_id = _find_cluster_id(sent_idx, start_token, end_token, doc.clusters)
    if cluster_id < 0:
        return []

    correct_positions = _get_cluster_positions(cluster_id, doc)
    origin_tok = doc.parsed.sentences[sent_idx].tokens[start_token]

    # Already canonical
    if origin_tok.pos == "PROPN":
        return []

    origin_sent_idx = sent_idx
    origin_token_idx = start_token
    current_sent_idx = sent_idx
    current_token_idx = start_token
    visited = {(sent_idx, start_token)}

    episodes = []

    for hop in range(MAX_HOPS):
        current_tok = doc.parsed.sentences[current_sent_idx].tokens[current_token_idx]
        origin_tok_ref = doc.parsed.sentences[origin_sent_idx].tokens[origin_token_idx]

        # Get candidates within window from current position
        current_abs = doc.abs_pos(current_sent_idx, current_token_idx)
        candidates = []
        for si, sent in enumerate(doc.parsed.sentences):
            for ti, tok in enumerate(sent.tokens):
                if tok.pos not in ("NOUN", "PROPN", "PRON"):
                    continue
                if (si, ti) in visited:
                    continue
                abs_pos = doc.abs_pos(si, ti)
                if abs(abs_pos - current_abs) <= window_tokens:
                    candidates.append((si, ti, tok, abs_pos))

        if not candidates:
            break

        # Compute embedding similarities using precomputed sentence embeddings
        origin_emb = doc.sentence_embeddings[origin_sent_idx]
        current_emb = doc.sentence_embeddings[current_sent_idx]
        cand_sent_indices = [c[0] for c in candidates]
        cand_embs = doc.sentence_embeddings[cand_sent_indices]

        sims_to_origin = cosine_similarity([origin_emb], cand_embs)[0]
        sims_to_current = cosine_similarity([current_emb], cand_embs)[0]

        # Rank candidates by similarity to origin
        rank_origin = np.argsort(-sims_to_origin)
        rank_current = np.argsort(-sims_to_current)

        rank_origin_map = {idx: rank for rank, idx in enumerate(rank_origin)}
        rank_current_map = {idx: rank for rank, idx in enumerate(rank_current)}

        sorted_sims_origin = np.sort(sims_to_origin)[::-1]
        num_propn = sum(1 for c in candidates if c[2].pos == "PROPN")

        # Generate episode for each candidate
        correct_idx = None
        for i, (csi, cti, ctok, cabs) in enumerate(candidates):
            is_correct = (csi, cti) in correct_positions

            # Sim gap: difference between this candidate's sim and the next best
            rank_pos = rank_origin_map[i]
            sim_gap = 0.0
            if rank_pos < len(sorted_sims_origin) - 1:
                sim_gap = sorted_sims_origin[rank_pos] - sorted_sims_origin[rank_pos + 1]

            ep = EpisodeRecord(
                origin_pos=encode_pos(origin_tok_ref.pos),
                origin_dep=encode_dep(origin_tok_ref.dep_rel),
                origin_sent_idx=origin_sent_idx,
                current_pos=encode_pos(current_tok.pos),
                current_dep=encode_dep(current_tok.dep_rel),
                current_sent_idx=current_sent_idx,
                hop_count=hop,
                cand_pos=encode_pos(ctok.pos),
                cand_dep=encode_dep(ctok.dep_rel),
                cand_sent_idx=csi,
                distance_tokens=abs(cabs - current_abs),
                same_dep_as_current=int(ctok.dep_rel == current_tok.dep_rel),
                same_dep_as_origin=int(ctok.dep_rel == origin_tok_ref.dep_rel),
                same_pos_as_current=int(ctok.pos == current_tok.pos),
                dep_tree_overlap_current=len(
                    set(t.text for t in doc.parsed.sentences[csi].tokens if t.head_idx == cti or cti == t.idx_in_sent)
                    & set(t.text for t in doc.parsed.sentences[current_sent_idx].tokens if t.head_idx == current_token_idx or current_token_idx == t.idx_in_sent)
                ),
                dep_tree_overlap_origin=len(
                    set(t.text for t in doc.parsed.sentences[csi].tokens if t.head_idx == cti or cti == t.idx_in_sent)
                    & set(t.text for t in doc.parsed.sentences[origin_sent_idx].tokens if t.head_idx == origin_token_idx or origin_token_idx == t.idx_in_sent)
                ),
                context_overlap_current=0,  # TODO: add if needed
                context_overlap_origin=0,
                embed_sim_to_origin=float(sims_to_origin[i]),
                embed_sim_to_current=float(sims_to_current[i]),
                rank_by_embed_sim_origin=rank_origin_map[i],
                rank_by_embed_sim_current=rank_current_map[i],
                num_candidates=len(candidates),
                num_propn_candidates=num_propn,
                sim_gap_to_next_best=float(sim_gap),
                reward=1.0 if is_correct else -1.0,
            )
            episodes.append(ep)

            if is_correct and correct_idx is None:
                correct_idx = i

        # Teacher forcing: move to nearest correct candidate
        if correct_idx is None:
            break

        correct_cands = [(i, candidates[i][3]) for i, (csi, cti, _, _) in enumerate(candidates) if (csi, cti) in correct_positions]
        if not correct_cands:
            break
        best_i = min(correct_cands, key=lambda x: abs(x[1] - current_abs))[0]
        next_si, next_ti, next_tok, _ = candidates[best_i]
        visited.add((next_si, next_ti))
        current_sent_idx = next_si
        current_token_idx = next_ti

        if next_tok.pos == "PROPN":
            break

    return episodes


def episode_record_to_vector(ep: EpisodeRecord) -> np.ndarray:
    return np.array([
        ep.origin_pos, ep.origin_dep, ep.current_pos, ep.current_dep,
        ep.hop_count, ep.cand_pos, ep.cand_dep, ep.cand_sent_idx,
        ep.distance_tokens, ep.same_dep_as_current, ep.same_dep_as_origin,
        ep.same_pos_as_current, ep.dep_tree_overlap_current, ep.dep_tree_overlap_origin,
        ep.context_overlap_current, ep.context_overlap_origin,
        ep.embed_sim_to_origin, ep.embed_sim_to_current,
        ep.rank_by_embed_sim_origin, ep.rank_by_embed_sim_current,
        ep.num_candidates, ep.num_propn_candidates, ep.sim_gap_to_next_best,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "origin_pos", "origin_dep", "current_pos", "current_dep",
    "hop_count", "cand_pos", "cand_dep", "cand_sent_idx",
    "distance_tokens", "same_dep_as_current", "same_dep_as_origin",
    "same_pos_as_current", "dep_tree_overlap_current", "dep_tree_overlap_origin",
    "context_overlap_current", "context_overlap_origin",
    "embed_sim_to_origin", "embed_sim_to_current",
    "rank_by_embed_sim_origin", "rank_by_embed_sim_current",
    "num_candidates", "num_propn_candidates", "sim_gap_to_next_best",
]


def process_documents(num_docs: int = 100, window_tokens: int = 200) -> tuple[np.ndarray, np.ndarray]:
    parser = load_parser(device="cuda", dtype=torch.float16)
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    config_path = hf_hub_download(
        repo_id="ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt", filename="config.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    embed_store = EmbeddingStore.load()
    ds = load_from_disk("data/preco")

    all_vectors = []
    all_labels = []

    for doc_idx in range(num_docs):
        sample = ds["train"][doc_idx]
        doc = precompute_document(sample, doc_idx, parser, tokenizer, config, embed_store)

        for cluster in doc.clusters:
            if len(cluster) < 2:
                continue
            for mention in cluster:
                sent_idx, start, end = mention
                if sent_idx >= len(doc.parsed.sentences):
                    continue
                if end > len(doc.parsed.sentences[sent_idx].tokens):
                    continue
                tok = doc.parsed.sentences[sent_idx].tokens[start]
                if tok.pos == "PROPN":
                    continue

                episodes = generate_episodes_for_mention(doc, sent_idx, start, end, window_tokens)
                for ep in episodes:
                    all_vectors.append(episode_record_to_vector(ep))
                    all_labels.append(1 if ep.reward > 0 else 0)

        if (doc_idx + 1) % 10 == 0:
            print(f"  Processed {doc_idx + 1}/{num_docs} docs, {len(all_vectors)} episodes")

    return np.array(all_vectors), np.array(all_labels, dtype=np.int32)
