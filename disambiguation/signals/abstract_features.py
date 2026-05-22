import json
from dataclasses import dataclass, field

import numpy as np
import spacy
import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import DebertaV2TokenizerFast

from disambiguation.parsing.loader import load_parser
from disambiguation.signals.embeddings import EmbeddingStore
from disambiguation.signals.extraction import ParsedDocument, parse_document
from disambiguation.signals.unified_data import UnifiedDocument

MAX_HOPS = 512
TOP_K_CANDIDATES = 20

POS_IDS = {"PROPN": 0, "NOUN": 1, "PRON": 2, "ADJ": 3, "VERB": 4, "DET": 5, "ADP": 6, "AUX": 7, "ADV": 8, "SCONJ": 9, "CCONJ": 10, "PART": 11, "NUM": 12, "PUNCT": 13, "X": 14}
DEP_IDS = {
    "nsubj": 0, "obj": 1, "obl": 2, "nmod": 3, "nmod:poss": 4, "appos": 5, "conj": 6,
    "compound": 7, "flat": 8, "det": 9, "amod": 10, "root": 11, "nsubj:pass": 12,
    "obl:agent": 13, "iobj": 14, "ccomp": 15, "xcomp": 16, "acl": 17, "acl:relcl": 18,
    "advcl": 19, "expl": 20, "cop": 21, "advmod": 22, "mark": 23, "aux": 24,
    "aux:pass": 25, "case": 26, "cc": 27, "punct": 28, "nummod": 29,
}
GENDER_IDS = {"Masc": 0, "Fem": 1, "Neut": 2, "unknown": 3}
NUMBER_IDS = {"Sing": 0, "Plur": 1, "unknown": 2}
PRONTYPE_IDS = {"Prs": 0, "Art": 1, "Dem": 2, "Rel": 3, "Int": 4, "unknown": 5}


@dataclass
class TokenInfo:
    pos: int
    dep: int
    gender: int
    number: int
    person: int
    prontype: int
    is_subject: int  # nsubj or nsubj:pass
    is_object: int  # obj or iobj
    is_possessive: int  # nmod:poss
    depth_to_root: int  # hops to root in dep tree
    n_children: int
    sent_position: float


@dataclass
class DocCache:
    parsed: ParsedDocument
    sentence_embeddings: np.ndarray
    sentences: list[list[str]]
    clusters: list[list[list[int]]]
    token_info: list[list[TokenInfo]]
    mention_embeddings: dict = field(default_factory=dict)  # (si, ti) -> np.ndarray for NOUN/PROPN
    sentence_lengths: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sentence_lengths:
            self.sentence_lengths = [len(s.tokens) for s in self.parsed.sentences]

    def abs_pos(self, sent_idx: int, token_idx: int) -> int:
        return sum(self.sentence_lengths[:sent_idx]) + token_idx


@dataclass
class Chain:
    origin_sent: int
    origin_token: int
    current_sent: int
    current_token: int
    hop_count: int
    visited: set = field(default_factory=set)
    resolved_gender: int = 3
    resolved_number: int = 2
    chain_deps: list[int] = field(default_factory=list)
    chain_pos: list[int] = field(default_factory=list)
    chain_noun_embs: list[np.ndarray] = field(default_factory=list)
    has_seen_propn: int = 0


def _compute_depth(spacy_token) -> int:
    depth = 0
    current = spacy_token
    while current.head != current:
        depth += 1
        current = current.head
        if depth > 20:
            break
    return depth


def _extract_token_info(spacy_doc) -> list[TokenInfo]:
    infos = []
    for token in spacy_doc:
        morph = token.morph
        dep = token.dep_
        infos.append(TokenInfo(
            pos=POS_IDS.get(token.pos_, len(POS_IDS)),
            dep=DEP_IDS.get(dep, len(DEP_IDS)),
            gender=GENDER_IDS.get(morph.get("Gender", ["unknown"])[0], 3),
            number=NUMBER_IDS.get(morph.get("Number", ["unknown"])[0], 2),
            person=int(morph.get("Person", ["0"])[0]),
            prontype=PRONTYPE_IDS.get(morph.get("PronType", ["unknown"])[0], 5),
            is_subject=int(dep in ("nsubj", "nsubj:pass", "nsubj:outer", "csubj")),
            is_object=int(dep in ("obj", "iobj")),
            is_possessive=int(dep in ("nmod:poss",)),
            depth_to_root=_compute_depth(token),
            n_children=len(list(token.children)),
            sent_position=token.i / max(len(spacy_doc) - 1, 1),
        ))
    return infos


def precompute(
    doc: UnifiedDocument,
    doc_idx: int,
    parser,
    tokenizer,
    config: dict,
    embed_store: EmbeddingStore,
    spacy_nlp,
    mention_embedder=None,
) -> DocCache:
    parsed_doc = parse_document(doc.sentences, parser, tokenizer, config, device="cuda")
    num_sents = len(doc.sentences)
    sentence_embeddings = embed_store.get_document_embeddings(doc_idx, num_sents)

    token_info = []
    for si, sent in enumerate(doc.sentences):
        text = " ".join(sent)
        spacy_doc = spacy_nlp(text)
        infos = _extract_token_info(spacy_doc)
        target_len = len(parsed_doc.sentences[si].tokens) if si < len(parsed_doc.sentences) else len(sent)
        while len(infos) < target_len:
            infos.append(TokenInfo(0, 0, 3, 2, 0, 5, 0, 0, 0, 0, 0, 0.0))
        token_info.append(infos[:target_len])

    # Compute mention embeddings for NOUN/PROPN
    mention_embeddings = {}
    if mention_embedder is not None:
        texts = []
        keys = []
        for si, sent in enumerate(doc.sentences):
            for ti in range(min(len(sent), len(token_info[si]))):
                if token_info[si][ti].pos in (POS_IDS["NOUN"], POS_IDS["PROPN"]):
                    texts.append(sent[ti])
                    keys.append((si, ti))
        if texts:
            embs = mention_embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            for key, emb in zip(keys, embs):
                mention_embeddings[key] = emb.astype(np.float32)

    return DocCache(
        parsed=parsed_doc,
        sentence_embeddings=sentence_embeddings,
        sentences=doc.sentences,
        clusters=doc.clusters,
        token_info=token_info,
        mention_embeddings=mention_embeddings,
    )


def build_features(doc: DocCache, chain: Chain, cand_sent: int, cand_token: int, rank: int, num_cands: int, num_gender_match: int, num_propn_cands: int, noun_sim_to_chain: float = 0.0) -> np.ndarray:
    origin_info = doc.token_info[chain.origin_sent][chain.origin_token]
    current_info = doc.token_info[chain.current_sent][chain.current_token]
    cand_info = doc.token_info[cand_sent][cand_token]

    current_abs = doc.abs_pos(chain.current_sent, chain.current_token)
    cand_abs = doc.abs_pos(cand_sent, cand_token)

    # Gender/number agreement
    gender_match_origin = int(origin_info.gender == 3 or cand_info.gender == 3 or origin_info.gender == cand_info.gender)
    number_match_origin = int(origin_info.number == 2 or cand_info.number == 2 or origin_info.number == cand_info.number)
    gender_match_current = int(current_info.gender == 3 or cand_info.gender == 3 or current_info.gender == cand_info.gender)
    number_match_current = int(current_info.number == 2 or cand_info.number == 2 or current_info.number == cand_info.number)
    resolved_gender_match = int(chain.resolved_gender == 3 or cand_info.gender == 3 or chain.resolved_gender == cand_info.gender)
    resolved_number_match = int(chain.resolved_number == 2 or cand_info.number == 2 or chain.resolved_number == cand_info.number)

    # Chain consistency
    dep_consistent = sum(1 for d in chain.chain_deps if d == cand_info.dep) / max(len(chain.chain_deps), 1)
    pos_consistent = sum(1 for p in chain.chain_pos if p == cand_info.pos) / max(len(chain.chain_pos), 1)

    # Terminal signal
    is_propn_after_pron_chain = int(cand_info.pos == POS_IDS["PROPN"] and origin_info.pos == POS_IDS["PRON"])

    return np.array([
        # Origin (6)
        origin_info.pos, origin_info.dep, origin_info.gender, origin_info.number,
        origin_info.is_subject, origin_info.is_possessive,
        # Current (6)
        current_info.pos, current_info.dep, current_info.gender, current_info.number,
        current_info.is_subject, current_info.is_possessive,
        # Candidate (8)
        cand_info.pos, cand_info.dep, cand_info.gender, cand_info.number,
        cand_info.person, cand_info.is_subject, cand_info.is_object, cand_info.is_possessive,
        # Structural relational (6)
        int(cand_info.dep == current_info.dep),
        int(cand_info.dep == origin_info.dep),
        int(cand_info.pos == current_info.pos),
        int(cand_info.is_subject == origin_info.is_subject),
        cand_info.depth_to_root,
        int(cand_sent == chain.current_sent),
        # Distance (3)
        abs(cand_abs - current_abs),
        abs(cand_sent - chain.current_sent),
        chain.hop_count,
        # Agreement (6)
        gender_match_origin, number_match_origin,
        gender_match_current, number_match_current,
        resolved_gender_match, resolved_number_match,
        # Chain consistency (3)
        dep_consistent, pos_consistent,
        is_propn_after_pron_chain,
        # Competition (4)
        rank, num_cands, num_gender_match, num_propn_cands,
        # Noun embedding similarity to chain (1)
        noun_sim_to_chain,
    ], dtype=np.float32)


NUM_FEATURES = 43

FEATURE_NAMES = [
    "o_pos", "o_dep", "o_gender", "o_number", "o_is_subj", "o_is_poss",
    "cur_pos", "cur_dep", "cur_gender", "cur_number", "cur_is_subj", "cur_is_poss",
    "c_pos", "c_dep", "c_gender", "c_number", "c_person", "c_is_subj", "c_is_obj", "c_is_poss",
    "same_dep_current", "same_dep_origin", "same_pos_current", "same_subj_origin",
    "c_depth_to_root", "same_sentence",
    "token_distance", "sent_distance", "hop_count",
    "gender_match_origin", "number_match_origin",
    "gender_match_current", "number_match_current",
    "resolved_gender_match", "resolved_number_match",
    "dep_consistent", "pos_consistent", "is_propn_terminal",
    "rank", "num_cands", "num_gender_match_cands", "num_propn_cands",
    "noun_sim_to_chain",
]


def generate_episodes(doc: DocCache, sent_idx: int, start_token: int, end_token: int, window_tokens: int = 150) -> list[tuple[np.ndarray, int]]:
    cluster_id = -1
    for ci, cluster in enumerate(doc.clusters):
        for mention in cluster:
            if mention[0] == sent_idx and mention[1] == start_token and mention[2] == end_token:
                cluster_id = ci
                break
        if cluster_id >= 0:
            break

    if cluster_id < 0:
        return []

    correct_positions = set()
    for mention in doc.clusters[cluster_id]:
        si, st, en = mention
        if si < len(doc.parsed.sentences):
            correct_positions.add((si, st))

    origin_info = doc.token_info[sent_idx][start_token]
    if origin_info.pos == POS_IDS["PROPN"]:
        return []

    origin_noun_embs = []
    if (sent_idx, start_token) in doc.mention_embeddings:
        origin_noun_embs.append(doc.mention_embeddings[(sent_idx, start_token)])

    chain = Chain(
        origin_sent=sent_idx,
        origin_token=start_token,
        current_sent=sent_idx,
        current_token=start_token,
        hop_count=0,
        visited={(sent_idx, start_token)},
        resolved_gender=origin_info.gender if origin_info.gender != 3 else 3,
        resolved_number=origin_info.number if origin_info.number != 2 else 2,
        chain_deps=[origin_info.dep],
        chain_pos=[origin_info.pos],
        chain_noun_embs=origin_noun_embs,
    )

    results = []

    for hop in range(MAX_HOPS):
        current_abs = doc.abs_pos(chain.current_sent, chain.current_token)

        # Gather candidates
        all_candidates = []
        for si in range(len(doc.parsed.sentences)):
            for ti in range(len(doc.token_info[si])):
                info = doc.token_info[si][ti]
                if info.pos not in (POS_IDS["NOUN"], POS_IDS["PROPN"], POS_IDS["PRON"]):
                    continue
                if (si, ti) in chain.visited:
                    continue
                abs_pos = doc.abs_pos(si, ti)
                if abs(abs_pos - current_abs) <= window_tokens:
                    all_candidates.append((si, ti, abs_pos))

        if not all_candidates:
            break

        # Use embedding to rank candidates (filter only, not a feature)
        origin_emb = doc.sentence_embeddings[chain.origin_sent].astype(np.float32)
        cand_sent_indices = [c[0] for c in all_candidates]
        cand_embs = doc.sentence_embeddings[cand_sent_indices].astype(np.float32)
        sims = cosine_similarity([origin_emb], cand_embs)[0]
        top_indices = np.argsort(-sims)[:TOP_K_CANDIDATES]

        # Competition features (computed once per state)
        num_propn_cands = sum(1 for si, ti, _ in all_candidates if doc.token_info[si][ti].pos == POS_IDS["PROPN"])
        num_gender_match = sum(
            1 for si, ti, _ in all_candidates
            if chain.resolved_gender == 3 or doc.token_info[si][ti].gender == 3 or doc.token_info[si][ti].gender == chain.resolved_gender
        )

        # Generate features for top-K
        for rank, idx in enumerate(top_indices):
            csi, cti, _ = all_candidates[idx]
            is_correct = (csi, cti) in correct_positions

            # Compute noun embedding similarity to chain (only for NOUN/PROPN candidates)
            noun_sim = 0.0
            cand_key = (csi, cti)
            if cand_key in doc.mention_embeddings and chain.chain_noun_embs:
                cand_emb = doc.mention_embeddings[cand_key]
                cand_norm = np.linalg.norm(cand_emb) + 1e-8
                best = 0.0
                for chain_emb in chain.chain_noun_embs:
                    s = float(np.dot(cand_emb, chain_emb) / (cand_norm * (np.linalg.norm(chain_emb) + 1e-8)))
                    if s > best:
                        best = s
                noun_sim = best

            features = build_features(doc, chain, csi, cti, rank, len(all_candidates), num_gender_match, num_propn_cands, noun_sim)
            results.append((features, 1 if is_correct else 0))

        # Teacher forcing
        correct_cands = [
            (idx, all_candidates[idx][2])
            for idx, (csi, cti, _) in enumerate(all_candidates)
            if (csi, cti) in correct_positions
        ]
        if not correct_cands:
            break

        best_idx = min(correct_cands, key=lambda x: abs(x[1] - current_abs))[0]
        next_si, next_ti, _ = all_candidates[best_idx]

        chain.visited.add((next_si, next_ti))
        chain.current_sent = next_si
        chain.current_token = next_ti
        chain.hop_count += 1

        next_info = doc.token_info[next_si][next_ti]
        chain.chain_deps.append(next_info.dep)
        chain.chain_pos.append(next_info.pos)

        # Add noun embedding to chain if NOUN/PROPN
        if (next_si, next_ti) in doc.mention_embeddings:
            chain.chain_noun_embs.append(doc.mention_embeddings[(next_si, next_ti)])

        if chain.resolved_gender == 3 and next_info.gender != 3:
            chain.resolved_gender = next_info.gender
        if chain.resolved_number == 2 and next_info.number != 2:
            chain.resolved_number = next_info.number

        if next_info.pos == POS_IDS["PROPN"]:
            break

    return results
