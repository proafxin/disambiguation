import json
from dataclasses import dataclass, field

import numpy as np
import spacy
import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DebertaV2TokenizerFast

from disambiguation.parsing.loader import load_parser
from disambiguation.signals.embeddings import EmbeddingStore
from disambiguation.signals.extraction import ParsedDocument, parse_document

MAX_HOPS = 10

DISCOURSE_MARKERS = frozenset({
    "however", "meanwhile", "moreover", "furthermore", "nevertheless",
    "therefore", "consequently", "instead", "otherwise", "similarly",
    "additionally", "finally", "subsequently", "conversely", "nonetheless",
})

POS_MAP = {"PROPN": 0, "NOUN": 1, "PRON": 2, "ADJ": 3, "VERB": 4, "DET": 5, "ADP": 6, "AUX": 7, "X": 8}
DEP_MAP = {
    "nsubj": 0, "obj": 1, "obl": 2, "nmod": 3, "nmod:poss": 4, "appos": 5,
    "conj": 6, "compound": 7, "flat": 8, "det": 9, "amod": 10, "root": 11,
    "nsubj:pass": 12, "obl:agent": 13, "iobj": 14, "ccomp": 15, "xcomp": 16,
    "acl": 17, "acl:relcl": 18, "advcl": 19, "expl": 20, "cop": 21,
}
GENDER_MAP = {"Masc": 0, "Fem": 1, "Neut": 2, "unknown": 3}
NUMBER_MAP = {"Sing": 0, "Plur": 1, "unknown": 2}
MENTION_TYPE_MAP = {"definite": 0, "indefinite": 1, "pronoun": 2, "proper": 3, "bare": 4}


def encode_pos(pos: str) -> int:
    return POS_MAP.get(pos, len(POS_MAP))


def encode_dep(dep: str) -> int:
    return DEP_MAP.get(dep, len(DEP_MAP))


@dataclass
class PrecomputedDoc:
    parsed: ParsedDocument
    sentence_embeddings: np.ndarray
    sentences: list[list[str]]
    clusters: list[list[list[int]]]
    morph_data: list[list[dict]]  # [sent_idx][token_idx] -> morph dict
    sentence_lengths: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sentence_lengths:
            self.sentence_lengths = [len(s.tokens) for s in self.parsed.sentences]

    def abs_pos(self, sent_idx: int, token_idx: int) -> int:
        return sum(self.sentence_lengths[:sent_idx]) + token_idx


def _extract_morph(nlp_doc) -> list[dict]:
    morphs = []
    for token in nlp_doc:
        morph = {}
        morph["gender"] = token.morph.get("Gender", ["unknown"])[0]
        morph["number"] = token.morph.get("Number", ["unknown"])[0]
        morph["person"] = token.morph.get("Person", ["0"])[0]
        morph["definite"] = token.morph.get("Definite", ["unknown"])[0]
        morphs.append(morph)
    return morphs


def _get_mention_type(tokens: list[str], pos: str) -> int:
    if pos == "PRON":
        return MENTION_TYPE_MAP["pronoun"]
    if pos == "PROPN":
        return MENTION_TYPE_MAP["proper"]
    first = tokens[0].lower() if tokens else ""
    if first in ("the", "this", "that", "these", "those"):
        return MENTION_TYPE_MAP["definite"]
    if first in ("a", "an", "some", "any"):
        return MENTION_TYPE_MAP["indefinite"]
    return MENTION_TYPE_MAP["bare"]


def _get_head_verb(sent_tokens, token_idx: int) -> str:
    tok = sent_tokens[token_idx]
    # Walk up the dep tree to find the governing verb
    visited = set()
    current = tok
    while current.head_idx >= 0 and current.head_idx not in visited:
        visited.add(current.idx_in_sent)
        head = sent_tokens[current.head_idx]
        if head.pos in ("VERB", "AUX"):
            return head.text
        current = head
    return ""


def _get_dep_path_to_root(sent_tokens, token_idx: int) -> list[str]:
    path = []
    visited = set()
    current = sent_tokens[token_idx]
    while current.head_idx >= 0 and current.head_idx not in visited:
        visited.add(current.idx_in_sent)
        path.append(current.dep_rel)
        current = sent_tokens[current.head_idx]
    return path


def _has_discourse_marker_between(doc: PrecomputedDoc, sent_a: int, sent_b: int) -> int:
    low, high = min(sent_a, sent_b), max(sent_a, sent_b)
    for si in range(low + 1, high + 1):
        if si >= len(doc.sentences):
            break
        sent = doc.sentences[si]
        if sent and sent[0].lower() in DISCOURSE_MARKERS:
            return 1
    return 0


def precompute_document(
    sample: dict,
    doc_idx: int,
    parser,
    tokenizer,
    config: dict,
    embed_store: EmbeddingStore,
    spacy_nlp,
) -> PrecomputedDoc:
    parsed_doc = parse_document(sample["sentences"], parser, tokenizer, config, device="cuda")
    num_sents = len(sample["sentences"])
    sentence_embeddings = embed_store.get_document_embeddings(doc_idx, num_sents)

    # Get morph features from spacy
    morph_data = []
    for sent in sample["sentences"]:
        text = " ".join(sent)
        spacy_doc = spacy_nlp(text)
        morphs = _extract_morph(spacy_doc)
        # Align spacy tokens to our tokens (may differ in tokenization)
        # Simple approach: pad/truncate to match our token count
        while len(morphs) < len(sent):
            morphs.append({"gender": "unknown", "number": "unknown", "person": "0", "definite": "unknown"})
        morph_data.append(morphs[:len(sent)])

    return PrecomputedDoc(
        parsed=parsed_doc,
        sentence_embeddings=sentence_embeddings,
        sentences=sample["sentences"],
        clusters=sample["mention_clusters"],
        morph_data=morph_data,
    )


def _find_cluster_id(sent_idx: int, start: int, end: int, clusters: list[list[list[int]]]) -> int:
    for ci, cluster in enumerate(clusters):
        for mention in cluster:
            if mention[0] == sent_idx and mention[1] == start and mention[2] == end:
                return ci
    return -1


def _get_cluster_positions(cluster_id: int, doc: PrecomputedDoc) -> set[tuple[int, int]]:
    if cluster_id < 0:
        return set()
    positions = set()
    for mention in doc.clusters[cluster_id]:
        sent_idx, start, end = mention
        if sent_idx < len(doc.parsed.sentences):
            positions.add((sent_idx, start))
    return positions


def generate_episodes_for_mention(
    doc: PrecomputedDoc,
    sent_idx: int,
    start_token: int,
    end_token: int,
    window_tokens: int = 200,
) -> list[np.ndarray]:
    cluster_id = _find_cluster_id(sent_idx, start_token, end_token, doc.clusters)
    if cluster_id < 0:
        return []

    correct_positions = _get_cluster_positions(cluster_id, doc)
    origin_tok = doc.parsed.sentences[sent_idx].tokens[start_token]

    if origin_tok.pos == "PROPN":
        return []

    origin_sent_idx = sent_idx
    origin_token_idx = start_token
    current_sent_idx = sent_idx
    current_token_idx = start_token
    visited = {(sent_idx, start_token)}

    # Origin features (computed once)
    origin_morph = doc.morph_data[origin_sent_idx][origin_token_idx]
    origin_gender = GENDER_MAP.get(origin_morph["gender"], 3)
    origin_number = NUMBER_MAP.get(origin_morph["number"], 2)
    origin_verb = _get_head_verb(doc.parsed.sentences[origin_sent_idx].tokens, origin_token_idx)
    origin_dep_path = _get_dep_path_to_root(doc.parsed.sentences[origin_sent_idx].tokens, origin_token_idx)
    origin_mention_type = _get_mention_type(
        doc.sentences[origin_sent_idx][start_token:end_token], origin_tok.pos
    )

    all_vectors = []

    for hop in range(MAX_HOPS):
        current_tok = doc.parsed.sentences[current_sent_idx].tokens[current_token_idx]
        current_verb = _get_head_verb(doc.parsed.sentences[current_sent_idx].tokens, current_token_idx)
        current_dep_path = _get_dep_path_to_root(doc.parsed.sentences[current_sent_idx].tokens, current_token_idx)
        current_morph = doc.morph_data[current_sent_idx][current_token_idx]

        # Get candidates
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

        # Embedding similarities
        origin_emb = doc.sentence_embeddings[origin_sent_idx]
        current_emb = doc.sentence_embeddings[current_sent_idx]
        cand_sent_indices = [c[0] for c in candidates]
        cand_embs = doc.sentence_embeddings[cand_sent_indices]

        sims_to_origin = cosine_similarity([origin_emb], cand_embs)[0]
        sims_to_current = cosine_similarity([current_emb], cand_embs)[0]

        # Rankings
        rank_origin = np.argsort(-sims_to_origin)
        rank_current = np.argsort(-sims_to_current)
        rank_origin_map = {idx: rank for rank, idx in enumerate(rank_origin)}
        rank_current_map = {idx: rank for rank, idx in enumerate(rank_current)}

        sorted_sims = np.sort(sims_to_origin)[::-1]
        num_propn = sum(1 for c in candidates if c[2].pos == "PROPN")

        # Competition: how many candidates have sim > 0.8 of top sim
        top_sim = sorted_sims[0] if len(sorted_sims) > 0 else 0
        num_competing = int(np.sum(sims_to_origin > top_sim * 0.8))

        # Generate features for each candidate
        correct_idx = None
        for i, (csi, cti, ctok, cabs) in enumerate(candidates):
            is_correct = (csi, cti) in correct_positions

            cand_morph = doc.morph_data[csi][cti] if cti < len(doc.morph_data[csi]) else {"gender": "unknown", "number": "unknown", "person": "0", "definite": "unknown"}
            cand_verb = _get_head_verb(doc.parsed.sentences[csi].tokens, cti)
            cand_dep_path = _get_dep_path_to_root(doc.parsed.sentences[csi].tokens, cti)
            cand_mention_type = _get_mention_type([ctok.text], ctok.pos)

            # Morph agreement
            gender_match = int(
                origin_morph["gender"] == "unknown"
                or cand_morph["gender"] == "unknown"
                or origin_morph["gender"] == cand_morph["gender"]
            )
            number_match = int(
                origin_morph["number"] == "unknown"
                or cand_morph["number"] == "unknown"
                or origin_morph["number"] == cand_morph["number"]
            )

            # Verb association
            same_verb_as_origin = int(origin_verb != "" and origin_verb == cand_verb)
            same_verb_as_current = int(current_verb != "" and current_verb == cand_verb)

            # Dep path overlap
            dep_path_overlap_origin = len(set(origin_dep_path) & set(cand_dep_path))
            dep_path_overlap_current = len(set(current_dep_path) & set(cand_dep_path))

            # Discourse marker between current and candidate
            discourse_marker = _has_discourse_marker_between(doc, current_sent_idx, csi)

            # Dep tree overlap (tokens sharing head)
            dep_tree_overlap_current = len(
                set(t.text for t in doc.parsed.sentences[csi].tokens if t.head_idx == cti or cti == t.idx_in_sent)
                & set(t.text for t in doc.parsed.sentences[current_sent_idx].tokens if t.head_idx == current_token_idx or current_token_idx == t.idx_in_sent)
            )
            dep_tree_overlap_origin = len(
                set(t.text for t in doc.parsed.sentences[csi].tokens if t.head_idx == cti or cti == t.idx_in_sent)
                & set(t.text for t in doc.parsed.sentences[origin_sent_idx].tokens if t.head_idx == origin_token_idx or origin_token_idx == t.idx_in_sent)
            )

            # Sim gap
            rank_pos = rank_origin_map[i]
            sim_gap = 0.0
            if rank_pos < len(sorted_sims) - 1:
                sim_gap = sorted_sims[rank_pos] - sorted_sims[rank_pos + 1]

            vec = np.array([
                # Origin features
                encode_pos(origin_tok.pos), encode_dep(origin_tok.dep_rel),
                origin_gender, origin_number, origin_mention_type,
                # Current features
                encode_pos(current_tok.pos), encode_dep(current_tok.dep_rel),
                hop,
                # Candidate features
                encode_pos(ctok.pos), encode_dep(ctok.dep_rel),
                GENDER_MAP.get(cand_morph["gender"], 3),
                NUMBER_MAP.get(cand_morph["number"], 2),
                cand_mention_type,
                # Relational features
                abs(cabs - current_abs),  # distance
                int(ctok.dep_rel == current_tok.dep_rel),  # same dep as current
                int(ctok.dep_rel == origin_tok.dep_rel),  # same dep as origin
                int(ctok.pos == current_tok.pos),  # same pos as current
                gender_match, number_match,
                same_verb_as_origin, same_verb_as_current,
                dep_path_overlap_origin, dep_path_overlap_current,
                dep_tree_overlap_origin, dep_tree_overlap_current,
                # Discourse
                discourse_marker,
                abs(csi - current_sent_idx),  # sentence distance
                # Embedding
                float(sims_to_origin[i]), float(sims_to_current[i]),
                # Competition
                rank_origin_map[i], rank_current_map[i],
                len(candidates), num_propn, num_competing,
                float(sim_gap),
            ], dtype=np.float32)

            label = 1.0 if is_correct else -1.0
            all_vectors.append(np.append(vec, label))

            if is_correct and correct_idx is None:
                correct_idx = i

        # Teacher forcing
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

    return all_vectors


FEATURE_NAMES = [
    "origin_pos", "origin_dep", "origin_gender", "origin_number", "origin_mention_type",
    "current_pos", "current_dep", "hop_count",
    "cand_pos", "cand_dep", "cand_gender", "cand_number", "cand_mention_type",
    "distance_tokens", "same_dep_as_current", "same_dep_as_origin", "same_pos_as_current",
    "gender_match", "number_match",
    "same_verb_as_origin", "same_verb_as_current",
    "dep_path_overlap_origin", "dep_path_overlap_current",
    "dep_tree_overlap_origin", "dep_tree_overlap_current",
    "discourse_marker", "sentence_distance",
    "embed_sim_to_origin", "embed_sim_to_current",
    "rank_by_embed_sim_origin", "rank_by_embed_sim_current",
    "num_candidates", "num_propn_candidates", "num_competing",
    "sim_gap_to_next_best",
]

NUM_FEATURES = len(FEATURE_NAMES)


def process_documents(num_docs: int = 100, window_tokens: int = 200) -> tuple[np.ndarray, np.ndarray]:
    parser = load_parser(device="cuda", dtype=torch.float16)
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    config_path = hf_hub_download(
        repo_id="ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt", filename="config.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    embed_store = EmbeddingStore.load()
    spacy_nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer"])
    ds = load_from_disk("data/preco")

    all_vectors = []

    for doc_idx in range(num_docs):
        sample = ds["train"][doc_idx]
        doc = precompute_document(sample, doc_idx, parser, tokenizer, config, embed_store, spacy_nlp)

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

                vecs = generate_episodes_for_mention(doc, sent_idx, start, end, window_tokens)
                all_vectors.extend(vecs)

        if (doc_idx + 1) % 10 == 0:
            print(f"  Processed {doc_idx + 1}/{num_docs} docs, {len(all_vectors)} episodes")

    data = np.array(all_vectors)
    X = data[:, :NUM_FEATURES]
    y = (data[:, NUM_FEATURES] > 0).astype(np.int32)
    return X, y
