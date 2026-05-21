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

MAX_HOPS = 20

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


def _encode_pos(pos: str) -> int:
    return POS_MAP.get(pos, len(POS_MAP))


def _encode_dep(dep: str) -> int:
    return DEP_MAP.get(dep, len(DEP_MAP))


@dataclass
class DocData:
    parsed: ParsedDocument
    sentence_embeddings: np.ndarray
    sentences: list[list[str]]
    coref_chains: list[list[list[int]]]
    morph_data: list[list[dict]]
    sentence_lengths: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sentence_lengths:
            self.sentence_lengths = [len(s.tokens) for s in self.parsed.sentences]

    def abs_pos(self, sent_idx: int, token_idx: int) -> int:
        return sum(self.sentence_lengths[:sent_idx]) + token_idx


@dataclass
class PathState:
    origin_sent: int
    origin_token: int
    current_sent: int
    current_token: int
    hop_count: int
    visited: set = field(default_factory=set)
    # Path history: list of (sent_idx, token_idx, pos, dep_rel) for each hop
    path_history: list[tuple] = field(default_factory=list)
    # Running signals accumulated along the path
    path_dep_roles: list[str] = field(default_factory=list)
    path_pos_tags: list[str] = field(default_factory=list)


def _extract_morph(spacy_nlp, text: str) -> list[dict]:
    doc = spacy_nlp(text)
    morphs = []
    for token in doc:
        morphs.append({
            "gender": token.morph.get("Gender", ["unknown"])[0],
            "number": token.morph.get("Number", ["unknown"])[0],
            "person": token.morph.get("Person", ["0"])[0],
        })
    return morphs


def _get_head_verb(sent_tokens, token_idx: int) -> str:
    visited = set()
    current = sent_tokens[token_idx]
    while current.head_idx >= 0 and current.head_idx not in visited:
        visited.add(current.idx_in_sent)
        head = sent_tokens[current.head_idx]
        if head.pos in ("VERB", "AUX"):
            return head.text
        current = head
    return ""


def _get_dep_path(sent_tokens, token_idx: int) -> list[str]:
    path = []
    visited = set()
    current = sent_tokens[token_idx]
    while current.head_idx >= 0 and current.head_idx not in visited:
        visited.add(current.idx_in_sent)
        path.append(current.dep_rel)
        current = sent_tokens[current.head_idx]
    return path


def _get_mention_type(token_text: str, pos: str, sent_tokens: list, token_idx: int) -> int:
    if pos == "PRON":
        return MENTION_TYPE_MAP["pronoun"]
    if pos == "PROPN":
        return MENTION_TYPE_MAP["proper"]
    # Check preceding token for determiner
    if token_idx > 0:
        prev = sent_tokens[token_idx - 1].text.lower()
        if prev in ("the", "this", "that", "these", "those"):
            return MENTION_TYPE_MAP["definite"]
        if prev in ("a", "an", "some"):
            return MENTION_TYPE_MAP["indefinite"]
    return MENTION_TYPE_MAP["bare"]


def precompute_litbank_doc(
    sample: dict,
    doc_idx: int,
    parser,
    tokenizer,
    config: dict,
    embed_store: EmbeddingStore,
    spacy_nlp,
) -> DocData:
    parsed_doc = parse_document(sample["sentences"], parser, tokenizer, config, device="cuda")
    num_sents = len(sample["sentences"])
    sentence_embeddings = embed_store.get_document_embeddings(doc_idx, num_sents)

    morph_data = []
    for sent in sample["sentences"]:
        text = " ".join(sent)
        morphs = _extract_morph(spacy_nlp, text)
        while len(morphs) < len(sent):
            morphs.append({"gender": "unknown", "number": "unknown", "person": "0"})
        morph_data.append(morphs[:len(sent)])

    return DocData(
        parsed=parsed_doc,
        sentence_embeddings=sentence_embeddings,
        sentences=sample["sentences"],
        coref_chains=sample["coref_chains"],
        morph_data=morph_data,
    )


def _find_chain_id(sent_idx: int, start: int, end: int, chains: list[list[list[int]]]) -> int:
    for ci, chain in enumerate(chains):
        for mention in chain:
            # LitBank: inclusive end
            if mention[0] == sent_idx and mention[1] == start and mention[2] == end:
                return ci
    return -1


def _get_chain_positions(chain_id: int, doc: DocData) -> set[tuple[int, int]]:
    if chain_id < 0:
        return set()
    positions = set()
    for mention in doc.coref_chains[chain_id]:
        si, st, en = mention
        if si < len(doc.parsed.sentences):
            positions.add((si, st))
    return positions


def build_episode_features(
    doc: DocData,
    state: PathState,
    cand_sent: int,
    cand_token: int,
) -> np.ndarray:
    origin_tok = doc.parsed.sentences[state.origin_sent].tokens[state.origin_token]
    current_tok = doc.parsed.sentences[state.current_sent].tokens[state.current_token]
    cand_tok = doc.parsed.sentences[cand_sent].tokens[cand_token]

    origin_morph = doc.morph_data[state.origin_sent][state.origin_token]
    current_morph = doc.morph_data[state.current_sent][state.current_token]
    cand_morph = doc.morph_data[cand_sent][cand_token] if cand_token < len(doc.morph_data[cand_sent]) else {"gender": "unknown", "number": "unknown", "person": "0"}

    # Verb associations
    origin_verb = _get_head_verb(doc.parsed.sentences[state.origin_sent].tokens, state.origin_token)
    current_verb = _get_head_verb(doc.parsed.sentences[state.current_sent].tokens, state.current_token)
    cand_verb = _get_head_verb(doc.parsed.sentences[cand_sent].tokens, cand_token)

    # Dep paths
    origin_dep_path = _get_dep_path(doc.parsed.sentences[state.origin_sent].tokens, state.origin_token)
    cand_dep_path = _get_dep_path(doc.parsed.sentences[cand_sent].tokens, cand_token)
    current_dep_path = _get_dep_path(doc.parsed.sentences[state.current_sent].tokens, state.current_token)

    # Mention types
    origin_mtype = _get_mention_type(origin_tok.text, origin_tok.pos, doc.parsed.sentences[state.origin_sent].tokens, state.origin_token)
    cand_mtype = _get_mention_type(cand_tok.text, cand_tok.pos, doc.parsed.sentences[cand_sent].tokens, cand_token)

    # Distances
    current_abs = doc.abs_pos(state.current_sent, state.current_token)
    cand_abs = doc.abs_pos(cand_sent, cand_token)
    origin_abs = doc.abs_pos(state.origin_sent, state.origin_token)

    # Morph agreement
    gender_match_origin = int(origin_morph["gender"] == "unknown" or cand_morph["gender"] == "unknown" or origin_morph["gender"] == cand_morph["gender"])
    number_match_origin = int(origin_morph["number"] == "unknown" or cand_morph["number"] == "unknown" or origin_morph["number"] == cand_morph["number"])
    gender_match_current = int(current_morph["gender"] == "unknown" or cand_morph["gender"] == "unknown" or current_morph["gender"] == cand_morph["gender"])
    number_match_current = int(current_morph["number"] == "unknown" or cand_morph["number"] == "unknown" or current_morph["number"] == cand_morph["number"])

    # Dep tree overlap
    cand_tree = set(t.text for t in doc.parsed.sentences[cand_sent].tokens if t.head_idx == cand_token or cand_token == t.idx_in_sent)
    origin_tree = set(t.text for t in doc.parsed.sentences[state.origin_sent].tokens if t.head_idx == state.origin_token or state.origin_token == t.idx_in_sent)
    current_tree = set(t.text for t in doc.parsed.sentences[state.current_sent].tokens if t.head_idx == state.current_token or state.current_token == t.idx_in_sent)

    # Path-dependent features
    path_dep_consistency = 0
    if state.path_dep_roles:
        path_dep_consistency = sum(1 for d in state.path_dep_roles if d == cand_tok.dep_rel) / len(state.path_dep_roles)

    path_pos_consistency = 0
    if state.path_pos_tags:
        path_pos_consistency = sum(1 for p in state.path_pos_tags if p == cand_tok.pos) / len(state.path_pos_tags)

    # Embedding: similarity trend (is this hop getting closer to origin?)
    origin_emb = doc.sentence_embeddings[state.origin_sent]
    current_emb = doc.sentence_embeddings[state.current_sent]
    cand_emb = doc.sentence_embeddings[cand_sent]

    sim_cand_to_origin = float(cosine_similarity([cand_emb], [origin_emb])[0, 0])
    sim_cand_to_current = float(cosine_similarity([cand_emb], [current_emb])[0, 0])
    sim_current_to_origin = float(cosine_similarity([current_emb], [origin_emb])[0, 0])
    # Progress: is candidate closer to origin than current is?
    embed_progress = sim_cand_to_origin - sim_current_to_origin

    return np.array([
        # Origin (5)
        _encode_pos(origin_tok.pos), _encode_dep(origin_tok.dep_rel),
        GENDER_MAP.get(origin_morph["gender"], 3), NUMBER_MAP.get(origin_morph["number"], 2),
        origin_mtype,
        # Current (3)
        _encode_pos(current_tok.pos), _encode_dep(current_tok.dep_rel),
        state.hop_count,
        # Candidate (5)
        _encode_pos(cand_tok.pos), _encode_dep(cand_tok.dep_rel),
        GENDER_MAP.get(cand_morph["gender"], 3), NUMBER_MAP.get(cand_morph["number"], 2),
        cand_mtype,
        # Distances (3)
        abs(cand_abs - current_abs),
        abs(cand_abs - origin_abs),
        abs(cand_sent - state.current_sent),
        # Dep/structural (7)
        int(cand_tok.dep_rel == current_tok.dep_rel),
        int(cand_tok.dep_rel == origin_tok.dep_rel),
        int(cand_tok.pos == current_tok.pos),
        len(cand_tree & origin_tree),
        len(cand_tree & current_tree),
        len(set(cand_dep_path) & set(origin_dep_path)),
        len(set(cand_dep_path) & set(current_dep_path)),
        # Morph agreement (4)
        gender_match_origin, number_match_origin,
        gender_match_current, number_match_current,
        # Verb association (2)
        int(origin_verb != "" and origin_verb == cand_verb),
        int(current_verb != "" and current_verb == cand_verb),
        # Embedding (4)
        sim_cand_to_origin, sim_cand_to_current,
        sim_current_to_origin, embed_progress,
        # Path-dependent (2)
        path_dep_consistency, path_pos_consistency,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "origin_pos", "origin_dep", "origin_gender", "origin_number", "origin_mtype",
    "current_pos", "current_dep", "hop_count",
    "cand_pos", "cand_dep", "cand_gender", "cand_number", "cand_mtype",
    "dist_to_current", "dist_to_origin", "sent_dist_to_current",
    "same_dep_current", "same_dep_origin", "same_pos_current",
    "dep_tree_overlap_origin", "dep_tree_overlap_current",
    "dep_path_overlap_origin", "dep_path_overlap_current",
    "gender_match_origin", "number_match_origin",
    "gender_match_current", "number_match_current",
    "same_verb_origin", "same_verb_current",
    "embed_sim_origin", "embed_sim_current", "embed_current_to_origin", "embed_progress",
    "path_dep_consistency", "path_pos_consistency",
]

NUM_FEATURES = len(FEATURE_NAMES)


def generate_episodes(
    doc: DocData,
    sent_idx: int,
    start_token: int,
    end_token: int,
    window_tokens: int = 150,
) -> list[tuple[np.ndarray, int]]:
    # end_token is inclusive for LitBank
    chain_id = _find_chain_id(sent_idx, start_token, end_token, doc.coref_chains)
    if chain_id < 0:
        return []

    correct_positions = _get_chain_positions(chain_id, doc)
    origin_tok = doc.parsed.sentences[sent_idx].tokens[start_token]

    if origin_tok.pos == "PROPN":
        return []

    state = PathState(
        origin_sent=sent_idx,
        origin_token=start_token,
        current_sent=sent_idx,
        current_token=start_token,
        hop_count=0,
        visited={(sent_idx, start_token)},
        path_history=[(sent_idx, start_token, origin_tok.pos, origin_tok.dep_rel)],
        path_dep_roles=[origin_tok.dep_rel],
        path_pos_tags=[origin_tok.pos],
    )

    results = []

    for hop in range(MAX_HOPS):
        current_abs = doc.abs_pos(state.current_sent, state.current_token)

        # Gather candidates
        candidates = []
        for si, sent in enumerate(doc.parsed.sentences):
            for ti, tok in enumerate(sent.tokens):
                if tok.pos not in ("NOUN", "PROPN", "PRON"):
                    continue
                if (si, ti) in state.visited:
                    continue
                abs_pos = doc.abs_pos(si, ti)
                if abs(abs_pos - current_abs) <= window_tokens:
                    candidates.append((si, ti, tok))

        if not candidates:
            break

        # Build features for each candidate
        correct_idx = None
        for i, (csi, cti, ctok) in enumerate(candidates):
            is_correct = (csi, cti) in correct_positions
            features = build_episode_features(doc, state, csi, cti)
            results.append((features, 1 if is_correct else 0))

            if is_correct and correct_idx is None:
                correct_idx = i

        # Teacher forcing: move to nearest correct
        if correct_idx is None:
            break

        correct_cands = [
            (i, doc.abs_pos(candidates[i][0], candidates[i][1]))
            for i, (csi, cti, _) in enumerate(candidates)
            if (csi, cti) in correct_positions
        ]
        if not correct_cands:
            break

        best_i = min(correct_cands, key=lambda x: abs(x[1] - current_abs))[0]
        next_si, next_ti, next_tok = candidates[best_i]

        # Update state with path history
        state.visited.add((next_si, next_ti))
        state.current_sent = next_si
        state.current_token = next_ti
        state.hop_count += 1
        state.path_history.append((next_si, next_ti, next_tok.pos, next_tok.dep_rel))
        state.path_dep_roles.append(next_tok.dep_rel)
        state.path_pos_tags.append(next_tok.pos)

        if next_tok.pos == "PROPN":
            break

    return results


def process_litbank(num_docs: int = 80, window_tokens: int = 150) -> tuple[np.ndarray, np.ndarray]:
    parser = load_parser(device="cuda", dtype=torch.float16)
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    config_path = hf_hub_download(
        repo_id="ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt", filename="config.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    embed_store = EmbeddingStore(
        np.load("data/litbank_sentence_embeddings.npz")["embeddings"],
        [tuple(x) for x in json.loads(open("data/litbank_embedding_index.json").read())],
    )
    spacy_nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer"])
    ds = load_from_disk("data/litbank")

    all_features = []
    all_labels = []

    for doc_idx in range(min(num_docs, ds["train"].num_rows)):
        sample = ds["train"][doc_idx]
        doc = precompute_litbank_doc(sample, doc_idx, parser, tokenizer, config, embed_store, spacy_nlp)

        for chain in doc.coref_chains:
            if len(chain) < 2:
                continue
            for mention in chain:
                si, st, en = mention
                if si >= len(doc.parsed.sentences):
                    continue
                if st >= len(doc.parsed.sentences[si].tokens):
                    continue

                episodes = generate_episodes(doc, si, st, en, window_tokens)
                for features, label in episodes:
                    all_features.append(features)
                    all_labels.append(label)

        if (doc_idx + 1) % 5 == 0:
            print(f"  Processed {doc_idx + 1}/{num_docs} docs, {len(all_features)} episodes")

    return np.array(all_features), np.array(all_labels, dtype=np.int32)
