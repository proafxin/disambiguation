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

MAX_HOPS = 512
TOP_K_CANDIDATES = 20

# Encode categoricals as integers
POS_IDS = {"PROPN": 0, "NOUN": 1, "PRON": 2, "ADJ": 3, "VERB": 4, "DET": 5, "ADP": 6, "AUX": 7, "ADV": 8, "SCONJ": 9, "CCONJ": 10, "PART": 11, "NUM": 12, "PUNCT": 13, "X": 14, "INTJ": 15, "SYM": 16}
DEP_IDS = {
    "nsubj": 0, "obj": 1, "obl": 2, "nmod": 3, "nmod:poss": 4, "appos": 5, "conj": 6,
    "compound": 7, "flat": 8, "det": 9, "amod": 10, "root": 11, "nsubj:pass": 12,
    "obl:agent": 13, "iobj": 14, "ccomp": 15, "xcomp": 16, "acl": 17, "acl:relcl": 18,
    "advcl": 19, "expl": 20, "cop": 21, "advmod": 22, "mark": 23, "aux": 24,
    "aux:pass": 25, "case": 26, "cc": 27, "punct": 28, "nummod": 29, "compound:prt": 30,
    "nmod:unmarked": 31, "obl:unmarked": 32, "fixed": 33, "parataxis": 34,
}
GENDER_IDS = {"Masc": 0, "Fem": 1, "Neut": 2, "unknown": 3}
NUMBER_IDS = {"Sing": 0, "Plur": 1, "unknown": 2}
CASE_IDS = {"Nom": 0, "Acc": 1, "Gen": 2, "Dat": 3, "unknown": 4}
PRONTYPE_IDS = {"Prs": 0, "Art": 1, "Dem": 2, "Rel": 3, "Int": 4, "unknown": 5}
DEFINITE_IDS = {"Def": 0, "Ind": 1, "unknown": 2}


@dataclass
class TokenAttrs:
    pos: int
    dep: int
    gender: int
    number: int
    case: int
    prontype: int
    definite: int
    person: int
    is_alpha: int
    is_stop: int
    head_offset: int  # distance to head token
    n_children: int
    sent_position: float  # relative position in sentence (0-1)


@dataclass
class DocData:
    parsed: ParsedDocument
    sentence_embeddings: np.ndarray
    sentences: list[list[str]]
    coref_chains: list[list[list[int]]]
    token_attrs: list[list[TokenAttrs]]  # [sent_idx][token_idx]
    mention_embeddings: dict = field(default_factory=dict)  # (sent_idx, token_idx) -> embedding (NOUN/PROPN only)
    sentence_lengths: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sentence_lengths:
            self.sentence_lengths = [len(s.tokens) for s in self.parsed.sentences]

    def abs_pos(self, sent_idx: int, token_idx: int) -> int:
        return sum(self.sentence_lengths[:sent_idx]) + token_idx


@dataclass
class ChainState:
    origin_sent: int
    origin_token: int
    current_sent: int
    current_token: int
    hop_count: int
    visited: set = field(default_factory=set)
    # Chain noun embeddings (only from NOUN/PROPN mentions in the chain)
    chain_noun_embs: list[np.ndarray] = field(default_factory=list)
    # Chain profile: accumulated attributes
    chain_genders: list[int] = field(default_factory=list)
    chain_numbers: list[int] = field(default_factory=list)
    chain_deps: list[int] = field(default_factory=list)
    chain_pos: list[int] = field(default_factory=list)
    # Resolved gender from pronouns (strongest signal)
    resolved_gender: int = 3  # 3 = unknown


def _extract_token_attrs(spacy_doc, parsed_tokens) -> list[TokenAttrs]:
    attrs = []
    for i, token in enumerate(spacy_doc):
        morph = token.morph
        head_offset = token.head.i - token.i if token.head != token else 0
        n_children = len(list(token.children))
        sent_len = len(spacy_doc)
        attrs.append(TokenAttrs(
            pos=POS_IDS.get(token.pos_, len(POS_IDS)),
            dep=DEP_IDS.get(token.dep_, len(DEP_IDS)),
            gender=GENDER_IDS.get(morph.get("Gender", ["unknown"])[0], 3),
            number=NUMBER_IDS.get(morph.get("Number", ["unknown"])[0], 2),
            case=CASE_IDS.get(morph.get("Case", ["unknown"])[0], 4),
            prontype=PRONTYPE_IDS.get(morph.get("PronType", ["unknown"])[0], 5),
            definite=DEFINITE_IDS.get(morph.get("Definite", ["unknown"])[0], 2),
            person=int(morph.get("Person", ["0"])[0]),
            is_alpha=int(token.is_alpha),
            is_stop=int(token.is_stop),
            head_offset=head_offset,
            n_children=n_children,
            sent_position=i / max(sent_len - 1, 1),
        ))
    # Pad/truncate to match parsed token count
    target_len = len(parsed_tokens) if parsed_tokens else 0
    while len(attrs) < target_len:
        attrs.append(TokenAttrs(0, 0, 3, 2, 4, 5, 2, 0, 0, 0, 0, 0, 0.0))
    return attrs[:target_len]


def precompute_doc(
    sample: dict,
    doc_idx: int,
    parser,
    tokenizer,
    config: dict,
    embed_store: EmbeddingStore,
    spacy_nlp,
    mention_embedder=None,
) -> DocData:
    parsed_doc = parse_document(sample["sentences"], parser, tokenizer, config, device="cuda")
    num_sents = len(sample["sentences"])
    sentence_embeddings = embed_store.get_document_embeddings(doc_idx, num_sents)

    token_attrs = []
    for si, sent in enumerate(sample["sentences"]):
        text = " ".join(sent)
        spacy_doc = spacy_nlp(text)
        parsed_tokens = parsed_doc.sentences[si].tokens if si < len(parsed_doc.sentences) else []
        attrs = _extract_token_attrs(spacy_doc, parsed_tokens)
        token_attrs.append(attrs)

    # Compute mention embeddings for NOUN/PROPN tokens only
    mention_embeddings = {}
    if mention_embedder is not None:
        noun_mentions = []
        noun_keys = []
        for si, sent in enumerate(sample["sentences"]):
            for ti in range(len(sent)):
                if ti >= len(token_attrs[si]):
                    continue
                if token_attrs[si][ti].pos in (POS_IDS["NOUN"], POS_IDS["PROPN"]):
                    noun_mentions.append(sent[ti])
                    noun_keys.append((si, ti))
        if noun_mentions:
            embs = mention_embedder.encode(noun_mentions, convert_to_numpy=True, show_progress_bar=False)
            for key, emb in zip(noun_keys, embs):
                mention_embeddings[key] = emb.astype(np.float16)

    return DocData(
        parsed=parsed_doc,
        sentence_embeddings=sentence_embeddings,
        sentences=sample["sentences"],
        coref_chains=sample["coref_chains"],
        token_attrs=token_attrs,
        mention_embeddings=mention_embeddings,
    )


def _token_to_vector(attrs: TokenAttrs) -> np.ndarray:
    return np.array([
        attrs.pos, attrs.dep, attrs.gender, attrs.number,
        attrs.case, attrs.prontype, attrs.definite, attrs.person,
        attrs.is_alpha, attrs.is_stop, attrs.head_offset,
        attrs.n_children, attrs.sent_position,
    ], dtype=np.float32)


TOKEN_ATTR_DIM = 13


def build_candidate_features(
    doc: DocData,
    state: ChainState,
    cand_sent: int,
    cand_token: int,
    rank_in_topk: int,
    total_candidates: int,
    sim_to_origin: float,
    sim_to_current: float,
) -> np.ndarray:
    origin_attrs = doc.token_attrs[state.origin_sent][state.origin_token]
    current_attrs = doc.token_attrs[state.current_sent][state.current_token]
    cand_attrs = doc.token_attrs[cand_sent][cand_token]

    # Token attribute vectors
    origin_vec = _token_to_vector(origin_attrs)
    current_vec = _token_to_vector(current_attrs)
    cand_vec = _token_to_vector(cand_attrs)

    # Distances
    current_abs = doc.abs_pos(state.current_sent, state.current_token)
    cand_abs = doc.abs_pos(cand_sent, cand_token)
    origin_abs = doc.abs_pos(state.origin_sent, state.origin_token)

    # Agreement signals
    gender_match_origin = int(origin_attrs.gender == 3 or cand_attrs.gender == 3 or origin_attrs.gender == cand_attrs.gender)
    number_match_origin = int(origin_attrs.number == 2 or cand_attrs.number == 2 or origin_attrs.number == cand_attrs.number)
    gender_match_current = int(current_attrs.gender == 3 or cand_attrs.gender == 3 or current_attrs.gender == cand_attrs.gender)
    number_match_current = int(current_attrs.number == 2 or cand_attrs.number == 2 or current_attrs.number == cand_attrs.number)

    # Chain consistency
    chain_gender_consistent = 0.0
    if state.chain_genders:
        known = [g for g in state.chain_genders if g != 3]
        if known:
            chain_gender_consistent = float(cand_attrs.gender == 3 or cand_attrs.gender in known)

    chain_number_consistent = 0.0
    if state.chain_numbers:
        known = [n for n in state.chain_numbers if n != 2]
        if known:
            chain_number_consistent = float(cand_attrs.number == 2 or cand_attrs.number in known)

    chain_dep_consistent = 0.0
    if state.chain_deps:
        chain_dep_consistent = sum(1 for d in state.chain_deps if d == cand_attrs.dep) / len(state.chain_deps)

    chain_pos_consistent = 0.0
    if state.chain_pos:
        chain_pos_consistent = sum(1 for p in state.chain_pos if p == cand_attrs.pos) / len(state.chain_pos)

    # Dep tree overlap
    cand_tree_texts = set()
    current_tree_texts = set()
    origin_tree_texts = set()
    if cand_sent < len(doc.parsed.sentences):
        for t in doc.parsed.sentences[cand_sent].tokens:
            if t.head_idx == cand_token or t.idx_in_sent == cand_token:
                cand_tree_texts.add(t.text.lower())
    if state.current_sent < len(doc.parsed.sentences):
        for t in doc.parsed.sentences[state.current_sent].tokens:
            if t.head_idx == state.current_token or t.idx_in_sent == state.current_token:
                current_tree_texts.add(t.text.lower())
    if state.origin_sent < len(doc.parsed.sentences):
        for t in doc.parsed.sentences[state.origin_sent].tokens:
            if t.head_idx == state.origin_token or t.idx_in_sent == state.origin_token:
                origin_tree_texts.add(t.text.lower())

    # Resolved gender match (from pronouns in chain)
    resolved_gender_match = int(state.resolved_gender == 3 or cand_attrs.gender == 3 or state.resolved_gender == cand_attrs.gender)

    # Noun embedding similarity (only meaningful for NOUN/PROPN candidates)
    noun_emb_sim_to_chain = 0.0
    noun_emb_sim_to_origin = 0.0
    cand_key = (cand_sent, cand_token)
    if cand_key in doc.mention_embeddings:
        cand_mention_emb = doc.mention_embeddings[cand_key].astype(np.float32)
        cand_norm = np.linalg.norm(cand_mention_emb) + 1e-8
        # Similarity to origin noun embedding
        origin_key = (state.origin_sent, state.origin_token)
        if origin_key in doc.mention_embeddings:
            origin_m_emb = doc.mention_embeddings[origin_key].astype(np.float32)
            noun_emb_sim_to_origin = float(np.dot(cand_mention_emb, origin_m_emb) / (cand_norm * (np.linalg.norm(origin_m_emb) + 1e-8)))
        # Max similarity to any noun in the chain
        if state.chain_noun_embs:
            best = 0.0
            for e in state.chain_noun_embs:
                s = float(np.dot(cand_mention_emb, e) / (cand_norm * (np.linalg.norm(e) + 1e-8)))
                if s > best:
                    best = s
            noun_emb_sim_to_chain = best

    features = np.concatenate([
        origin_vec,                          # 13
        current_vec,                         # 13
        cand_vec,                            # 13
        np.array([
            state.hop_count,                 # 1
            abs(cand_abs - current_abs),     # 1
            abs(cand_abs - origin_abs),      # 1
            abs(cand_sent - state.current_sent),  # 1
            # Agreement
            gender_match_origin,             # 1
            number_match_origin,             # 1
            gender_match_current,            # 1
            number_match_current,            # 1
            resolved_gender_match,           # 1
            # Structural
            int(cand_attrs.dep == current_attrs.dep),  # 1
            int(cand_attrs.dep == origin_attrs.dep),   # 1
            int(cand_attrs.pos == current_attrs.pos),  # 1
            len(cand_tree_texts & origin_tree_texts),  # 1
            len(cand_tree_texts & current_tree_texts), # 1
            # Embedding (sentence level)
            sim_to_origin,                   # 1
            sim_to_current,                  # 1
            # Noun embedding (mention level, NOUN/PROPN only)
            noun_emb_sim_to_origin,          # 1
            noun_emb_sim_to_chain,           # 1
            # Competition
            rank_in_topk,                    # 1
            total_candidates,                # 1
            # Chain consistency
            chain_gender_consistent,         # 1
            chain_number_consistent,         # 1
            chain_dep_consistent,            # 1
            chain_pos_consistent,            # 1
        ], dtype=np.float32),
    ])
    return features


# 13 + 13 + 13 + 24 = 63 features
NUM_FEATURES = TOKEN_ATTR_DIM * 3 + 24


def _find_chain_id(sent_idx: int, start: int, end: int, chains: list[list[list[int]]]) -> int:
    for ci, chain in enumerate(chains):
        for mention in chain:
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


def generate_episodes(
    doc: DocData,
    sent_idx: int,
    start_token: int,
    end_token: int,
    window_tokens: int = 150,
) -> list[tuple[np.ndarray, int]]:
    chain_id = _find_chain_id(sent_idx, start_token, end_token, doc.coref_chains)
    if chain_id < 0:
        return []

    correct_positions = _get_chain_positions(chain_id, doc)

    # Skip if already PROPN
    if start_token < len(doc.token_attrs[sent_idx]):
        if doc.token_attrs[sent_idx][start_token].pos == POS_IDS["PROPN"]:
            return []

    origin_emb = doc.sentence_embeddings[sent_idx]
    origin_attrs = doc.token_attrs[sent_idx][start_token]

    # Initialize chain noun embeddings
    origin_noun_embs = []
    if (sent_idx, start_token) in doc.mention_embeddings:
        origin_noun_embs.append(doc.mention_embeddings[(sent_idx, start_token)].astype(np.float32))

    state = ChainState(
        origin_sent=sent_idx,
        origin_token=start_token,
        current_sent=sent_idx,
        current_token=start_token,
        hop_count=0,
        visited={(sent_idx, start_token)},
        chain_noun_embs=origin_noun_embs,
        chain_genders=[origin_attrs.gender],
        chain_numbers=[origin_attrs.number],
        chain_deps=[origin_attrs.dep],
        chain_pos=[origin_attrs.pos],
        resolved_gender=origin_attrs.gender if origin_attrs.gender != 3 else 3,
    )

    results = []

    for hop in range(MAX_HOPS):
        current_abs = doc.abs_pos(state.current_sent, state.current_token)

        # Gather ALL nominal candidates in window
        all_candidates = []
        for si, sent in enumerate(doc.parsed.sentences):
            for ti in range(len(sent.tokens)):
                if ti >= len(doc.token_attrs[si]):
                    continue
                attrs = doc.token_attrs[si][ti]
                if attrs.pos not in (POS_IDS["NOUN"], POS_IDS["PROPN"], POS_IDS["PRON"]):
                    continue
                if (si, ti) in state.visited:
                    continue
                abs_pos = doc.abs_pos(si, ti)
                if abs(abs_pos - current_abs) <= window_tokens:
                    all_candidates.append((si, ti, abs_pos))

        if not all_candidates:
            break

        # Rank by embedding similarity to origin sentence — pick top-K
        cand_sent_indices = [c[0] for c in all_candidates]
        cand_embs = doc.sentence_embeddings[cand_sent_indices]
        sims_to_origin = cosine_similarity([origin_emb.astype(np.float32)], cand_embs.astype(np.float32))[0]
        current_emb = doc.sentence_embeddings[state.current_sent]
        sims_to_current = cosine_similarity([current_emb.astype(np.float32)], cand_embs.astype(np.float32))[0]

        # Top-K by origin similarity
        top_indices = np.argsort(-sims_to_origin)[:TOP_K_CANDIDATES]

        # Generate features for top-K candidates
        correct_in_topk = None
        for rank, idx in enumerate(top_indices):
            csi, cti, cabs = all_candidates[idx]
            is_correct = (csi, cti) in correct_positions

            features = build_candidate_features(
                doc, state, csi, cti,
                rank_in_topk=rank,
                total_candidates=len(all_candidates),
                sim_to_origin=float(sims_to_origin[idx]),
                sim_to_current=float(sims_to_current[idx]),
            )
            results.append((features, 1 if is_correct else 0))

            if is_correct and correct_in_topk is None:
                correct_in_topk = idx

        # If correct candidate not in top-K, check if it exists at all
        if correct_in_topk is None:
            # Find correct candidate in full list
            for idx, (csi, cti, cabs) in enumerate(all_candidates):
                if (csi, cti) in correct_positions:
                    correct_in_topk = idx
                    break

        if correct_in_topk is None:
            break

        # Teacher forcing: move to nearest correct
        correct_cands = [
            (idx, all_candidates[idx][2])
            for idx, (csi, cti, _) in enumerate(all_candidates)
            if (csi, cti) in correct_positions
        ]
        if not correct_cands:
            break

        best_idx = min(correct_cands, key=lambda x: abs(x[1] - current_abs))[0]
        next_si, next_ti, _ = all_candidates[best_idx]

        # Update chain state
        state.visited.add((next_si, next_ti))
        state.current_sent = next_si
        state.current_token = next_ti
        state.hop_count += 1

        # Add noun embedding if this hop is a NOUN/PROPN
        if (next_si, next_ti) in doc.mention_embeddings:
            state.chain_noun_embs.append(doc.mention_embeddings[(next_si, next_ti)].astype(np.float32))

        # Update chain profile
        if next_ti < len(doc.token_attrs[next_si]):
            next_attrs = doc.token_attrs[next_si][next_ti]
            state.chain_genders.append(next_attrs.gender)
            state.chain_numbers.append(next_attrs.number)
            state.chain_deps.append(next_attrs.dep)
            state.chain_pos.append(next_attrs.pos)
            # Resolve gender from pronouns (lock it once known)
            if state.resolved_gender == 3 and next_attrs.gender != 3:
                state.resolved_gender = next_attrs.gender

        # Terminal check
        if next_ti < len(doc.token_attrs[next_si]) and doc.token_attrs[next_si][next_ti].pos == POS_IDS["PROPN"]:
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

    from sentence_transformers import SentenceTransformer
    mention_embedder = SentenceTransformer("all-MiniLM-L6-v2")

    all_features = []
    all_labels = []

    for doc_idx in range(min(num_docs, ds["train"].num_rows)):
        sample = ds["train"][doc_idx]
        doc = precompute_doc(sample, doc_idx, parser, tokenizer, config, embed_store, spacy_nlp, mention_embedder)

        for chain in doc.coref_chains:
            if len(chain) < 2:
                continue
            for mention in chain:
                si, st, en = mention
                if si >= len(doc.parsed.sentences):
                    continue
                if st >= len(doc.token_attrs[si]):
                    continue

                episodes = generate_episodes(doc, si, st, en, window_tokens)
                for features, label in episodes:
                    all_features.append(features)
                    all_labels.append(label)

        if (doc_idx + 1) % 5 == 0:
            print(f"  Processed {doc_idx + 1}/{num_docs} docs, {len(all_features)} episodes")

    return np.array(all_features), np.array(all_labels, dtype=np.int32)
