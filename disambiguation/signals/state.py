from dataclasses import dataclass, field

from disambiguation.signals.extraction import MentionSignals, ParsedDocument, extract_mention_signals


@dataclass
class Candidate:
    mention: MentionSignals
    token_position: int  # absolute position in document token stream
    sentence_idx: int


@dataclass
class State:
    origin: MentionSignals  # starting mention (never changes)
    current: MentionSignals  # where we are now
    current_token_position: int
    candidates: list[Candidate]
    hop_count: int
    path: list[int] = field(default_factory=list)  # token positions visited


def get_absolute_token_position(sent_idx: int, token_idx: int, sentence_lengths: list[int]) -> int:
    return sum(sentence_lengths[:sent_idx]) + token_idx


def build_initial_state(
    parsed_doc: ParsedDocument,
    sent_idx: int,
    start_token: int,
    end_token: int,
    window_tokens: int = 200,
    context_window: int = 5,
) -> State:
    sentence_lengths = [len(s.tokens) for s in parsed_doc.sentences]
    origin_signals = extract_mention_signals(parsed_doc, sent_idx, start_token, end_token, context_window)
    origin_pos = get_absolute_token_position(sent_idx, start_token, sentence_lengths)

    # Find candidates within token window
    candidates = []
    for si, sent in enumerate(parsed_doc.sentences):
        for ti, tok in enumerate(sent.tokens):
            abs_pos = get_absolute_token_position(si, ti, sentence_lengths)
            if abs(abs_pos - origin_pos) > window_tokens:
                continue
            if si == sent_idx and ti >= start_token and ti < end_token:
                continue  # skip self
            if tok.pos not in ("NOUN", "PROPN", "PRON"):
                continue
            cand_signals = extract_mention_signals(parsed_doc, si, ti, ti + 1, context_window)
            candidates.append(Candidate(
                mention=cand_signals,
                token_position=abs_pos,
                sentence_idx=si,
            ))

    return State(
        origin=origin_signals,
        current=origin_signals,
        current_token_position=origin_pos,
        candidates=candidates,
        hop_count=0,
        path=[origin_pos],
    )


def make_move(state: State, candidate_idx: int, parsed_doc: ParsedDocument, window_tokens: int = 200, context_window: int = 5) -> State:
    chosen = state.candidates[candidate_idx]
    sentence_lengths = [len(s.tokens) for s in parsed_doc.sentences]
    new_pos = chosen.token_position

    # Rebuild candidates from new position
    new_candidates = []
    for si, sent in enumerate(parsed_doc.sentences):
        for ti, tok in enumerate(sent.tokens):
            abs_pos = get_absolute_token_position(si, ti, sentence_lengths)
            if abs(abs_pos - new_pos) > window_tokens:
                continue
            if abs_pos in state.path:
                continue  # don't revisit
            if si == chosen.sentence_idx and ti == chosen.mention.start_token:
                continue  # skip self
            if tok.pos not in ("NOUN", "PROPN", "PRON"):
                continue
            cand_signals = extract_mention_signals(parsed_doc, si, ti, ti + 1, context_window)
            new_candidates.append(Candidate(
                mention=cand_signals,
                token_position=abs_pos,
                sentence_idx=si,
            ))

    return State(
        origin=state.origin,
        current=chosen.mention,
        current_token_position=new_pos,
        candidates=new_candidates,
        hop_count=state.hop_count + 1,
        path=state.path + [new_pos],
    )


def is_terminal(state: State) -> bool:
    return state.current.pos == "PROPN"


def extract_state_features(state: State) -> dict:
    # Raw features exposed to the evaluation function
    # Origin features
    origin = state.origin
    current = state.current

    # Candidate landscape summary
    candidate_pos_counts = {"NOUN": 0, "PROPN": 0, "PRON": 0}
    candidate_dep_rels = []
    propn_distances = []

    for cand in state.candidates:
        pos = cand.mention.pos
        if pos in candidate_pos_counts:
            candidate_pos_counts[pos] += 1
        candidate_dep_rels.append(cand.mention.dep_rel)
        if cand.mention.pos == "PROPN":
            propn_distances.append(abs(cand.token_position - state.current_token_position))

    return {
        "origin_pos": origin.pos,
        "origin_dep_rel": origin.dep_rel,
        "origin_dep_tree": origin.dep_tree_tokens,
        "current_pos": current.pos,
        "current_dep_rel": current.dep_rel,
        "current_dep_tree": current.dep_tree_tokens,
        "current_forward_ctx": current.forward_context,
        "current_backward_ctx": current.backward_context,
        "hop_count": state.hop_count,
        "num_candidates": len(state.candidates),
        "num_propn_visible": candidate_pos_counts["PROPN"],
        "num_noun_visible": candidate_pos_counts["NOUN"],
        "num_pron_visible": candidate_pos_counts["PRON"],
        "nearest_propn_distance": min(propn_distances) if propn_distances else -1,
        "candidate_dep_rels": candidate_dep_rels,
    }
