from dataclasses import dataclass, field

import json
import torch
from transformers import DebertaV2TokenizerFast
from huggingface_hub import hf_hub_download

from disambiguation.parsing.loader import load_parser
from disambiguation.parsing.model import BiaffineParser

REPO_ID = "ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt"


@dataclass
class Token:
    text: str
    pos: str
    dep_rel: str
    head_idx: int
    idx_in_sent: int
    sent_idx: int


@dataclass
class MentionSignals:
    mention_text: str
    sent_idx: int
    start_token: int
    end_token: int
    pos: str
    dep_rel: str
    head_idx: int
    backward_context: list[str]
    forward_context: list[str]
    dep_tree_tokens: list[str]
    cluster_id: int = -1


@dataclass
class ParsedSentence:
    tokens: list[Token]
    words: list[str]


@dataclass
class ParsedDocument:
    sentences: list[ParsedSentence]
    mentions: list[MentionSignals] = field(default_factory=list)


def _load_config() -> dict:
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path) as f:
        return json.load(f)


def _build_subword_grid(words: list[str], tokenizer: DebertaV2TokenizerFast, fix_len: int) -> torch.Tensor:
    grid = [[1] + [0] * (fix_len - 1)]
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)[:fix_len]
        ids = ids + [0] * (fix_len - len(ids))
        grid.append(ids)
    return torch.tensor([grid], dtype=torch.long)


def _get_dep_tree_tokens(token_idx: int, tokens: list[Token]) -> list[str]:
    # Collect all tokens reachable from this token via dep relations
    # Go up to head, and collect all dependents of the head
    visited = set()
    result = []

    head_idx = tokens[token_idx].head_idx
    if head_idx != token_idx:
        result.append(tokens[head_idx].text)
        visited.add(head_idx)

    # Collect siblings (other dependents of the same head)
    for i, tok in enumerate(tokens):
        if i == token_idx:
            continue
        if tok.head_idx == head_idx and i not in visited:
            result.append(tok.text)
            visited.add(i)

    # Collect direct dependents of this token
    for i, tok in enumerate(tokens):
        if tok.head_idx == token_idx and i not in visited:
            result.append(tok.text)
            visited.add(i)

    return result


def parse_sentence_words(
    words: list[str],
    sent_idx: int,
    parser: BiaffineParser,
    tokenizer: DebertaV2TokenizerFast,
    config: dict,
    device: str,
) -> ParsedSentence:
    fix_len = config["fix_len"]
    rel_vocab_inv = {v: k for k, v in config["rel_vocab"].items()}
    pos_vocab_inv = {v: k for k, v in config["pos_vocab"].items()}

    grid = _build_subword_grid(words, tokenizer, fix_len).to(device)
    with torch.no_grad():
        s_arc, s_rel, s_pos = parser(grid)

    arc_preds = s_arc[0].argmax(dim=-1).cpu().tolist()
    pos_preds = s_pos[0].argmax(dim=-1).cpu().tolist()
    rel_preds = s_rel[0].cpu()

    tokens = []
    for i, word in enumerate(words):
        word_idx = i + 1  # offset by ROOT
        head_idx = arc_preds[word_idx]
        rel_idx = rel_preds[word_idx, head_idx].argmax().item()
        pos_idx = pos_preds[word_idx]
        tokens.append(Token(
            text=word,
            pos=pos_vocab_inv.get(pos_idx, "X"),
            dep_rel=rel_vocab_inv.get(rel_idx, "dep"),
            head_idx=head_idx - 1,  # convert to 0-based (ROOT becomes -1)
            idx_in_sent=i,
            sent_idx=sent_idx,
        ))

    return ParsedSentence(tokens=tokens, words=words)


def extract_mention_signals(
    parsed_doc: ParsedDocument,
    sent_idx: int,
    start_token: int,
    end_token: int,
    context_window: int = 5,
    cluster_id: int = -1,
) -> MentionSignals:
    sent = parsed_doc.sentences[sent_idx]
    mention_tokens = sent.tokens[start_token:end_token]
    mention_text = " ".join(t.text for t in mention_tokens)

    # Use the head token of the mention span for dep features
    head_token = mention_tokens[0]
    for t in mention_tokens:
        if t.head_idx < start_token or t.head_idx >= end_token:
            head_token = t
            break

    # Backward context (nearest first)
    backward = []
    for i in range(start_token - 1, max(start_token - 1 - context_window, -1), -1):
        backward.append(sent.tokens[i].text)
    # If not enough tokens in this sentence, look at previous sentence
    if len(backward) < context_window and sent_idx > 0:
        prev_sent = parsed_doc.sentences[sent_idx - 1]
        remaining = context_window - len(backward)
        for i in range(len(prev_sent.tokens) - 1, max(len(prev_sent.tokens) - 1 - remaining, -1), -1):
            backward.append(prev_sent.tokens[i].text)

    # Forward context (nearest first)
    forward = []
    for i in range(end_token, min(end_token + context_window, len(sent.tokens))):
        forward.append(sent.tokens[i].text)
    # If not enough, look at next sentence
    if len(forward) < context_window and sent_idx < len(parsed_doc.sentences) - 1:
        next_sent = parsed_doc.sentences[sent_idx + 1]
        remaining = context_window - len(forward)
        for i in range(min(remaining, len(next_sent.tokens))):
            forward.append(next_sent.tokens[i].text)

    # Dep tree tokens for the head token of the mention
    dep_tree = _get_dep_tree_tokens(head_token.idx_in_sent, sent.tokens)

    return MentionSignals(
        mention_text=mention_text,
        sent_idx=sent_idx,
        start_token=start_token,
        end_token=end_token,
        pos=head_token.pos,
        dep_rel=head_token.dep_rel,
        head_idx=head_token.head_idx,
        backward_context=backward,
        forward_context=forward,
        dep_tree_tokens=dep_tree,
        cluster_id=cluster_id,
    )


def parse_document(
    sentences: list[list[str]],
    parser: BiaffineParser,
    tokenizer: DebertaV2TokenizerFast,
    config: dict,
    device: str = "cuda",
) -> ParsedDocument:
    parsed_sents = []
    for i, words in enumerate(sentences):
        parsed = parse_sentence_words(words, i, parser, tokenizer, config, device)
        parsed_sents.append(parsed)
    return ParsedDocument(sentences=parsed_sents)
