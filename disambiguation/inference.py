import re

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from disambiguation.paths import MODELS_DIR, SPAN_CTX_CACHE
from disambiguation.stage2_context_encoder import (
    BACKBONE,
    BGE_DIM,
    CKPT_B_NAME,
    CKPT_NAME,
    CONTENT,
    CTX_DIM,
    AntecedentScorer,
    ClusterMatcher,
    ContextEncoder,
    decode_antecedents,
    decode_cluster_matches,
    encode_document_ctx,
    load_tokenizer,
)

BGE_MODEL = "BAAI/bge-large-en-v1.5"

WIN_TAG = f"_k{CONTENT}"
SPAN_CTX_CACHE = SPAN_CTX_CACHE.with_name(SPAN_CTX_CACHE.name + WIN_TAG)
CKPT_B_NAME = f"stage2_cluster_matcher{WIN_TAG}.pt"


def _load_models(device: str) -> tuple:
    tokenizer = load_tokenizer()
    encoder = ContextEncoder().to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    stage_a = AntecedentScorer().to(device)
    stage_a.load_state_dict(torch.load(MODELS_DIR / f"stage2_frozen_head{WIN_TAG}.pt", map_location=device)["scorer"])
    stage_a.eval()
    for p in stage_a.parameters():
        p.requires_grad_(False)

    matcher = ClusterMatcher().to(device)
    matcher.load_state_dict(torch.load(MODELS_DIR / CKPT_B_NAME, map_location=device)["cluster_matcher"])
    matcher.eval()
    for p in matcher.parameters():
        p.requires_grad_(False)

    bge = SentenceTransformer(BGE_MODEL, device=device)
    nlp = spacy.load("en_core_web_sm")
    return tokenizer, encoder, stage_a, matcher, bge, nlp


def _extract_mentions(text: str, nlp) -> tuple[list[str], list[tuple[int, int]]]:
    doc = nlp(text)
    tokens = [t.text for t in doc]
    mentions = []
    spans = []
    for chunk in doc.noun_chunks:
        spans.append((chunk.start, chunk.end))
        mentions.append(chunk.text)
    for ent in doc.ents:
        span = (ent.start, ent.end)
        if span not in spans:
            spans.append(span)
            mentions.append(ent.text)
    # pronouns not covered by noun chunks
    for t in doc:
        if t.pos_ == "PRON" and (t.i, t.i + 1) not in spans:
            spans.append((t.i, t.i + 1))
            mentions.append(t.text)
    order = sorted(range(len(spans)), key=lambda i: spans[i])
    spans = [spans[i] for i in order]
    mentions = [mentions[i] for i in order]
    return tokens, spans, mentions


def _word_to_subtok(words: list, tokenizer: AutoTokenizer) -> tuple[np.ndarray, dict]:
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    word_ids = enc.word_ids()
    w2s: dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in w2s:
            w2s[wid] = pos
    return np.asarray(enc["input_ids"], dtype=np.int64), w2s


def _stage_a(
    ctx: torch.Tensor,
    bge_vecs: torch.Tensor,
    tok_pos: np.ndarray,
    stage_a: AntecedentScorer,
    device: str,
) -> dict[int, list[list[int]]]:
    win_ids = tok_pos // CONTENT
    per_window: dict[int, list[list[int]]] = {}
    with torch.inference_mode():
        for w in np.unique(win_ids):
            idx = np.where(win_ids == w)[0]
            if len(idx) < 2:
                per_window[int(w)] = [[int(i)] for i in idx]
                continue
            w_scores, w_mask = stage_a(ctx[idx], bge_vecs[idx])
            groups = decode_antecedents(
                w_scores.float().cpu().numpy(), w_mask.cpu().numpy(), float(stage_a.null_bias.item())
            )
            grouped = {i for g in groups for i in g}
            clusters = [[int(idx[i]) for i in g] for g in groups]
            clusters += [[int(idx[i])] for i in range(len(idx)) if i not in grouped]
            per_window[int(w)] = clusters
    return per_window


def _stage_b(
    per_window: dict[int, list[list[int]]],
    ctx: torch.Tensor,
    bge_vecs: torch.Tensor,
    matcher: ClusterMatcher,
    device: str,
) -> tuple[list[list[int]], list[tuple[int, int, int, int, float]]]:
    # returns final clusters (global mention indices) and merge log
    # merge log: (win_k, left_cluster_idx, win_k+1, right_cluster_idx, score)
    from disambiguation.train_stage2 import _cluster_reps

    all_clusters: list[list[int]] = []
    win_offsets: dict[int, int] = {}
    windows = sorted(per_window.keys())
    for w in windows:
        win_offsets[w] = len(all_clusters)
        all_clusters.extend(per_window[w])

    parent = list(range(len(all_clusters)))

    def uf_find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merge_log = []
    ctx_np = ctx.cpu().numpy()
    bge_np = bge_vecs.cpu().numpy()

    with torch.inference_mode():
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                lc = per_window[windows[i]]
                rc = per_window[windows[j]]
                if not lc or not rc:
                    continue
                left  = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(c, ctx_np, bge_np) for c in lc]]
                right = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(c, ctx_np, bge_np) for c in rc]]
                scores = matcher(left, right).float().cpu().numpy()
                pairs = decode_cluster_matches(scores, float(matcher.null_bias.item()))
                lo, ro = win_offsets[windows[i]], win_offsets[windows[j]]
                for li, ri in pairs:
                    a, b = uf_find(lo + li), uf_find(ro + ri)
                    if a != b:
                        parent[a] = b
                        merge_log.append((windows[i], li, windows[j], ri, float(scores[li, ri])))

    groups: dict[int, list[int]] = {}
    for i in range(len(all_clusters)):
        groups.setdefault(uf_find(i), []).append(i)

    final = []
    for members in groups.values():
        mention_indices = [m for ci in members for m in all_clusters[ci]]
        if len(mention_indices) >= 2:
            final.append(sorted(mention_indices))
    return final, merge_log


def resolve(
    text: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> list[list[str]]:
    tokenizer, encoder, stage_a, matcher, bge, nlp = _load_models(device)

    tokens, spans, mentions = _extract_mentions(text, nlp)
    if len(mentions) < 2:
        print("Fewer than 2 mentions found.")
        return []

    content_ids, w2s = _word_to_subtok(tokens, tokenizer)
    last = len(content_ids) - 1

    tok_pos = np.array([w2s.get(s, min(s, last)) for s, _ in spans], dtype=np.int64)

    with torch.inference_mode():
        ctx_doc = encode_document_ctx(
            content_ids, encoder, tokenizer.cls_token_id, tokenizer.sep_token_id, device
        )
    ctx = torch.stack([ctx_doc[int(tok_pos[k])] for k in range(len(spans))])  # (M, CTX_DIM)
    # build (M, 2, CTX_DIM): use tok_pos as both start and end for single-token mentions
    ctx_2 = torch.zeros(len(spans), 2, CTX_DIM, device=device)
    for k, (s, e) in enumerate(spans):
        ss = w2s.get(s, min(s, last))
        se = w2s.get(e - 1, min(e - 1, last))
        ctx_2[k, 0] = ctx_doc[ss]
        ctx_2[k, 1] = ctx_doc[se]

    bge_vecs = torch.from_numpy(
        bge.encode(mentions, normalize_embeddings=True, batch_size=512).astype(np.float32)
    ).to(device)

    per_window = _stage_a(ctx_2, bge_vecs, tok_pos, stage_a, device)
    final_clusters, merge_log = _stage_b(per_window, ctx_2, bge_vecs, matcher, device)

    result = [[mentions[i] for i in cluster] for cluster in final_clusters]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"\nMentions ({len(mentions)}): {mentions}")
        print(f"\nWindows: {sorted(per_window.keys())}")
        print("\nStage A clusters per window:")
        for w in sorted(per_window.keys()):
            print(f"  Window {w}:")
            for ci, cluster in enumerate(per_window[w]):
                print(f"    Cluster {ci}: {[mentions[i] for i in cluster]}")
        if merge_log:
            print("\nStage B merges:")
            for wl, li, wr, ri, score in merge_log:
                lm = [mentions[i] for i in per_window[wl][li]]
                rm = [mentions[i] for i in per_window[wr][ri]]
                print(f"  Window {wl} cluster {li} {lm} <-> Window {wr} cluster {ri} {rm}  (score {score:.3f})")
        else:
            print("\nStage B: no merges")
        print("\nFinal coreference clusters:")
        for ci, cluster in enumerate(result):
            print(f"  Cluster {ci}: {cluster}")
        print('='*60)

    return result
