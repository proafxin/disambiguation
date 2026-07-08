import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from disambiguation.paths import (
    DATA_DIR,
    NOM_CACHE,
    NOM_CACHE_CP8K,
    SPAN_CTX_CACHE,
    SPAN_CTX_CP8K,
)
from disambiguation.stage2_context_encoder import (
    CONTENT,
    CTX_DIM,
    CTX_ENCODE_MAX_WINDOWS,
    ENCODER_TAG,
    encode_docs_ctx_batched,
    encode_document_ctx,
    load_tokenizer,
)

BGE_MODEL = "BAAI/bge-large-en-v1.5"
CONLL_SPLITS = ["train", "validation", "test"]
PRECO_SUBSAMPLE = 8000  # ctx_vecs held as float16 in RAM; conll+preco@8k fits ~4.4 GB fully resident
PRECO_FULL = 36620  # full PreCo train split
RANDOM_SEED = 42


_SUBSET_TAG = {"all": "", "conll": "_conllonly", "nopreco": "_nopreco", "cp8k": "_cp8k", "all8k": "_all8k", "p2c": "_p2c"}
_PRECO_CAP = {"all8k": 8000}  # cap preco *training* docs (cache may hold more)


def _data_cfg(subset: str) -> tuple:
    # (nom_cache, datasets, preco_n, single_ctx) for building/loading this subset's data.
    # cp8k is a self-contained build (conll+preco only) with one single-file ctx. all8k reuses
    # the existing all-4 per-doc cache (10k preco) and caps training to 8k preco (no rebuild).
    # The nominal cache is encoder-independent (BGE vectors + structure); only the subtoken
    # tokenization is encoder-specific and is re-derived at load by _retokenize for a non-default
    # contextual encoder — so swapping encoders never re-encodes BGE or re-runs spaCy.
    if subset in ("cp8k", "p2c"):
        return (NOM_CACHE_CP8K, ("conll2012", "preco"), PRECO_FULL, True)
    return (NOM_CACHE, ("conll2012", "litbank", "preco", "corefud"), 10000, False)


def _train_idx(docs: list, subset: str) -> list:
    # indices of training docs honoring subset rules: drop preco for nopreco; cap preco docs
    # for cp8k/all8k. Keeps the master list (and thus ctx positions) intact.
    cap = _PRECO_CAP.get(subset)
    out, preco = [], 0
    for i, d in enumerate(docs):
        if d["split"] != "train":
            continue
        name = d["name"]
        if subset == "nopreco" and name.startswith("preco/"):
            continue
        if subset == "p2c" and not name.startswith("preco/"):
            continue
        if cap is not None and name.startswith("preco/"):
            if preco >= cap:
                continue
            preco += 1
        out.append(i)
    return out


def _filter_docs(docs: list, subset: str) -> list:
    # Only prefix subsets may physically shrink the master list: conll is the cache's
    # leading block, so dropping the rest keeps positions 0..n_conll aligned with the
    # ctx .npy files. nopreco removes an *interior* block (order is conll,litbank,preco,
    # corefud), so the master list is left intact to preserve ctx alignment and preco is
    # excluded from training instead (see _train_idx).
    if subset == "conll":
        return [d for d in docs if d["name"].startswith("conll2012/")]
    return docs


def _win_names(window: int, subset: str = "all", channel: str = "both", sent_aligned: bool = False) -> tuple:
    # window-tagged artifact names so K=128/256/512 never collide or reuse each other:
    # (ctx cache dir, Stage A head base, Stage B matcher, Stage A clusters cache).
    # ctx dir is NOT subset-tagged: conll docs are the cache's stable prefix, so a
    # subset run reuses the same ctx files. Heads/matchers/clusters get the subset tag.
    # channel (bge/ctx) tags single-channel ablations so they never collide with the
    # both-channels artifacts; ctx dir is channel-agnostic (same cached features either way).
    # sent_aligned tags the sentence-packed-window variant; its ctx is encoded with
    # different chunk boundaries, so it gets its own ctx dir as well as head/matcher/clusters.
    # ENCODER_TAG separates a non-default contextual encoder's artifacts (e.g. SpanBERT) from RoBERTa's
    sa = "_sent" if sent_aligned else ""
    t = f"_k{window}" + _SUBSET_TAG[subset] + ("" if channel == "both" else f"_{channel}") + sa + ENCODER_TAG
    ctx = SPAN_CTX_CP8K if subset in ("cp8k", "p2c") else SPAN_CTX_CACHE
    return (
        ctx.with_name(ctx.name + f"_k{window}{sa}{ENCODER_TAG}"),
        f"stage2_frozen_head{t}",
        f"stage2_cluster_matcher{t}.pt",
        f"stage_a_clusters_cache{t}.pkl",
    )


def _win_ids(d: dict, window: int) -> np.ndarray:
    # Per-mention window id. Sentence-aligned runs precompute d["win_ids"] (see
    # _apply_sentence_windows); everything else falls back to fixed-K blocks tok_pos // window.
    wi = d.get("win_ids")
    return wi if wi is not None else d["tok_pos"] // window


def _sentence_chunks(sent_sub_offsets: np.ndarray, sent_sub_lengths: np.ndarray, window: int) -> list[tuple[int, int]]:
    # Greedily pack whole sentences into chunks of <= `window` subtokens, never splitting a
    # sentence. A sentence longer than `window` is split at K (RoBERTa's max length) on its own.
    chunks: list[tuple[int, int]] = []
    cur_s = cur_e = None
    for off, ln in zip(sent_sub_offsets, sent_sub_lengths, strict=True):
        off, end = int(off), int(off) + int(ln)
        if int(ln) > window:
            if cur_s is not None:
                chunks.append((cur_s, cur_e))
                cur_s = cur_e = None
            chunks.extend((s, min(s + window, end)) for s in range(off, end, window))
            continue
        if cur_s is None:
            cur_s, cur_e = off, end
        elif end - cur_s > window:
            chunks.append((cur_s, cur_e))
            cur_s, cur_e = off, end
        else:
            cur_e = end
    if cur_s is not None:
        chunks.append((cur_s, cur_e))
    return chunks


def _apply_sentence_windows(docs: list, window: int) -> None:
    # Compute sentence-packed encoding chunks + per-mention window ids, before precompute drops
    # the sentence offsets. d["win_chunks"] drives encode_document_ctx; d["win_ids"] drives scoring.
    for d in docs:
        off, ln = d.get("sent_sub_offsets"), d.get("sent_sub_lengths")
        if off is None or ln is None:
            continue
        chunks = _sentence_chunks(off, ln, window)
        d["win_chunks"] = chunks
        starts = np.asarray([s for s, _ in chunks], dtype=np.int64)
        d["win_ids"] = np.searchsorted(starts, d["tok_pos"], side="right") - 1


def _doc_structure_generic(sents: list[list[str]], clusters: list[list[tuple]]) -> tuple:
    # clusters: list of clusters, each a list of (sent_idx, start, end) end-exclusive
    offsets, off = [], 0
    for s in sents:
        offsets.append(off)
        off += len(s)
    spans, cluster_id, mention_surfaces, head_lex = [], [], [], []
    for cid, cluster in enumerate(clusters):
        for si, a, b in cluster:
            if si >= len(sents) or a >= len(sents[si]):
                continue
            spans.append((si, a, b))
            cluster_id.append(cid)
            mention_surfaces.append(" ".join(sents[si][a:b]))
            # No parse here: head proxy = rightmost token; lemma falls back to surface
            head = sents[si][min(b, len(sents[si])) - 1].lower()
            head_lex.append((head, head))
    return sents, offsets, spans, cluster_id, mention_surfaces, head_lex


def _clusters_litbank(sample: dict) -> list[list[tuple]]:
    # coref_chains: list of chains; each span [sent_idx, start, end] end-inclusive -> convert to exclusive
    return [[(si, a, b + 1) for si, a, b in chain] for chain in sample["coref_chains"]]


def _clusters_preco(sample: dict) -> list[list[tuple]]:
    # mention_clusters: same format as conll (end-exclusive)
    return [[(si, a, b) for si, a, b in cluster] for cluster in sample["mention_clusters"]]


def _clusters_corefud(sample: dict) -> list[list[tuple]]:
    # coref_entities: list of entities; each mention has sent_id and span '1-2' (1-based, inclusive)
    sent_id_to_idx = {sent["sent_id"]: i for i, sent in enumerate(sample["sentences"])}
    clusters = []
    for entity in sample["coref_entities"]:
        cluster = []
        for mention in entity:
            idx = sent_id_to_idx.get(mention["sent_id"])
            if idx is None:
                continue
            span_str = mention["span"]
            parts = span_str.split("-")
            a = int(parts[0]) - 1  # 0-based
            b = int(parts[-1])  # exclusive end (1-based inclusive -> 0-based exclusive = same number)
            cluster.append((idx, a, b))
        if len(cluster) >= 2:
            clusters.append(cluster)
    return clusters


def _word_to_subtok(words: list, tokenizer) -> tuple:
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    word_ids = enc.word_ids()
    w2s: dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in w2s:
            w2s[wid] = pos
    return np.asarray(enc["input_ids"], dtype=np.int64), w2s


def _raw_from_doc(
    name: str,
    split: str,
    sents: list,
    offsets: list,
    spans: list,
    cluster_id: list,
    mention_surfaces: list,
    head_lex: list,
    tokenizer,
) -> tuple | None:
    if len(spans) < 2:
        return None
    words_flat = [w for s in sents for w in s]
    content_ids, w2s = _word_to_subtok(words_flat, tokenizer)
    last = len(content_ids) - 1
    # Sentence subtoken offsets and lengths: for sent_mean computation
    sent_sub_off, sent_sub_len = [], []
    for si, sent in enumerate(sents):
        gw_start = offsets[si]
        gw_end = offsets[si] + len(sent) - 1
        ss = w2s.get(gw_start, min(gw_start, last))
        se = w2s.get(gw_end + 1, last + 1)
        sent_sub_off.append(ss)
        sent_sub_len.append(max(1, se - ss))
    span_sub, mention_sent_off, mention_sent_len = [], [], []
    for si, a, b in spans:
        gw_start, gw_end = offsets[si] + a, offsets[si] + b - 1
        ss = w2s.get(gw_start, min(gw_start, last))
        se = min(max(w2s.get(gw_end + 1, len(content_ids)) - 1, ss), last)
        span_sub.append((ss, se))
        mention_sent_off.append(sent_sub_off[si])
        mention_sent_len.append(sent_sub_len[si])
    order = sorted(range(len(spans)), key=lambda k: span_sub[k])
    return (
        name,
        split,
        sents,
        [spans[k] for k in order],
        [cluster_id[k] for k in order],
        [mention_surfaces[k] for k in order],
        content_ids,
        [span_sub[k] for k in order],
        [mention_sent_off[k] for k in order],
        [mention_sent_len[k] for k in order],
    )


def _retokenize(d: dict, tokenizer) -> None:
    # Re-derive subtoken ids + span/sentence subtoken offsets for the current contextual tokenizer,
    # in place, reusing the cached (encoder-independent) sentences/spans/BGE/clusters — no BGE
    # re-encode, no spaCy. A different tokenizer can reorder mentions by subtoken position, so
    # spans/cluster_id/mention_bge are re-sorted to match.
    sents = d["sentences"]
    offsets, off = [], 0
    for s in sents:
        offsets.append(off)
        off += len(s)
    words_flat = [w for s in sents for w in s]
    content_ids, w2s = _word_to_subtok(words_flat, tokenizer)
    last = len(content_ids) - 1
    sent_sub_off, sent_sub_len = [], []
    for si, sent in enumerate(sents):
        ss = w2s.get(offsets[si], min(offsets[si], last))
        se = w2s.get(offsets[si] + len(sent), last + 1)
        sent_sub_off.append(ss)
        sent_sub_len.append(max(1, se - ss))
    span_sub, m_off, m_len = [], [], []
    for si, a, b in d["spans"]:
        ss = w2s.get(offsets[si] + a, min(offsets[si] + a, last))
        se = min(max(w2s.get(offsets[si] + b, len(content_ids)) - 1, ss), last)
        span_sub.append((ss, se))
        m_off.append(sent_sub_off[si])
        m_len.append(sent_sub_len[si])
    order = sorted(range(len(span_sub)), key=lambda k: span_sub[k])
    d["content_ids"] = content_ids
    d["spans"] = [d["spans"][k] for k in order]
    d["cluster_id"] = d["cluster_id"][order]
    d["mention_bge"] = d["mention_bge"][order]
    d["span_sub"] = np.asarray([span_sub[k] for k in order], dtype=np.int64)
    d["sent_sub_offsets"] = np.asarray([m_off[k] for k in order], dtype=np.int64)
    d["sent_sub_lengths"] = np.asarray([m_len[k] for k in order], dtype=np.int64)


def _assemble_docs(raw: list, device: str, cache_path) -> list:
    surface_vocab = sorted({surf for r in raw for surf in r[5]})
    print(f"Encoding {len(surface_vocab)} unique mention surfaces with BGE...")
    bge = SentenceTransformer(BGE_MODEL, device=device)
    if device != "cpu":
        bge = bge.half()
    embs = np.asarray(
        bge.encode(surface_vocab, normalize_embeddings=True, batch_size=512, show_progress_bar=True), dtype=np.float16
    )
    surf2bge = {s: embs[i] for i, s in enumerate(surface_vocab)}
    del bge
    docs = []
    for (
        name,
        split,
        sents,
        spans,
        cluster_id,
        mention_surfaces,
        content_ids,
        span_sub,
        sent_sub_offsets,
        sent_sub_lengths,
    ) in raw:
        docs.append(
            {
                "name": name,
                "split": split,
                "sentences": sents,
                "spans": spans,
                "cluster_id": np.asarray(cluster_id, dtype=np.int64),
                "content_ids": content_ids,
                "span_sub": np.asarray(span_sub, dtype=np.int64),
                "sent_sub_offsets": np.asarray(sent_sub_offsets, dtype=np.int64),
                "sent_sub_lengths": np.asarray(sent_sub_lengths, dtype=np.int64),
                "mention_bge": np.stack([surf2bge[s] for s in mention_surfaces]).astype(np.float16),
            }
        )
    with cache_path.open("wb") as f:
        pickle.dump(docs, f)
    print(f"Cached {len(docs)} docs to {cache_path.name}")
    return docs


def build_docs(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    datasets: tuple = ("conll2012", "litbank", "preco", "corefud"),
    nom_cache: Path = NOM_CACHE,
    preco_n: int = PRECO_SUBSAMPLE,
) -> list:
    if nom_cache.exists():
        with nom_cache.open("rb") as f:
            docs = pickle.load(f)
        if ENCODER_TAG:  # cache is tokenized for the default encoder; re-derive subtokens for this one
            tokenizer = load_tokenizer()
            for d in tqdm(docs, desc="re-tokenizing for current encoder"):
                _retokenize(d, tokenizer)
        print(f"Loaded cached nominal docs: {len(docs)}")
        return docs

    tokenizer = load_tokenizer()
    raw = []

    # --- CoNLL-2012 (all splits) ---
    for split in CONLL_SPLITS:
        ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
        for sample in tqdm(ds, total=len(ds), desc=f"conll2012/{split}"):
            clusters = [[(si, a, b) for si, a, b in cl] for cl in sample["mention_clusters"]]
            sents_str, offsets, spans, cluster_id, mention_surfaces, head_lex = _doc_structure_generic(
                sample["sentences"], clusters
            )
            rec = _raw_from_doc(
                f"conll2012/{sample['doc_id']}#{len(raw)}",
                split,
                sents_str,
                offsets,
                spans,
                cluster_id,
                mention_surfaces,
                head_lex,
                tokenizer,
            )
            if rec:
                raw.append(rec)

    # --- LitBank (train/validation/test) ---
    if "litbank" in datasets:
        for split in ("train", "validation", "test"):
            ds = load_from_disk(str(DATA_DIR / "litbank"))[split]
            for sample in tqdm(ds, desc=f"litbank/{split}"):
                sents = sample["sentences"]
                clusters = _clusters_litbank(sample)
                sents_list = [list(s) if not isinstance(s, list) else s for s in sents]
                sents_str, offsets, spans, cluster_id, mention_surfaces, head_lex = _doc_structure_generic(
                    sents_list, clusters
                )
                rec = _raw_from_doc(
                    f"litbank/{sample['doc_name']}#{len(raw)}",
                    split,
                    sents_str,
                    offsets,
                    spans,
                    cluster_id,
                    mention_surfaces,
                    head_lex,
                    tokenizer,
                )
                if rec:
                    raw.append(rec)

    # --- PreCo (train subsampled, validation) ---
    preco_ds = load_from_disk(str(DATA_DIR / "preco"))
    preco_train = list(preco_ds["train"])
    random.seed(RANDOM_SEED)
    random.shuffle(preco_train)
    for sample in tqdm(preco_train[:preco_n], desc="preco/train"):
        sents = sample["sentences"]
        clusters = _clusters_preco(sample)
        sents_str, offsets, spans, cluster_id, mention_surfaces, head_lex = _doc_structure_generic(sents, clusters)
        rec = _raw_from_doc(
            f"preco/{sample['id']}#{len(raw)}",
            "train",
            sents_str,
            offsets,
            spans,
            cluster_id,
            mention_surfaces,
            head_lex,
            tokenizer,
        )
        if rec:
            raw.append(rec)
    for sample in tqdm(preco_ds["validation"], desc="preco/validation"):
        sents = sample["sentences"]
        clusters = _clusters_preco(sample)
        sents_str, offsets, spans, cluster_id, mention_surfaces, head_lex = _doc_structure_generic(sents, clusters)
        rec = _raw_from_doc(
            f"preco/{sample['id']}#{len(raw)}",
            "validation",
            sents_str,
            offsets,
            spans,
            cluster_id,
            mention_surfaces,
            head_lex,
            tokenizer,
        )
        if rec:
            raw.append(rec)

    # --- CorefUD (train/validation) ---
    if "corefud" in datasets:
        for split in ("train", "validation"):
            ds = load_from_disk(str(DATA_DIR / "corefud"))[split]
            for sample in tqdm(ds, desc=f"corefud/{split}"):
                sents = [[t["form"] for t in sent["tokens"]] for sent in sample["sentences"]]
                clusters = _clusters_corefud(sample)
                sents_str, offsets, spans, cluster_id, mention_surfaces, head_lex = _doc_structure_generic(
                    sents, clusters
                )
                rec = _raw_from_doc(
                    f"corefud/{sample['doc_id']}#{len(raw)}",
                    split,
                    sents_str,
                    offsets,
                    spans,
                    cluster_id,
                    mention_surfaces,
                    head_lex,
                    tokenizer,
                )
                if rec:
                    raw.append(rec)

    # --- Encode all unique mention surfaces with BGE ---
    return _assemble_docs(raw, device, nom_cache)


def gather_spans_np(ctx: np.ndarray, span_sub: np.ndarray) -> np.ndarray:
    # Returns (M, 2, CTX_DIM) float16: (ctx_start, ctx_end) per mention.
    M = len(span_sub)
    out = np.zeros((M, 2, ctx.shape[1]), dtype=np.float16)
    for k, (s, e) in enumerate(span_sub):
        out[k, 0] = ctx[int(s)]
        out[k, 1] = ctx[int(e)]
    return out


def _ctx_path(i: int, cache_dir: Path = SPAN_CTX_CACHE) -> Path:
    return cache_dir / f"{i:06d}.npy"


def _single_ctx_paths(base: Path) -> tuple:
    # one concatenated (sum_M, 2, CTX_DIM) float16 array + an (n_docs+1,) int64 offset index
    return base.with_name(base.name + ".npy"), base.with_name(base.name + "_off.npy")


def _pop_built(d: dict) -> None:
    for k in ("content_ids", "span_sub", "sent_sub_offsets", "sent_sub_lengths"):
        d.pop(k, None)


def _n_windows(d: dict, window: int) -> int:
    ch = d.get("win_chunks")
    if ch is not None:
        return len(ch)
    n = len(d["content_ids"])
    return max(1, (n + window - 1) // window)


def _ctx_encode_groups(indices: list[int], docs: list, window: int) -> list[list[int]]:
    # Greedily pack consecutive docs into groups whose total window count <= CTX_ENCODE_MAX_WINDOWS,
    # so each encoder forward is GPU-saturating instead of one doc's worth of windows. A single doc
    # exceeding the cap is its own group.
    groups: list[list[int]] = []
    cur: list[int] = []
    cnt = 0
    for i in indices:
        w = _n_windows(docs[i], window)
        if cur and cnt + w > CTX_ENCODE_MAX_WINDOWS:
            groups.append(cur)
            cur, cnt = [], 0
        cur.append(i)
        cnt += w
    if cur:
        groups.append(cur)
    return groups


def _precompute_span_ctx_single(encoder, docs, cls_id, sep_id, device, base: Path, window: int) -> None:
    # Single-file variant: all docs' span ctx packed into one .npy (+ offset index), memory-mapped from
    # disk (mmap_mode="c") so the full array is never RAM-resident — per-doc slices page in on access.
    big_p, off_p = _single_ctx_paths(base)
    if big_p.exists() and off_p.exists():
        big = np.load(big_p, mmap_mode="c")  # memory-mapped; per-doc slices are paged views into it
        off = np.load(off_p)
        for i, d in enumerate(docs):
            d["ctx_vecs"] = big[off[i] : off[i + 1]]
            _pop_built(d)
        print(f"Loaded cached span ctx (single file): {len(docs)} docs, {big.shape[0]} mentions")
        return
    base.parent.mkdir(parents=True, exist_ok=True)
    total = sum(len(d["span_sub"]) for d in docs)
    big = np.lib.format.open_memmap(big_p, mode="w+", dtype=np.float16, shape=(total, 2, CTX_DIM))
    off = np.zeros(len(docs) + 1, dtype=np.int64)
    encoder.eval()
    pos = 0
    groups = _ctx_encode_groups(list(range(len(docs))), docs, window)
    with torch.inference_mode():
        for group in tqdm(groups, desc="span ctx (single)"):
            ctxs = encode_docs_ctx_batched(
                [docs[i]["content_ids"] for i in group],
                [docs[i].get("win_chunks") for i in group],
                encoder, cls_id, sep_id, device, window,
            )
            for i, ctx in zip(group, ctxs, strict=True):
                cv = gather_spans_np(ctx.float().cpu().numpy(), docs[i]["span_sub"])
                n = cv.shape[0]
                big[pos : pos + n] = cv
                off[i] = pos
                docs[i]["ctx_vecs"] = big[pos : pos + n]
                pos += n
                _pop_built(docs[i])
    off[len(docs)] = pos
    big.flush()
    np.save(off_p, off)
    print(f"Cached span ctx (single file) → {big_p.name}: {pos} mentions, {big.nbytes / 1e9:.2f} GB")


def load_span_ctx_single(docs: list, base: Path) -> None:
    # populate d["ctx_vecs"] from the single-file cache (used by Stage B, which re-loads from disk)
    big_p, off_p = _single_ctx_paths(base)
    big = np.load(big_p, mmap_mode="c")  # memory-mapped; per-doc slices are paged views into it
    off = np.load(off_p)
    for i, d in enumerate(docs):
        if "ctx_vecs" not in d:
            d["ctx_vecs"] = big[off[i] : off[i + 1]]


def precompute_span_ctx(
    encoder, docs, cls_id, sep_id, device, cache_dir: Path = SPAN_CTX_CACHE, window: int = CONTENT, single: bool = False
) -> None:
    # Writes each doc's (M, 2, CTX_DIM) float16 ctx_vecs (start, end) to an individual .npy file immediately.
    # Never accumulates more than one doc in RAM. Resumable: skips already-written files.
    # single=True packs everything into one .npy instead (see _precompute_span_ctx_single).
    if single:
        _precompute_span_ctx_single(encoder, docs, cls_id, sep_id, device, cache_dir, window)
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for i, d in enumerate(docs):
        if _ctx_path(i, cache_dir).exists():
            d["ctx_vecs"] = np.load(_ctx_path(i, cache_dir)).astype(np.float16)
            _pop_built(d)
        else:
            todo.append(i)
    if not todo:
        print(f"Loaded cached span ctx: {len(docs)} docs")
        return
    if len(todo) < len(docs):
        print(f"Resuming span ctx precompute: {len(docs) - len(todo)}/{len(docs)} already cached...")
    encoder.eval()
    groups = _ctx_encode_groups(todo, docs, window)
    with torch.inference_mode():
        for group in tqdm(groups, desc="span ctx precompute"):
            ctxs = encode_docs_ctx_batched(
                [docs[i]["content_ids"] for i in group],
                [docs[i].get("win_chunks") for i in group],
                encoder, cls_id, sep_id, device, window,
            )
            for i, ctx in zip(group, ctxs, strict=True):
                cv = gather_spans_np(ctx.float().cpu().numpy(), docs[i]["span_sub"])
                np.save(_ctx_path(i, cache_dir), cv)
                docs[i]["ctx_vecs"] = cv.astype(np.float16)
                _pop_built(docs[i])
    print(f"Cached span ctx to {cache_dir.name}/")


def precompute_full_ctx(encoder, docs, cls_id, sep_id, device, cache_dir: Path, window: int = CONTENT) -> None:
    # Like precompute_span_ctx but writes the FULL (n_subtokens, CTX_DIM) token reps per doc
    # (mention detection needs every token, not just gathered mention endpoints). Keeps span_sub
    # (the gold detection targets); only content_ids is dropped after encoding.
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder.eval()
    with torch.inference_mode():
        for i, d in enumerate(tqdm(docs, desc="full ctx precompute")):
            p = cache_dir / f"{i:06d}.npy"
            if p.exists():
                d["full_ctx"] = np.load(p).astype(np.float16)
                d.pop("content_ids", None)
                continue
            ctx = (
                encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device, window, spans=d.get("win_chunks"))
                .float().cpu().numpy()
            )
            np.save(p, ctx.astype(np.float16))
            d["full_ctx"] = ctx.astype(np.float16)
            d.pop("content_ids", None)
    print(f"Cached full ctx to {cache_dir.name}/")


def _key_clusters(d: dict, window: int | None = None) -> list:
    if window is not None:
        # gold for Stage A: split each entity cluster at the same window boundaries used to score
        win_ids = _win_ids(d, window)
        cid = d["cluster_id"]
        out: list = []
        for c in np.unique(cid):
            members = np.where(cid == c)[0]
            for w in np.unique(win_ids[members]):
                comp = members[win_ids[members] == w].tolist()
                if len(comp) >= 2:
                    out.append(comp)
        return [[d["spans"][i] for i in comp] for comp in out]
    return [[d["spans"][i] for i in np.where(d["cluster_id"] == c)[0]] for c in np.unique(d["cluster_id"])]


_PRONOUNS = frozenset(
    {
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "yourselves",
        "themselves",
        "this",
        "that",
        "these",
        "those",
        "who",
        "whom",
        "whose",
        "which",
        "what",
        "there",
        "here",
    }
)


def _build_lexical(docs: list) -> None:
    # per-mention lowercased token tuples + doc-local IDF, reconstructed from sentences+spans
    # (spans are (sent_idx, start, end)). Independent lexical-identity channel for Stage B.
    for d in docs:
        if "mention_tokens" in d:
            continue
        sents = d["sentences"]
        toks = [tuple(w.lower() for w in sents[si][s:e]) for (si, s, e) in d["spans"]]
        df: dict = {}
        for m in toks:
            for t in set(m):
                df[t] = df.get(t, 0) + 1
        n = max(len(toks), 1)
        d["mention_tokens"] = toks
        d["mention_idf"] = {t: math.log(n / (1 + c)) + 1.0 for t, c in df.items()}


def _is_subseq(a: tuple, b: tuple) -> bool:
    if not a or len(a) > len(b):
        return False
    return any(b[k : k + len(a)] == a for k in range(len(b) - len(a) + 1))


def _content(m: tuple) -> bool:
    # a content mention surface (not a bare pronoun) for exact-match / containment features
    return not (len(m) == 1 and m[0] in _PRONOUNS)
