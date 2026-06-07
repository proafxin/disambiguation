import csv
import datetime
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import spacy
import spacy.tokens
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from disambiguation.conll_scorer import conll_f1, conll_f1_by_type, write_conll
from disambiguation.paths import (
    DATA_DIR,
    MODELS_DIR,
    NOM_CACHE,
    NOM_CACHE_CP8K,
    SPACY_TRF_DIR,
    SPAN_CTX_CACHE,
    SPAN_CTX_CP8K,
    TENSORBOARD_DIR,
)
from disambiguation.stage2_context_encoder import (
    CONTENT,
    CTX_DIM,
    AntecedentScorer,
    ClusterMatcher,
    ContextEncoder,
    MentionMatcher,
    _uf_find,
    cluster_match_loss,
    compute_mention_ctx_vecs,
    decode_antecedents,
    decode_cluster_matches,
    encode_document_ctx,
    load_tokenizer,
    mll_loss,
)

BGE_MODEL = "BAAI/bge-large-en-v1.5"
CONLL_SPLITS = ["train", "validation", "test"]
PRECO_SUBSAMPLE = 8000  # ctx_vecs held as float16 in RAM; conll+preco@8k fits ~4.4 GB fully resident
RANDOM_SEED = 42


_SUBSET_TAG = {"all": "", "conll": "_conllonly", "nopreco": "_nopreco", "cp8k": "_cp8k", "all8k": "_all8k"}
_PRECO_CAP = {"cp8k": 8000, "all8k": 8000}  # cap preco *training* docs (cache may hold more)


def _data_cfg(subset: str) -> tuple:
    # (nom_cache, datasets, preco_n, single_ctx) for building/loading this subset's data.
    # cp8k is a self-contained build (conll+preco only) with one single-file ctx. all8k reuses
    # the existing all-4 per-doc cache (10k preco) and caps training to 8k preco (no rebuild).
    if subset == "cp8k":
        return (NOM_CACHE_CP8K, ("conll2012", "preco"), PRECO_SUBSAMPLE, True)
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


def _win_names(window: int, subset: str = "all") -> tuple:
    # window-tagged artifact names so K=128/256/512 never collide or reuse each other:
    # (ctx cache dir, Stage A head base, Stage B matcher, Stage A clusters cache).
    # ctx dir is NOT subset-tagged: conll docs are the cache's stable prefix, so a
    # subset run reuses the same ctx files. Heads/matchers/clusters get the subset tag.
    t = f"_k{window}" + _SUBSET_TAG[subset]
    ctx = SPAN_CTX_CP8K if subset == "cp8k" else SPAN_CTX_CACHE
    return (
        ctx.with_name(ctx.name + f"_k{window}"),
        f"stage2_frozen_head{t}",
        f"stage2_cluster_matcher{t}.pt",
        f"stage_a_clusters_cache{t}.pkl",
    )


def _doc_structure_conll(sample: dict, spacy_doc: spacy.tokens.Doc) -> tuple:
    # Returns (sents, offsets, spans, cluster_id, mention_surfaces)
    # spans: (sent_idx, start, end) end-exclusive; mention_surfaces: full span text for BGE
    sents = sample["sentences"]
    offsets, off = [], 0
    for s in sents:
        offsets.append(off)
        off += len(s)
    spacy_sents = list(spacy_doc.sents)
    spans, cluster_id, mention_surfaces, head_lex = [], [], [], []
    for cid, cluster in enumerate(sample["mention_clusters"]):
        for si, a, b in cluster:
            if si >= len(spacy_sents):
                continue
            sent_span = spacy_sents[si]
            if a >= len(sent_span):
                continue
            head_tok = sent_span[a : min(b, len(sent_span))].root
            head_local = head_tok.i - sent_span.start
            if head_local >= len(sents[si]):
                continue
            spans.append((si, a, b))
            cluster_id.append(cid)
            mention_surfaces.append(" ".join(sents[si][a:b]))
            head_lex.append((head_tok.lemma_, head_tok.lower_))
    return sents, offsets, spans, cluster_id, mention_surfaces, head_lex


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
    gold_clusters: list | None = None,
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
        gold_clusters,
    )


STAGE0_SPANS_CACHE = DATA_DIR / "stage0_predicted_spans.pkl"
NOM_CACHE_PRED = NOM_CACHE.with_name("stage2_conll_predicted_v1.pkl")
SPAN_CTX_CACHE_PRED = SPAN_CTX_CACHE.with_name(SPAN_CTX_CACHE.name + "_pred")


def _assemble_docs(raw: list, device: str, cache_path) -> list:
    surface_vocab = sorted({surf for r in raw for surf in r[5]})
    print(f"Encoding {len(surface_vocab)} unique mention surfaces with BGE...")
    bge = SentenceTransformer(BGE_MODEL, device=device)
    embs = np.asarray(
        bge.encode(surface_vocab, normalize_embeddings=True, batch_size=512, show_progress_bar=True), dtype=np.float32
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
        gold_clusters,
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
                "mention_bge": np.stack([surf2bge[s] for s in mention_surfaces]).astype(np.float32),
                "gold_clusters": gold_clusters,
            }
        )
    with cache_path.open("wb") as f:
        pickle.dump([{**d, "mention_bge": d["mention_bge"].astype(np.float16)} for d in docs], f)
    print(f"Cached {len(docs)} docs to {cache_path.name}")
    return docs


def _predicted_doc_spans() -> dict:
    # Map the detector's flat per-sentence prediction cache back to CoNLL docs.
    # Returns {(split, doc_index): {sent_idx: [(a, b_exclusive), ...]}}.
    with STAGE0_SPANS_CACHE.open("rb") as f:
        cache = pickle.load(f)
    by_split: dict[str, list] = {}
    for r in cache:
        by_split.setdefault(r["split"], []).append(r)
    out: dict = {}
    for split in CONLL_SPLITS:
        ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
        entries = by_split.get(split, [])
        ci = 0
        for di, sample in enumerate(ds):
            doc_map: dict[int, list] = {}
            for si, words in enumerate(sample["sentences"]):
                if len(words) < 1:
                    continue
                e = entries[ci]
                ci += 1
                if list(e["tokens"]) != list(words):
                    raise ValueError(f"predicted-cache misalignment at {split} doc {di} sent {si}")
                doc_map[si] = [(int(a), int(b) + 1) for a, b in e["pred_spans"]]  # inclusive -> exclusive
            out[split, di] = doc_map
    return out


def _ner_id(tok) -> int:
    e = tok.ent_type_
    if e == "PERSON":
        return 1
    if e in ("ORG", "NORP"):
        return 2
    if e in ("GPE", "LOC", "FAC"):
        return 3
    return 4 if e else 0


def _morph_num_id(tok) -> int:
    n = tok.morph.get("Number")
    return 1 if "Sing" in n else (2 if "Plur" in n else 0)


def _morph_gen_id(tok) -> int:
    g = tok.morph.get("Gender")
    if "Masc" in g:
        return 1
    if "Fem" in g:
        return 2
    if "Neut" in g:
        return 3
    return 0


def _doc_structure_conll_predicted(sample: dict, spacy_doc: spacy.tokens.Doc, pred_map: dict) -> tuple:
    # Like _doc_structure_conll but spans come from the detector; cluster_id is joined from gold by
    # exact coordinate (unmatched predictions -> own singleton). gold_clusters kept for the eval KEY.
    sents = sample["sentences"]
    offsets, off = [], 0
    for s in sents:
        offsets.append(off)
        off += len(s)
    spacy_sents = list(spacy_doc.sents)
    gold2cluster: dict[tuple, int] = {}
    gold_clusters: list = []
    for cid, cluster in enumerate(sample["mention_clusters"]):
        members = [(int(si), int(a), int(b)) for si, a, b in cluster]
        for key in members:
            gold2cluster[key] = cid
        gold_clusters.append(members)
    next_singleton = len(sample["mention_clusters"])
    spans, cluster_id, mention_surfaces, head_lex = [], [], [], []
    for si in sorted(pred_map):
        if si >= len(spacy_sents) or si >= len(sents):
            continue
        sent_span = spacy_sents[si]
        for a, b in pred_map[si]:  # b exclusive
            if a >= len(sents[si]) or a >= len(sent_span):
                continue
            head_tok = sent_span[a : min(b, len(sent_span))].root
            spans.append((si, a, b))
            key = (si, a, b)
            if key in gold2cluster:
                cluster_id.append(gold2cluster[key])
            else:
                cluster_id.append(next_singleton)
                next_singleton += 1
            mention_surfaces.append(" ".join(sents[si][a:b]))
            head_lex.append(
                (head_tok.lemma_, head_tok.lower_, _ner_id(head_tok), _morph_num_id(head_tok), _morph_gen_id(head_tok))
            )
    return sents, offsets, spans, cluster_id, mention_surfaces, head_lex, gold_clusters


def build_conll_predicted_docs(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> list:
    if NOM_CACHE_PRED.exists():
        with NOM_CACHE_PRED.open("rb") as f:
            docs = pickle.load(f)
        for d in docs:
            d["mention_bge"] = d["mention_bge"].astype(np.float32)
        print(f"Loaded cached predicted-mention docs: {len(docs)}")
        return docs
    vocab = spacy.blank("en").vocab
    tokenizer = load_tokenizer()
    pred_all = _predicted_doc_spans()
    raw = []
    for split in CONLL_SPLITS:
        ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
        db = spacy.tokens.DocBin().from_disk(SPACY_TRF_DIR / f"conll2012_{split}.spacy")
        for di, (sample, sdoc) in enumerate(
            tqdm(zip(ds, db.get_docs(vocab), strict=True), total=len(ds), desc=f"conll2012-pred/{split}")
        ):
            sents, offsets, spans, cluster_id, mention_surfaces, head_lex, gold_clusters = (
                _doc_structure_conll_predicted(sample, sdoc, pred_all[split, di])
            )
            rec = _raw_from_doc(
                f"conll2012/{sample['doc_id']}#{len(raw)}",
                split,
                sents,
                offsets,
                spans,
                cluster_id,
                mention_surfaces,
                head_lex,
                tokenizer,
                gold_clusters,
            )
            if rec:
                raw.append(rec)
    return _assemble_docs(raw, device, NOM_CACHE_PRED)


def build_docs(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    datasets: tuple = ("conll2012", "litbank", "preco", "corefud"),
    nom_cache: Path = NOM_CACHE,
    preco_n: int = PRECO_SUBSAMPLE,
) -> list:
    if nom_cache.exists():
        with nom_cache.open("rb") as f:
            docs = pickle.load(f)
        for d in docs:
            d["mention_bge"] = d["mention_bge"].astype(np.float32)
        print(f"Loaded cached nominal docs: {len(docs)}")
        return docs

    vocab = spacy.blank("en").vocab
    tokenizer = load_tokenizer()
    raw = []

    # --- CoNLL-2012 (all splits) ---
    for split in CONLL_SPLITS:
        ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
        db = spacy.tokens.DocBin().from_disk(SPACY_TRF_DIR / f"conll2012_{split}.spacy")
        for sample, sdoc in tqdm(zip(ds, db.get_docs(vocab), strict=True), total=len(ds), desc=f"conll2012/{split}"):
            sents, offsets, spans, cluster_id, mention_surfaces, head_lex = _doc_structure_conll(sample, sdoc)
            rec = _raw_from_doc(
                f"conll2012/{sample['doc_id']}#{len(raw)}",
                split,
                sents,
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


def _precompute_span_ctx_single(encoder, docs, cls_id, sep_id, device, base: Path, window: int) -> None:
    # Single-file variant: all docs' span ctx packed into one .npy (+ offset index), held fully in RAM.
    big_p, off_p = _single_ctx_paths(base)
    if big_p.exists() and off_p.exists():
        big = np.load(big_p)  # fully resident in RAM; per-doc slices are views into it
        off = np.load(off_p)
        for i, d in enumerate(docs):
            d["ctx_vecs"] = big[off[i] : off[i + 1]]
            _pop_built(d)
        print(f"Loaded cached span ctx (single file): {len(docs)} docs, {big.shape[0]} mentions")
        return
    base.parent.mkdir(parents=True, exist_ok=True)
    total = sum(len(d["span_sub"]) for d in docs)
    big = np.zeros((total, 2, CTX_DIM), dtype=np.float16)
    off = np.zeros(len(docs) + 1, dtype=np.int64)
    encoder.eval()
    pos = 0
    with torch.inference_mode():
        for i, d in enumerate(tqdm(docs, desc="span ctx (single)")):
            ctx = encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device, window).float().cpu().numpy()
            cv = gather_spans_np(ctx, d["span_sub"])
            big[pos : pos + cv.shape[0]] = cv
            off[i] = pos
            pos += cv.shape[0]
            d["ctx_vecs"] = cv
            _pop_built(d)
    off[len(docs)] = pos
    np.save(big_p, big)
    np.save(off_p, off)
    print(f"Cached span ctx (single file) → {big_p.name}: {pos} mentions, {big.nbytes / 1e9:.2f} GB")


def load_span_ctx_single(docs: list, base: Path) -> None:
    # populate d["ctx_vecs"] from the single-file cache (used by Stage B, which re-loads from disk)
    big_p, off_p = _single_ctx_paths(base)
    big = np.load(big_p)  # fully resident in RAM; per-doc slices are views into it
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
    n_existing = sum(1 for i in range(len(docs)) if _ctx_path(i, cache_dir).exists())
    if n_existing == len(docs):
        for i, d in enumerate(docs):
            d["ctx_vecs"] = np.load(_ctx_path(i, cache_dir)).astype(np.float16)
            d.pop("content_ids", None)
            d.pop("span_sub", None)
            d.pop("sent_sub_offsets", None)
            d.pop("sent_sub_lengths", None)
        print(f"Loaded cached span ctx: {n_existing} docs")
        return
    if n_existing > 0:
        print(f"Resuming span ctx precompute from doc {n_existing}/{len(docs)}...")
    encoder.eval()
    with torch.inference_mode():
        for i, d in enumerate(tqdm(docs, desc="span ctx precompute")):
            p = _ctx_path(i, cache_dir)
            if p.exists():
                d["ctx_vecs"] = np.load(p).astype(np.float16)
                d.pop("content_ids", None)
                d.pop("span_sub", None)
                d.pop("sent_sub_offsets", None)
                d.pop("sent_sub_lengths", None)
                continue
            ctx = encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device, window).float().cpu().numpy()
            ctx_vecs = gather_spans_np(ctx, d["span_sub"])
            np.save(p, ctx_vecs)
            d["ctx_vecs"] = ctx_vecs.astype(np.float16)
            d.pop("content_ids", None)
            d.pop("span_sub", None)
            d.pop("sent_sub_offsets", None)
            d.pop("sent_sub_lengths", None)
    print(f"Cached span ctx to {cache_dir.name}/")


def doc_scores(
    d: dict,
    encoder,
    scorer,
    cls_id,
    sep_id,
    device,
    all_pairs: bool = False,
    max_mentions: int | None = None,
    window: int = CONTENT,
) -> list[tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    if encoder is None:
        ctx = torch.from_numpy(d["ctx_vecs"]).to(device).float()
    else:
        ctx_doc = encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device, window)
        ctx = compute_mention_ctx_vecs(ctx_doc, d["span_sub"])
    bge = torch.from_numpy(d["mention_bge"]).to(device).float()
    # all_pairs: score every mention against every prior mention in the whole doc (one group);
    # otherwise partition into fixed CONTENT-subtoken windows (Stage A).
    # all_pairs scores the whole doc as one group; but a memory cap (used during fine-tune training,
    # where the live encoder's activations co-reside) falls back to windowing for oversized docs.
    n_ment = len(d["tok_pos"])
    if all_pairs and (max_mentions is None or n_ment <= max_mentions):
        win_ids = np.zeros(n_ment, dtype=np.int64)
    else:
        win_ids = d["tok_pos"] // window
    results = []
    for w in np.unique(win_ids):
        idx = np.where(win_ids == w)[0]
        if len(idx) < 2:
            continue
        w_scores, w_mask = scorer(ctx[idx], bge[idx])
        results.append((w_scores, w_mask, idx))
    return results


def stage_a_batched_loss(scorer, windows: list, device: str) -> torch.Tensor:
    # windows: list of (ctx (M,2,CTX), bge (M,BGE), cid (M,) long, weight float), all M>=2.
    # One mention_rep + one chunked FFNN over the global within-window pair list, then a
    # vectorized MLL over a padded (B, maxM, maxM) score tensor. Equivalent to looping
    # mll_loss per window (window-level mean), but batched so the GPU isn't fed one window
    # at a time. Window losses are combined as a dataset-weighted mean.
    sizes = [w[0].shape[0] for w in windows]
    b, max_m = len(windows), max(sizes)
    g = scorer.mention_rep(torch.cat([w[0] for w in windows], 0), torch.cat([w[1] for w in windows], 0))
    gi, gj, li, lj, wid, dist = [], [], [], [], [], []
    off = 0
    for w, m in enumerate(sizes):
        idx = torch.arange(m, device=device)
        ii, jj = (idx.unsqueeze(0) < idx.unsqueeze(1)).nonzero(as_tuple=True)  # ii>jj (antecedent jj)
        gi.append(ii + off)
        gj.append(jj + off)
        li.append(ii)
        lj.append(jj)
        wid.append(torch.full_like(ii, w))
        dist.append(ii - jj)
        off += m
    gi, gj, li, lj, wid, dist = (torch.cat(t) for t in (gi, gj, li, lj, wid, dist))
    pair = torch.empty(gi.shape[0], device=device, dtype=g.dtype)
    for s0 in range(0, gi.shape[0], scorer.chunk):
        sl = slice(s0, s0 + scorer.chunk)
        bucket = torch.bucketize(dist[sl].clamp(min=0), scorer.dist_bounds, right=True)
        feat = torch.cat([g[gi[sl]], g[gj[sl]], scorer.dist_emb(bucket)], dim=-1)
        pair[sl] = scorer.ffnn(feat).squeeze(-1).to(g.dtype)

    neg = torch.finfo(g.dtype).min
    scores = torch.zeros(b, max_m, max_m, device=device, dtype=g.dtype)
    scores[wid, li, lj] = pair
    ar = torch.arange(max_m, device=device)
    size_t = torch.tensor(sizes, device=device)
    valid = ar.unsqueeze(0) < size_t.unsqueeze(1)  # (B, maxM)
    ante = (ar.unsqueeze(1) > ar.unsqueeze(0)).unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)  # [b,i,j]: j<i
    cid = torch.full((b, max_m), -1, device=device, dtype=torch.long)
    for w, win in enumerate(windows):
        cid[w, : sizes[w]] = win[2]
    null = scorer.null_bias
    null_col = null.view(1, 1, 1).expand(b, max_m, 1)
    denom = torch.logsumexp(torch.cat([null_col, scores.masked_fill(~ante, neg)], dim=2), dim=2)  # (B, maxM)
    gold = (cid.unsqueeze(2) == cid.unsqueeze(1)) & ante  # [b,i,j]: same cluster, j<i
    has_gold = gold.any(dim=2)
    num = torch.where(has_gold, torch.logsumexp(scores.masked_fill(~gold, neg), dim=2), null.squeeze().expand(b, max_m))
    per_ment = denom - num  # (B, maxM)
    loss_mask = valid & (ar.unsqueeze(0) >= 1)
    w_loss = (per_ment * loss_mask).sum(1) / loss_mask.sum(1).clamp(min=1)  # (B,) per-window mean
    wt = torch.tensor([w[3] for w in windows], device=device, dtype=w_loss.dtype)
    return (w_loss * wt).sum() / wt.sum()


def _collect_windows(batch: list, device: str, all_pairs: bool, max_mentions, window: int, loss_weights) -> list:
    # frozen path: turn a batch of docs into a flat list of (ctx, bge, cid, weight) windows
    windows = []
    for d in batch:
        ctx = torch.from_numpy(d["ctx_vecs"]).to(device).float()
        bge = torch.from_numpy(d["mention_bge"]).to(device).float()
        cid = torch.from_numpy(d["cluster_id"]).to(device)
        n_ment = len(d["tok_pos"])
        if all_pairs and (max_mentions is None or n_ment <= max_mentions):
            win_ids = np.zeros(n_ment, dtype=np.int64)
        else:
            win_ids = d["tok_pos"] // window
        wt = 1.0 if loss_weights is None else float(loss_weights.get(d["name"].split("/")[0], 1.0))
        for w in np.unique(win_ids):
            idx = np.where(win_ids == w)[0]
            if len(idx) >= 2:
                windows.append((ctx[idx], bge[idx], cid[idx], wt))
    return windows


def run_epoch(
    encoder,
    scorer,
    docs,
    optimizer,
    cls_id,
    sep_id,
    device,
    doc_bs,
    all_pairs: bool = False,
    max_mentions: int | None = None,
    window: int = CONTENT,
    loss_weights: dict | None = None,
) -> float:
    train = optimizer is not None
    total, ndoc = 0.0, 0
    for s in tqdm(range(0, len(docs), doc_bs), desc="train" if train else "val"):
        batch = docs[s : s + doc_bs]
        if train:
            optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            if encoder is None:
                windows = _collect_windows(batch, device, all_pairs, max_mentions, window, loss_weights)
                if not windows:
                    continue
                loss = stage_a_batched_loss(scorer, windows, device)
            else:
                losses = []
                for d in batch:
                    cluster_id = torch.from_numpy(d["cluster_id"]).to(device)
                    for w_scores, w_mask, idx in doc_scores(
                        d, encoder, scorer, cls_id, sep_id, device, all_pairs, max_mentions, window
                    ):
                        losses.append(mll_loss(w_scores, w_mask, cluster_id[idx], scorer.null_bias))
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
        if train:
            loss.backward()
            params = list(scorer.parameters())
            if encoder is not None:
                params += list(encoder.parameters())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        total += loss.item() * len(batch)
        ndoc += len(batch)
    return total / max(ndoc, 1)


def predict_clusters(
    d: dict, encoder, scorer, cls_id, sep_id, device, all_pairs: bool = False, window: int = CONTENT
) -> list:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
        windows = doc_scores(d, encoder, scorer, cls_id, sep_id, device, all_pairs, window=window)
    clusters = []
    null_bias = float(scorer.null_bias.item())
    for w_scores, w_mask, idx in windows:
        groups = decode_antecedents(w_scores.float().cpu().numpy(), w_mask.cpu().numpy(), null_bias)
        clusters.extend([d["spans"][idx[i]] for i in g] for g in groups)
    return clusters


def _key_clusters(d: dict, window: int | None = None) -> list:
    gc = d.get("gold_clusters")
    if gc is not None:
        # predicted-mention docs: KEY is the TRUE gold clusters (doc-level), so missed mentions count.
        return [list(c) for c in gc]
    if window is not None:
        # gold for Stage A: split each entity cluster at fixed window boundaries
        win_ids = d["tok_pos"] // window
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


def eval_conll(
    encoder,
    scorer,
    docs,
    cls_id,
    sep_id,
    device,
    tag,
    type_breakdown=False,
    window: int | None = None,
    all_pairs: bool = False,
) -> dict:
    key_docs = [(d["name"], d["sentences"], _key_clusters(d, window)) for d in docs]
    key_path = MODELS_DIR / f"stage2_{tag}_key.conll"
    write_conll(key_path, key_docs)
    resp_docs = [
        (
            d["name"],
            d["sentences"],
            predict_clusters(d, encoder, scorer, cls_id, sep_id, device, all_pairs, window=window or CONTENT),
        )
        for d in tqdm(docs, desc=f"eval/{tag}")
    ]
    resp_path = MODELS_DIR / f"stage2_{tag}_response.conll"
    write_conll(resp_path, resp_docs)
    result = conll_f1(key_path, resp_path)
    if type_breakdown:
        result["by_type"] = conll_f1_by_type(key_docs, resp_docs, MODELS_DIR)
    return result


def train_stage2(
    finetune: bool = False,
    roberta_lr: float = 2e-5,
    head_lr: float = 1e-3,
    max_epochs: int | None = None,
    patience: int | None = None,
    doc_bs: int | None = None,
    predicted: bool = False,
    all_pairs: bool = False,
    window: int = CONTENT,
    subset: str = "all",
    dropout: float = 0.3,
    weight_decay: float = 0.1,
    loss_weights: dict | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(
        f"\nBuilding Stage 2 nominal data... (finetune={finetune}, predicted={predicted}, window={window}, subset={subset})"
    )
    if predicted:
        print("Evaluation setting: PREDICTED mentions (end-to-end), CoNLL only; KEY = true gold clusters")
    else:
        print("Evaluation setting: gold mentions (not end-to-end; not comparable to predicted-mention systems)")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    nom_cache, datasets, preco_n, single_ctx = _data_cfg(subset)
    docs = (
        build_conll_predicted_docs(device=device)
        if predicted
        else build_docs(device=device, datasets=datasets, nom_cache=nom_cache, preco_n=preco_n)
    )
    docs = _filter_docs(docs, subset)
    for d in docs:  # retain mention subtoken start positions before precompute drops span_sub
        d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    tokenizer = load_tokenizer()
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    conll_val_docs = [d for d in docs if d["split"] == "validation" and d["name"].startswith("conll2012/")]
    conll_test_docs = [d for d in docs if d["split"] == "test" and d["name"].startswith("conll2012/")]
    train_docs = [docs[i] for i in _train_idx(docs, subset)]
    if predicted:
        val_sets = {"conll2012": conll_val_docs}
    else:
        val_sets = {
            "conll2012": conll_val_docs,
            "litbank": [d for d in docs if d["split"] == "validation" and d["name"].startswith("litbank/")],
            "preco": [d for d in docs if d["split"] == "validation" and d["name"].startswith("preco/")],
            "corefud": [d for d in docs if d["split"] == "validation" and d["name"].startswith("corefud/")],
        }

    ctx_dir, frozen_base, _, _ = _win_names(window, subset)
    frozen_name = frozen_base
    if predicted:
        frozen_name += "_pred"
    if all_pairs:
        frozen_name += "_ap"
    scorer = AntecedentScorer(dropout=dropout).to(device)
    encoder = ContextEncoder().to(device)
    if finetune:
        # Head warmup: start fine-tuning from the trained frozen head, not a random one — joint
        # fine-tuning from a random head corrupts the pretrained encoder (measured: -1.9 CoNLL F1).
        warm = MODELS_DIR / f"{frozen_name}.pt"
        if warm.exists():
            scorer.load_state_dict(torch.load(warm, map_location=device)["scorer"])
            print(f"Head warm-started from {warm.name}")
        else:
            print(f"WARNING: no frozen head {warm.name}; fine-tuning from a RANDOM head (expect degradation)")
        encoder.roberta.gradient_checkpointing_enable()
        max_epochs, patience, doc_bs = max_epochs or 3, patience or 2, doc_bs or 1
        optimizer = optim.AdamW(
            [
                {"params": encoder.parameters(), "lr": roberta_lr},
                {"params": list(scorer.parameters()), "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )
        enc_loop, ckpt_path, tag = encoder, MODELS_DIR / f"{frozen_name}_ft.pt", "finetune"
    else:
        for p in encoder.parameters():
            p.requires_grad_(False)
        precompute_span_ctx(
            encoder,
            docs,
            cls_id,
            sep_id,
            device,
            cache_dir=SPAN_CTX_CACHE_PRED if predicted else ctx_dir,
            window=window,
            single=single_ctx,
        )
        del encoder
        if device != "cpu":
            torch.cuda.empty_cache()
        max_epochs, patience, doc_bs = max_epochs or 60, patience or 8, doc_bs or 8
        optimizer = optim.AdamW(list(scorer.parameters()), lr=head_lr, weight_decay=weight_decay)
        enc_loop, ckpt_path, tag = None, MODELS_DIR / f"{frozen_name}.pt", "frozen"
    head_params = sum(p.numel() for p in scorer.parameters())
    print(f"head params: {head_params:,} | doc_bs={doc_bs} epochs={max_epochs}")
    print(f"train docs: {len(train_docs)} | conll val: {len(conll_val_docs)} | conll test: {len(conll_test_docs)}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=head_lr * 0.1)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(TENSORBOARD_DIR / f"stage2_{tag}_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
    best_f1, patience_ctr, disk_best_f1 = -1.0, 0, -1.0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        scorer.load_state_dict(ckpt["scorer"])
        if enc_loop is not None and "encoder" in ckpt:
            enc_loop.load_state_dict(ckpt["encoder"])
        disk_best_f1 = ckpt.get("best_val_f1", -1.0)
        print(f"Warm-started from {ckpt_path.name} (best_val_f1 {disk_best_f1 * 100:.2f}); training fresh from epoch 0")

    # During fine-tune training the live encoder's activations co-reside with all-pairs scoring;
    # cap the all-pairs group so the rare long docs (M up to ~660) window instead of OOM. Eval is uncapped.
    ft_cap = 160 if finetune else None
    for epoch in range(max_epochs):
        print(f"\n=== Stage A (K={window}) Epoch {epoch + 1}/{max_epochs} ===")
        random.shuffle(train_docs)
        if enc_loop is not None:
            enc_loop.train()
        scorer.train()
        tr_loss = run_epoch(
            enc_loop,
            scorer,
            train_docs,
            optimizer,
            cls_id,
            sep_id,
            device,
            doc_bs,
            all_pairs,
            ft_cap,
            window,
            loss_weights,
        )
        scheduler.step()
        if enc_loop is not None:
            enc_loop.eval()
        scorer.eval()
        with torch.inference_mode():
            val_losses = {
                ds: run_epoch(enc_loop, scorer, vdocs, None, cls_id, sep_id, device, doc_bs, all_pairs, ft_cap, window)
                for ds, vdocs in val_sets.items()
                if vdocs
            }
        val_loss = val_losses["conll2012"]
        val_scores = {
            ds: eval_conll(
                enc_loop,
                scorer,
                vdocs,
                cls_id,
                sep_id,
                device,
                f"{tag}_val_{ds}",
                window=window,
                all_pairs=all_pairs,
            )
            for ds, vdocs in val_sets.items()
            if vdocs
        }
        val = val_scores["conll2012"]
        print(f"Loss: {tr_loss:.6f} | Val loss: {val_loss:.6f}")
        for ds, sc in val_scores.items():
            print(
                f"  {ds:10s} loss {val_losses[ds]:.4f}  CoNLL {sc['CoNLL'] * 100:.2f} "
                f"(MUC {sc['muc'] * 100:.2f} B3 {sc['bcub'] * 100:.2f} CEAFe {sc['ceafe'] * 100:.2f})"
            )
        tb.add_scalar("loss/train", tr_loss, epoch + 1)
        tb.add_scalar("loss/val", val_loss, epoch + 1)
        for ds, lo in val_losses.items():
            tb.add_scalar(f"loss/val/{ds}", lo, epoch + 1)
        tb.add_scalar("model/null_bias", scorer.null_bias.item(), epoch + 1)
        for ds, sc in val_scores.items():
            for k in ("CoNLL", "muc", "bcub", "ceafe"):
                tb.add_scalar(f"val_f1/{ds}/{k}", sc[k], epoch + 1)
        if val["CoNLL"] > best_f1 + 1e-4:
            best_f1, patience_ctr = val["CoNLL"], 0
            if best_f1 > disk_best_f1 + 1e-4:
                disk_best_f1 = best_f1
                state = {
                    "scorer": scorer.state_dict(),
                    "best_val_f1": disk_best_f1,
                }
                if enc_loop is not None:
                    state["encoder"] = enc_loop.state_dict()
                torch.save(state, ckpt_path)
                print(f"✓ Best model saved (val CoNLL F1 {disk_best_f1 * 100:.2f})")
            else:
                print(
                    f"Improved this run to {best_f1 * 100:.2f} (all-time best {disk_best_f1 * 100:.2f}; not overwriting)"
                )
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"\n⊘ Early stopping. Best val CoNLL F1 {best_f1 * 100:.2f}")
                break

    print("\n✓ Training complete")
    tb.close()
    ckpt = torch.load(ckpt_path, map_location=device)
    scorer.load_state_dict(ckpt["scorer"])
    scorer.eval()
    if enc_loop is not None and "encoder" in ckpt:
        enc_loop.load_state_dict(ckpt["encoder"])
        enc_loop.eval()
    print("\n=== Test ===")
    metrics = eval_conll(
        enc_loop,
        scorer,
        conll_test_docs,
        cls_id,
        sep_id,
        device,
        f"{tag}_test",
        type_breakdown=True,
        window=window,
        all_pairs=all_pairs,
    )
    print(
        f"CoNLL {metrics['CoNLL'] * 100:.2f} | MUC {metrics['muc'] * 100:.2f} | "
        f"B3 {metrics['bcub'] * 100:.2f} | CEAFe {metrics['ceafe'] * 100:.2f}"
    )
    if "by_type" in metrics:
        print("\nType breakdown (official CoNLL F1 on cluster subsets):")
        for bucket, sc in sorted(metrics["by_type"].items()):
            print(
                f"  {bucket:12s}  CoNLL {sc['CoNLL'] * 100:.2f}  MUC {sc['muc'] * 100:.2f}  "
                f"B3 {sc['bcub'] * 100:.2f}  CEAFe {sc['ceafe'] * 100:.2f}"
            )
    metrics["best_val_f1"] = best_f1
    metrics["setting"] = "gold_mentions"
    metrics["random_seed"] = RANDOM_SEED
    metrics["train_datasets"] = ["conll2012", "litbank", "preco", "corefud"]
    metrics["eval_dataset"] = "conll2012_test"
    final_val_scores = {
        ds: eval_conll(
            enc_loop,
            scorer,
            vdocs,
            cls_id,
            sep_id,
            device,
            f"{tag}_final_val_{ds}",
            window=window,
            all_pairs=all_pairs,
        )
        for ds, vdocs in val_sets.items()
        if vdocs
    }
    for ds, sc in final_val_scores.items():
        print(
            f"  val/{ds:10s} CoNLL {sc['CoNLL'] * 100:.2f} "
            f"(MUC {sc['muc'] * 100:.2f} B3 {sc['bcub'] * 100:.2f} CEAFe {sc['ceafe'] * 100:.2f})"
        )
        metrics[f"val_{ds}"] = sc
    with (MODELS_DIR / f"stage2_eval_metrics_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _stage_a_clusters_for_doc(d: dict, scorer: AntecedentScorer, device: str, window: int = CONTENT) -> dict[int, dict]:
    # Returns {win_id: {"clusters": list[list[int]],  # window-local indices
    #                   "global_idx": np.ndarray}}    # local->global mention index map
    # ctx/bge arrays are NOT stored — loaded from d at training time
    ctx_all = d["ctx_vecs"]
    bge_all = d["mention_bge"]
    tok_pos = d["tok_pos"]
    win_ids = tok_pos // window
    per_window: dict[int, dict] = {}
    with torch.inference_mode():
        for w in np.unique(win_ids):
            global_idx = np.where(win_ids == w)[0]
            ctx_w = torch.from_numpy(ctx_all[global_idx]).to(device).float()
            bge_w = torch.from_numpy(bge_all[global_idx]).to(device).float()
            Mw = len(global_idx)
            if Mw < 2:
                local_clusters = [[i] for i in range(Mw)]
            else:
                w_scores, w_mask = scorer(ctx_w, bge_w)
                groups = decode_antecedents(
                    w_scores.float().cpu().numpy(), w_mask.cpu().numpy(), float(scorer.null_bias.item())
                )
                grouped = {i for g in groups for i in g}
                local_clusters = [list(g) for g in groups]
                local_clusters += [[i] for i in range(Mw) if i not in grouped]
            per_window[int(w)] = {
                "clusters": local_clusters,
                "global_idx": global_idx,
            }
    return per_window


def _cluster_reps(local_cluster: list[int], ctx_w: np.ndarray, bge_w: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    # ctx_w: (Mw, 2, CTX_DIM), bge_w: (Mw, BGE_DIM) — window-local arrays
    idx = np.array(local_cluster)
    return (
        torch.from_numpy(ctx_w[idx]).float(),
        torch.from_numpy(bge_w[idx]).float(),
    )


def _cluster_gold_cids(cluster: list[int], cluster_id: np.ndarray) -> int:
    # Representative cluster_id for a predicted cluster: majority vote.
    cids = cluster_id[np.array(cluster)]
    return int(np.bincount(cids).argmax())


def _precompute_stage_a_clusters(
    docs: list, scorer: AntecedentScorer, device: str, window: int = CONTENT, subset: str = "all"
) -> list[dict[int, dict]]:
    cache_path = MODELS_DIR / _win_names(window, subset)[3]
    if cache_path.exists():
        print("Loading cached Stage A clusters...")
        with cache_path.open("rb") as f:
            return pickle.load(f)
    print("Precomputing Stage A clusters...")
    result = []
    scorer.eval()
    with torch.inference_mode():
        for d in tqdm(docs, desc="stage A clusters"):
            if "ctx_vecs" not in d:
                result.append({})
                continue
            result.append(_stage_a_clusters_for_doc(d, scorer, device, window))
    with cache_path.open("wb") as f:
        pickle.dump(result, f)
    print(f"Cached Stage A clusters to {cache_path.name}")
    return result


def _stage_b_merge_edges(
    d: dict, per_window: dict[int, dict], cluster_matcher: ClusterMatcher, device: str
) -> tuple[list[tuple[int, int]], dict[int, int], int]:
    # Single source of truth for cross-window merges: scores every cross-window
    # cluster pair and returns the accepted merges as (global_a, global_b) edges,
    # the per-window cluster-index offsets, and the total cluster count. Both the
    # cluster decoder and the diagnostics consume these same edges so the reported
    # edge precision/recall can never diverge from what the model actually merges.
    windows = sorted(per_window.keys())
    win_offsets: dict[int, int] = {}
    total = 0
    for w in windows:
        win_offsets[w] = total
        total += len(per_window[w]["clusters"])
    edges: list[tuple[int, int]] = []
    with torch.inference_mode():
        ctx_all, bge_all = d["ctx_vecs"], d["mention_bge"]
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                wi, wj = per_window[windows[i]], per_window[windows[j]]
                lc, rc = wi["clusters"], wj["clusters"]
                if not lc or not rc:
                    continue
                ctx_i, bge_i = ctx_all[wi["global_idx"]], bge_all[wi["global_idx"]]
                ctx_j, bge_j = ctx_all[wj["global_idx"]], bge_all[wj["global_idx"]]
                left = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(cl, ctx_i, bge_i) for cl in lc]]
                right = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(cl, ctx_j, bge_j) for cl in rc]]
                if hasattr(cluster_matcher, "lex"):
                    lex = _lex_for_pair(d, int(wi["global_idx"][0]), int(wj["global_idx"][0]), lc, rc,
                                        wi["global_idx"], wj["global_idx"], device)
                    scores = cluster_matcher(left, right, lex).float().cpu().numpy()
                else:
                    scores = cluster_matcher(left, right).float().cpu().numpy()
                pairs = decode_cluster_matches(scores, float(cluster_matcher.null_bias.item()))
                lo, ro = win_offsets[windows[i]], win_offsets[windows[j]]
                for li, ri in pairs:
                    edges.append((lo + li, ro + ri))
    return edges, win_offsets, total


def _predict_full_doc_clusters(
    d: dict, per_window: dict[int, dict], cluster_matcher: ClusterMatcher, device: str
) -> list[list]:
    windows = sorted(per_window.keys())
    edges, win_offsets, total = _stage_b_merge_edges(d, per_window, cluster_matcher, device)
    parent = list(range(total))
    for a, b in edges:
        ra, rb = _uf_find(parent, a), _uf_find(parent, b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in range(total):
        groups.setdefault(_uf_find(parent, i), []).append(i)
    result = []
    for members in groups.values():
        spans = []
        for ci in members:
            # find which window this cluster belongs to
            for w in windows:
                lo = win_offsets[w]
                lc = per_window[w]["clusters"]
                if lo <= ci < lo + len(lc):
                    local_cluster = lc[ci - lo]
                    global_idx = per_window[w]["global_idx"]
                    spans.extend(d["spans"][global_idx[li]] for li in local_cluster)
                    break
        if len(spans) >= 2:
            result.append(spans)
    return result


def eval_stage_b(
    docs: list,
    stage_a_clusters: list[dict[int, list[list[int]]]],
    cluster_matcher: ClusterMatcher,
    device: str,
    tag: str,
    type_breakdown: bool = False,
) -> dict:
    cluster_matcher.eval()
    key_docs, resp_docs = [], []
    for d, per_window in zip(docs, stage_a_clusters):
        if not per_window:
            continue
        key_docs.append((d["name"], d["sentences"], _key_clusters(d)))
        resp_docs.append(
            (d["name"], d["sentences"], _predict_full_doc_clusters(d, per_window, cluster_matcher, device))
        )
    key_path = MODELS_DIR / f"stageb_{tag}_key.conll"
    resp_path = MODELS_DIR / f"stageb_{tag}_resp.conll"
    write_conll(key_path, key_docs)
    write_conll(resp_path, resp_docs)
    result = conll_f1(key_path, resp_path)
    if type_breakdown:
        result["by_type"] = conll_f1_by_type(key_docs, resp_docs, MODELS_DIR)
    return result


def _score_one_doc(name: str, sentences, key_clusters, resp_clusters) -> float:
    kp = MODELS_DIR / "_diag_doc_key.conll"
    rp = MODELS_DIR / "_diag_doc_resp.conll"
    write_conll(kp, [(name, sentences, key_clusters)])
    write_conll(rp, [(name, sentences, resp_clusters)])
    return conll_f1(kp, rp)["CoNLL"]


def _nearest_ante_dists(tok_pos: np.ndarray, cid: np.ndarray) -> list[int]:
    # subtoken distance from each mention to its nearest earlier same-entity mention
    dists = []
    for i in range(len(cid)):
        for j in range(i - 1, -1, -1):
            if cid[j] == cid[i]:
                dists.append(int(tok_pos[i] - tok_pos[j]))
                break
    return dists


def _stage_a_purity(per_window: dict, cid: np.ndarray) -> tuple[int, int]:
    # (pure, total): a Stage A cluster is pure if all its members share one gold entity
    pure = total = 0
    for w in per_window:
        gi = per_window[w]["global_idx"]
        for cl in per_window[w]["clusters"]:
            total += 1
            if len({int(cid[gi[x]]) for x in cl}) == 1:
                pure += 1
    return pure, total


def decode_threshold_sweep(
    docs: list,
    clusters: list,
    cluster_matcher: ClusterMatcher,
    device: str,
    offsets=(-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4),
) -> tuple[float, float, dict]:
    # Post-hoc (no retraining): sweep the Stage B decode threshold (null_bias + offset)
    # and score corpus CoNLL F1 each. Lowering the threshold merges more (recovers recall).
    base = float(cluster_matcher.null_bias.item())
    orig = cluster_matcher.null_bias.data.clone()
    results: dict[float, float] = {}
    for off in offsets:
        cluster_matcher.null_bias.data.fill_(base + off)
        key_docs, resp_docs = [], []
        for d, per_window in zip(docs, clusters):
            if not per_window:
                continue
            key_docs.append((d["name"], d["sentences"], _key_clusters(d)))
            resp_docs.append(
                (d["name"], d["sentences"], _predict_full_doc_clusters(d, per_window, cluster_matcher, device))
            )
        kp, rp = MODELS_DIR / "_thr_key.conll", MODELS_DIR / "_thr_resp.conll"
        write_conll(kp, key_docs)
        write_conll(rp, resp_docs)
        results[round(base + off, 3)] = conll_f1(kp, rp)["CoNLL"]
    cluster_matcher.null_bias.data.copy_(orig)
    best_thr = max(results, key=results.get)
    return best_thr, results[best_thr], results


def write_diagnostics(
    test_docs: list,
    test_clusters: list,
    scorer: AntecedentScorer,
    cluster_matcher: ClusterMatcher,
    device: str,
    window: int,
    dataset: str,
    split: str,
) -> None:
    # Emitted from the validated eval path for one dataset; appends a summary row keyed
    # by (K=CONTENT, dataset) and writes a per-doc CSV. Tracks every claim-relevant
    # metric: quotient compression, locality (nearest-antecedent within K), Stage A
    # purity, Stage B edge precision/recall + recall by window gap, and per-doc
    # runtime/VRAM scaling for Model C (windowed) vs Model B (all-pairs).
    rows, tp, fp = [], 0, 0
    gap_tot: dict[int, int] = {}
    gap_ok: dict[int, int] = {}
    key_all, resp_ab_all, resp_a_all = [], [], []
    for d, per_window in zip(test_docs, test_clusters):
        if not per_window:
            continue
        cid = d["cluster_id"]
        windows = sorted(per_window.keys())
        gold, cwin, total = [], [], 0
        for w in windows:
            for cl in per_window[w]["clusters"]:
                gold.append(_cluster_gold_cids([per_window[w]["global_idx"][x] for x in cl], cid))
                cwin.append(w)
                total += 1
        edges, _, _ = _stage_b_merge_edges(d, per_window, cluster_matcher, device)
        parent = list(range(total))
        for a, b in edges:
            if gold[a] == gold[b]:
                tp += 1
            else:
                fp += 1
            ra, rb = _uf_find(parent, a), _uf_find(parent, b)
            if ra != rb:
                parent[ra] = rb
        # Stage B recall over cross-window gold cluster pairs (same-window pairs are
        # unreachable), bucketed by window gap to test whether all-pairs window
        # comparison (vs adjacent-only) is what recovers long-range links
        for a in range(total):
            for b in range(a + 1, total):
                if gold[a] == gold[b] and cwin[a] != cwin[b]:
                    g = abs(cwin[a] - cwin[b])
                    gap_tot[g] = gap_tot.get(g, 0) + 1
                    if _uf_find(parent, a) == _uf_find(parent, b):
                        gap_ok[g] = gap_ok.get(g, 0) + 1
        key = _key_clusters(d)
        resp_ab = _predict_full_doc_clusters(d, per_window, cluster_matcher, device)
        resp_a = [
            [d["spans"][per_window[w]["global_idx"][li]] for li in cl]
            for w in windows
            for cl in per_window[w]["clusters"]
            if len(cl) >= 2
        ]
        key_all.append((d["name"], d["sentences"], key))
        resp_ab_all.append((d["name"], d["sentences"], resp_ab))
        resp_a_all.append((d["name"], d["sentences"], resp_a))

        tok_pos = np.asarray(d["tok_pos"])
        dists = _nearest_ante_dists(tok_pos, cid)
        within = sum(1 for x in dists if x <= window)
        occ = np.bincount(tok_pos // window)
        occ = occ[occ > 0]
        pure, ctot = _stage_a_purity(per_window, cid)
        n_sub, m = len(d["content_ids"]), len(cid)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            _predict_full_doc_clusters(d, _stage_a_clusters_for_doc(d, scorer, device, window), cluster_matcher, device)
        if device == "cuda":
            torch.cuda.synchronize()
        rt_c = (time.perf_counter() - t0) * 1000
        vr_c = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            ctx = torch.from_numpy(d["ctx_vecs"]).to(device).float()
            bge = torch.from_numpy(d["mention_bge"]).to(device).float()
            sb, mb = scorer(ctx, bge)
            decode_antecedents(sb.float().cpu().numpy(), mb.cpu().numpy(), float(scorer.null_bias.item()))
        if device == "cuda":
            torch.cuda.synchronize()
        rt_b = (time.perf_counter() - t0) * 1000
        vr_b = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0

        rows.append(
            {
                "doc": d["name"],
                "subtokens": n_sub,
                "mentions": m,
                "stageA_clusters": total,
                "gold_entities": len(set(int(c) for c in cid)),
                "density_M_over_N": round(m / n_sub, 4),
                "max_win_occ": int(occ.max()) if len(occ) else 0,
                "mean_win_occ": round(float(occ.mean()), 2) if len(occ) else 0.0,
                "median_ante_dist": int(np.median(dists)) if dists else 0,
                "frac_ante_within_K": round(within / len(dists), 4) if dists else 0.0,
                "stageA_purity": round(pure / ctot, 4) if ctot else 0.0,
                "conll_AB": round(_score_one_doc(d["name"], d["sentences"], key, resp_ab), 4),
                "conll_Aalone": round(_score_one_doc(d["name"], d["sentences"], key, resp_a), 4),
                "runtime_C_ms": round(rt_c, 3),
                "runtime_B_ms": round(rt_b, 3),
                "vram_C_mb": round(vr_c, 1),
                "vram_B_mb": round(vr_b, 1),
            }
        )

    n = len(rows)
    n_mentions = sum(r["mentions"] for r in rows)
    n_clusters = sum(r["stageA_clusters"] for r in rows)
    n_entities = sum(r["gold_entities"] for r in rows)
    n_subtokens = sum(r["subtokens"] for r in rows)
    # corpus-level (micro) CoNLL F1 — the canonical aggregation the paper reports,
    # scored once over all docs (not the per-doc macro mean, which over-weights small docs)
    corpus_key = MODELS_DIR / "_diag_corpus_key.conll"
    corpus_resp = MODELS_DIR / "_diag_corpus_resp.conll"
    write_conll(corpus_key, key_all)
    write_conll(corpus_resp, resp_ab_all)
    f_ab = conll_f1(corpus_key, corpus_resp)["CoNLL"]
    write_conll(corpus_resp, resp_a_all)
    f_a = conll_f1(corpus_key, corpus_resp)["CoNLL"]

    p_trained = sum(p.numel() for p in scorer.parameters()) + sum(p.numel() for p in cluster_matcher.parameters())

    pd_path = MODELS_DIR / f"per_doc_k{window}_{dataset}.csv"
    with pd_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gap3_tot = sum(v for g, v in gap_tot.items() if g >= 3)
    summary = {
        "K": window,
        "dataset": dataset,
        "split": split,
        "n_docs": n,
        "mean_mentions": round(n_mentions / n, 2),
        "mean_subtokens": round(n_subtokens / n, 1),
        "mean_clusters": round(n_clusters / n, 2),
        "mean_entities": round(n_entities / n, 2),
        "compression_M_over_C": round(n_mentions / n_clusters, 3),
        "fragments_C_over_E": round(n_clusters / n_entities, 3),
        "mention_density": round(n_mentions / n_subtokens, 4),
        "frac_ante_within_K": round(sum(r["frac_ante_within_K"] for r in rows) / n, 4),
        "stageA_purity": round(sum(r["stageA_purity"] for r in rows) / n, 4),
        "conll_AB": round(f_ab, 4),
        "conll_Aalone": round(f_a, 4),
        "stageB_gain": round(f_ab - f_a, 4),
        "edge_precision": round(tp / max(tp + fp, 1), 4),
        "edge_recall": round(sum(gap_ok.values()) / max(sum(gap_tot.values()), 1), 4),
        "recall_gap_1": round(gap_ok.get(1, 0) / gap_tot[1], 4) if gap_tot.get(1) else 0.0,
        "recall_gap_2": round(gap_ok.get(2, 0) / gap_tot[2], 4) if gap_tot.get(2) else 0.0,
        "recall_gap_ge3": round(sum(v for g, v in gap_ok.items() if g >= 3) / gap3_tot, 4) if gap3_tot else 0.0,
        "runtime_C_ms_total": round(sum(r["runtime_C_ms"] for r in rows), 1),
        "runtime_B_ms_total": round(sum(r["runtime_B_ms"] for r in rows), 1),
        "peak_vram_C_mb": round(max(r["vram_C_mb"] for r in rows), 1),
        "peak_vram_B_mb": round(max(r["vram_B_mb"] for r in rows), 1),
        "trained_params": p_trained,
    }
    sum_path = MODELS_DIR / "summary_by_window.csv"
    existing = []
    if sum_path.exists():
        with sum_path.open(newline="") as f:
            existing = [r for r in csv.DictReader(f) if not (int(r["K"]) == window and r["dataset"] == dataset)]
    with sum_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        for r in existing:
            writer.writerow(r)
        writer.writerow(summary)
    print(f"\n✓ Diagnostics K={window} {dataset}/{split} ({n} docs) → {pd_path.name}")
    print(
        f"  conll_AB {summary['conll_AB'] * 100:.2f} A-alone {summary['conll_Aalone'] * 100:.2f} | "
        f"edgeP {summary['edge_precision'] * 100:.2f} edgeR {summary['edge_recall'] * 100:.2f} | "
        f"within-K {summary['frac_ante_within_K'] * 100:.2f} purity {summary['stageA_purity'] * 100:.2f}"
    )


def _diagnostic_eval_sets(docs: list, all_stage_a: list) -> list[tuple[str, str, list, list]]:
    # conll/litbank have test splits; preco/corefud only have validation
    plan = [("conll2012", "test"), ("litbank", "test"), ("preco", "validation"), ("corefud", "validation")]
    sets = []
    for ds, split in plan:
        idx = [i for i, d in enumerate(docs) if d["split"] == split and d["name"].startswith(ds + "/")]
        if idx:
            sets.append((ds, split, [docs[i] for i in idx], [all_stage_a[i] for i in idx]))
    return sets


_PRONOUNS = frozenset({
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves",
    "this", "that", "these", "those", "who", "whom", "whose", "which", "what", "there", "here",
})


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


def _lex_matrix(lc: list, rc: list, lgi, rgi, mention_tokens: list, idf: dict) -> np.ndarray:
    # (L, R, 3) cluster-pair lexical features: [IDF-weighted token Jaccard, substring containment, exact match]
    lsurf = [[mention_tokens[lgi[i]] for i in cl] for cl in lc]
    rsurf = [[mention_tokens[rgi[i]] for i in cl] for cl in rc]
    lset = [{t for m in s for t in m} for s in lsurf]
    rset = [{t for m in s for t in m} for s in rsurf]
    lmass = [sum(idf.get(t, 0.0) for t in s) for s in lset]
    rmass = [sum(idf.get(t, 0.0) for t in s) for s in rset]
    lcont = [[m for m in s if _content(m)] for s in lsurf]
    rcont = [[m for m in s if _content(m)] for s in rsurf]
    out = np.zeros((len(lc), len(rc), 3), dtype=np.float32)
    for i in range(len(lc)):
        ci = set(lcont[i])
        for j in range(len(rc)):
            inter = lset[i] & rset[j]
            im = sum(idf.get(t, 0.0) for t in inter)
            um = lmass[i] + rmass[j] - im
            out[i, j, 0] = im / um if um > 0 else 0.0
            out[i, j, 2] = 1.0 if ci & set(rcont[j]) else 0.0
            cont = any(_is_subseq(a, b) or _is_subseq(b, a) for a in lcont[i] for b in rcont[j])
            out[i, j, 1] = 1.0 if cont else 0.0
    return out


def _lex_for_pair(d: dict, ki: int, kj: int, lc: list, rc: list, lgi, rgi, device: str) -> torch.Tensor:
    # memoized per (window-pair) — lexical features are constant across epochs
    if "mention_tokens" not in d:
        _build_lexical([d])
    cache = d.setdefault("_lex", {})
    key = (ki, kj)
    if key not in cache:
        cache[key] = _lex_matrix(lc, rc, lgi, rgi, d["mention_tokens"], d["mention_idf"])
    return torch.from_numpy(cache[key]).to(device)


def _stage_b_losses(cluster_matcher, pending: list) -> list:
    # pending items: (left, right, left_cids, right_cids, lex). batch all window-pairs in one
    # forward when the head supports it (mention head), else per-pair; lex is None for pooled.
    if hasattr(cluster_matcher, "forward_many"):
        lex_list = [p[4] for p in pending]
        if all(x is None for x in lex_list):
            lex_list = None
        mats = cluster_matcher.forward_many([(p[0], p[1]) for p in pending], lex_list)
        return [
            cluster_match_loss(m, p[2], p[3], cluster_matcher.null_bias)
            for m, p in zip(mats, pending, strict=True)
        ]
    return [cluster_match_loss(cluster_matcher(p[0], p[1]), p[2], p[3], cluster_matcher.null_bias) for p in pending]


def _meta_hardness(meta: tuple, bge_all: np.ndarray) -> float:
    # confusability of a null window-pair: max cosine between any left/right cluster's
    # mean BGE. High = semantically similar but non-coreferent = a hard negative worth keeping.
    wi, wj, lc, rc = meta[0], meta[1], meta[2], meta[3]

    def norm_means(w: dict, clusters: list) -> np.ndarray | None:
        out = []
        for cl in clusters:
            v = bge_all[[w["global_idx"][li] for li in cl]].mean(0)
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(out) if out else None

    left, right = norm_means(wi, lc), norm_means(wj, rc)
    if left is None or right is None:
        return 0.0
    return float((left @ right.T).max())


def train_stage_b(
    head_lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    doc_bs: int = 8,
    window: int = CONTENT,
    neg_ratio: float | None = None,
    subset: str = "all",
    dropout: float = 0.3,
    weight_decay: float = 0.1,
    head: str = "cluster",
    hard_neg: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_only: bool = False,
) -> None:
    print(f"\nTraining Stage B matcher... (window={window}, subset={subset}, head={head}, hard_neg={hard_neg})")
    ctx_dir, frozen_base, ckpt_b, _ = _win_names(window, subset)
    if head == "mention":
        ckpt_b = ckpt_b.replace(".pt", "_ment.pt")
    if hard_neg:
        ckpt_b = ckpt_b.replace(".pt", "_hn.pt")
    nom_cache, datasets, preco_n, single_ctx = _data_cfg(subset)
    docs = build_docs(device=device, datasets=datasets, nom_cache=nom_cache, preco_n=preco_n)
    docs = _filter_docs(docs, subset)
    for d in docs:
        if "tok_pos" not in d:
            d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    if head == "mention":
        _build_lexical(docs)  # independent lexical-identity channel (§4.5/§8); mention head only

    stage_a = AntecedentScorer().to(device)
    stage_a.load_state_dict(torch.load(MODELS_DIR / f"{frozen_base}.pt", map_location=device)["scorer"])
    for p in stage_a.parameters():
        p.requires_grad_(False)
    stage_a.eval()

    if single_ctx:
        load_span_ctx_single(docs, ctx_dir)
    else:
        for i, d in enumerate(docs):
            p = ctx_dir / f"{i:06d}.npy"
            if p.exists() and "ctx_vecs" not in d:
                d["ctx_vecs"] = np.load(p).astype(np.float16)

    train_idx = _train_idx(docs, subset)
    train_docs = [docs[i] for i in train_idx]
    val_docs = [d for d in docs if d["split"] == "validation" and d["name"].startswith("conll2012/")]
    test_docs = [d for d in docs if d["split"] == "test" and d["name"].startswith("conll2012/")]

    # precompute Stage A clusters once — avoids re-running Stage A every epoch
    all_stage_a = _precompute_stage_a_clusters(docs, stage_a, device, window, subset)
    train_clusters = [all_stage_a[i] for i in train_idx]
    val_clusters = [
        all_stage_a[i] for i, d in enumerate(docs) if d["split"] == "validation" and d["name"].startswith("conll2012/")
    ]
    test_clusters = [
        all_stage_a[i] for i, d in enumerate(docs) if d["split"] == "test" and d["name"].startswith("conll2012/")
    ]

    all_train_pairs_count = sum(
        len(per_window) * (len(per_window) - 1) // 2 for per_window in train_clusters if per_window
    )
    print(f"Training cluster pairs per epoch: {all_train_pairs_count}")
    cluster_matcher = (MentionMatcher(dropout=dropout) if head == "mention" else ClusterMatcher(dropout=dropout)).to(
        device
    )
    if eval_only:
        cluster_matcher.load_state_dict(torch.load(MODELS_DIR / ckpt_b, map_location=device)["cluster_matcher"])
        cluster_matcher.eval()
        print("\n=== Stage B Test (eval only) ===")
        test_scores = eval_stage_b(test_docs, test_clusters, cluster_matcher, device, "test", type_breakdown=True)
        print(
            f"Test CoNLL {test_scores['CoNLL'] * 100:.2f} MUC {test_scores['muc'] * 100:.2f} "
            f"B3 {test_scores['bcub'] * 100:.2f} CEAFe {test_scores['ceafe'] * 100:.2f}"
        )
        for ds, split, dd, dc in _diagnostic_eval_sets(docs, all_stage_a):
            write_diagnostics(dd, dc, stage_a, cluster_matcher, device, window, ds, split)
        return
    optimizer = optim.AdamW(cluster_matcher.parameters(), lr=head_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=head_lr * 0.1)
    ckpt_path = MODELS_DIR / ckpt_b
    best_val_f1, patience_ctr = -1.0, 0

    for epoch in range(max_epochs):
        print(f"\n=== Stage B (K={window}) Epoch {epoch + 1}/{max_epochs} ===")
        order = list(range(len(train_docs)))
        random.shuffle(order)
        cluster_matcher.train()
        total, n_pairs = 0.0, 0
        # accumulate (left_clusters, right_clusters, left_cids, right_cids) across docs;
        # the mention head materializes an Lm*Rm grid per pair, so cap a batch by accumulated
        # mention-pair count (not just window-pair count) to bound peak VRAM on dense windows.
        pending: list[tuple] = []
        pending_pairs = 0
        pair_budget = 40000 if head == "mention" else 10**12
        for bi in tqdm(order, desc="train_b"):
            d = train_docs[bi]
            per_window = train_clusters[bi]
            if not per_window:
                continue
            ctx_all, bge_all, cid = d["ctx_vecs"], d["mention_bge"], d["cluster_id"]
            windows = sorted(per_window.keys())
            # cheap pass: classify each window pair as positive (some left cluster has a
            # gold match in the right window) or null, so nulls can be subsampled
            metas = []
            for i in range(len(windows)):
                for j in range(i + 1, len(windows)):
                    wi, wj = per_window[windows[i]], per_window[windows[j]]
                    lc, rc = wi["clusters"], wj["clusters"]
                    if not lc or not rc:
                        continue
                    left_cids = [_cluster_gold_cids([wi["global_idx"][li] for li in cl], cid) for cl in lc]
                    right_cids = [_cluster_gold_cids([wj["global_idx"][li] for li in cl], cid) for cl in rc]
                    rset = set(right_cids)
                    is_pos = any(lcid in rset for lcid in left_cids)
                    metas.append((wi, wj, lc, rc, left_cids, right_cids, is_pos))
            if neg_ratio is not None and metas:
                pos = [m for m in metas if m[6]]
                neg = [m for m in metas if not m[6]]
                cap = int(neg_ratio * len(pos)) if pos else min(len(neg), 1)
                if len(neg) > cap:
                    if hard_neg:
                        neg = sorted(neg, key=lambda m: _meta_hardness(m, bge_all), reverse=True)[:cap]
                    else:
                        neg = random.sample(neg, cap)
                metas = pos + neg
            for wi, wj, lc, rc, left_cids, right_cids, _ in metas:
                ctx_i, bge_i = ctx_all[wi["global_idx"]], bge_all[wi["global_idx"]]
                ctx_j, bge_j = ctx_all[wj["global_idx"]], bge_all[wj["global_idx"]]
                left = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(cl, ctx_i, bge_i) for cl in lc]]
                right = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(cl, ctx_j, bge_j) for cl in rc]]
                lex = (
                    _lex_for_pair(d, int(wi["global_idx"][0]), int(wj["global_idx"][0]), lc, rc,
                                  wi["global_idx"], wj["global_idx"], device)
                    if head == "mention" else None
                )
                pending.append((left, right, left_cids, right_cids, lex))
                pending_pairs += sum(c.shape[0] for c, _ in left) * sum(c.shape[0] for c, _ in right)
                n_pairs += 1
                if len(pending) >= doc_bs or pending_pairs >= pair_budget:
                    optimizer.zero_grad()
                    losses = _stage_b_losses(cluster_matcher, pending)
                    torch.stack(losses).mean().backward()
                    torch.nn.utils.clip_grad_norm_(cluster_matcher.parameters(), 1.0)
                    optimizer.step()
                    total += sum(lo.item() for lo in losses)
                    pending, pending_pairs = [], 0
        if pending:
            optimizer.zero_grad()
            losses = _stage_b_losses(cluster_matcher, pending)
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(cluster_matcher.parameters(), 1.0)
            optimizer.step()
            total += sum(lo.item() for lo in losses)
        scheduler.step()
        tr_loss = total / max(n_pairs, 1)

        val_sets = {
            "conll2012": (val_docs, val_clusters),
            "litbank": (
                [d for d in docs if d["split"] == "validation" and d["name"].startswith("litbank/")],
                [
                    all_stage_a[i]
                    for i, d in enumerate(docs)
                    if d["split"] == "validation" and d["name"].startswith("litbank/")
                ],
            ),
            "preco": (
                [d for d in docs if d["split"] == "validation" and d["name"].startswith("preco/")],
                [
                    all_stage_a[i]
                    for i, d in enumerate(docs)
                    if d["split"] == "validation" and d["name"].startswith("preco/")
                ],
            ),
            "corefud": (
                [d for d in docs if d["split"] == "validation" and d["name"].startswith("corefud/")],
                [
                    all_stage_a[i]
                    for i, d in enumerate(docs)
                    if d["split"] == "validation" and d["name"].startswith("corefud/")
                ],
            ),
        }
        val_scores_all = {
            ds: eval_stage_b(vdocs, vclusters, cluster_matcher, device, f"val_{ds}_epoch{epoch + 1}")
            for ds, (vdocs, vclusters) in val_sets.items()
            if vdocs
        }
        val_scores = val_scores_all["conll2012"]
        print(f"Loss: {tr_loss:.6f}")
        for ds, sc in val_scores_all.items():
            print(
                f"  {ds:10s} CoNLL {sc['CoNLL'] * 100:.2f} "
                f"(MUC {sc['muc'] * 100:.2f} B3 {sc['bcub'] * 100:.2f} CEAFe {sc['ceafe'] * 100:.2f})"
            )

        if val_scores["CoNLL"] > best_val_f1 + 1e-4:
            best_val_f1, patience_ctr = val_scores["CoNLL"], 0
            torch.save({"cluster_matcher": cluster_matcher.state_dict(), "best_val_f1": best_val_f1}, ckpt_path)
            print(f"✓ Best cluster matcher saved (val CoNLL F1 {best_val_f1 * 100:.2f})")
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"\n⊘ Early stopping. Best val CoNLL F1 {best_val_f1 * 100:.2f}")
                break

    print("\n✓ Stage B training complete")
    cluster_matcher.load_state_dict(torch.load(ckpt_path, map_location=device)["cluster_matcher"])
    cluster_matcher.eval()
    print("\n=== Stage B Test ===")
    test_scores = eval_stage_b(test_docs, test_clusters, cluster_matcher, device, "test", type_breakdown=True)
    print(
        f"CoNLL {test_scores['CoNLL'] * 100:.2f} | MUC {test_scores['muc'] * 100:.2f} | "
        f"B3 {test_scores['bcub'] * 100:.2f} | CEAFe {test_scores['ceafe'] * 100:.2f}"
    )
    for ds, split, dd, dc in _diagnostic_eval_sets(docs, all_stage_a):
        write_diagnostics(dd, dc, stage_a, cluster_matcher, device, window, ds, split)
    if "by_type" in test_scores:
        for bucket, sc in sorted(test_scores["by_type"].items()):
            print(
                f"  {bucket:12s}  CoNLL {sc['CoNLL'] * 100:.2f}  MUC {sc['muc'] * 100:.2f}  "
                f"B3 {sc['bcub'] * 100:.2f}  CEAFe {sc['ceafe'] * 100:.2f}"
            )


if __name__ == "__main__":
    # mention-level Stage B (no subsampling) on the all8k Stage A head
    train_stage_b(window=256, subset="all8k", head="mention", dropout=0.2)
