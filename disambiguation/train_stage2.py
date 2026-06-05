import csv
import datetime
import json
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
    SPACY_TRF_DIR,
    SPAN_CTX_CACHE,
    TENSORBOARD_DIR,
)
from disambiguation.stage2_context_encoder import (
    CKPT_B_NAME,
    CONTENT,
    AntecedentScorer,
    ClusterMatcher,
    ContextEncoder,
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
PRECO_SUBSAMPLE = 8000  # RAM cap: (M,2,1024) ctx_vecs; 8000 docs fits comfortably in ~10 GB
RANDOM_SEED = 42

def _win_names(window: int) -> tuple:
    # window-tagged artifact names so K=128/256/512 never collide or reuse each other:
    # (ctx cache dir, Stage A head base, Stage B matcher, Stage A clusters cache)
    return (
        SPAN_CTX_CACHE.with_name(SPAN_CTX_CACHE.name + f"_k{window}"),
        f"stage2_frozen_head_k{window}",
        f"stage2_cluster_matcher_k{window}.pt",
        f"stage_a_clusters_cache_k{window}.pkl",
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


def build_docs(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> list:
    if NOM_CACHE.exists():
        with NOM_CACHE.open("rb") as f:
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
    for sample in tqdm(preco_train[:PRECO_SUBSAMPLE], desc="preco/train"):
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
    for split in ("train", "validation"):
        ds = load_from_disk(str(DATA_DIR / "corefud"))[split]
        for sample in tqdm(ds, desc=f"corefud/{split}"):
            sents = [[t["form"] for t in sent["tokens"]] for sent in sample["sentences"]]
            clusters = _clusters_corefud(sample)
            sents_str, offsets, spans, cluster_id, mention_surfaces, head_lex = _doc_structure_generic(sents, clusters)
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
    return _assemble_docs(raw, device, NOM_CACHE)


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


def precompute_span_ctx(encoder, docs, cls_id, sep_id, device, cache_dir: Path = SPAN_CTX_CACHE, window: int = CONTENT) -> None:
    # Writes each doc's (M, 2, CTX_DIM) float16 ctx_vecs (start, end) to an individual .npy file immediately.
    # Never accumulates more than one doc in RAM. Resumable: skips already-written files.
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_existing = sum(1 for i in range(len(docs)) if _ctx_path(i, cache_dir).exists())
    if n_existing == len(docs):
        for i, d in enumerate(docs):
            d["ctx_vecs"] = np.load(_ctx_path(i, cache_dir)).astype(np.float32)
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
                d["ctx_vecs"] = np.load(p).astype(np.float32)
                d.pop("content_ids", None)
                d.pop("span_sub", None)
                d.pop("sent_sub_offsets", None)
                d.pop("sent_sub_lengths", None)
                continue
            ctx = encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device, window).float().cpu().numpy()
            ctx_vecs = gather_spans_np(ctx, d["span_sub"])
            np.save(p, ctx_vecs)
            d["ctx_vecs"] = ctx_vecs.astype(np.float32)
            d.pop("content_ids", None)
            d.pop("span_sub", None)
            d.pop("sent_sub_offsets", None)
            d.pop("sent_sub_lengths", None)
    print(f"Cached span ctx to {cache_dir.name}/")


def doc_scores(
    d: dict, encoder, scorer, cls_id, sep_id, device, all_pairs: bool = False, max_mentions: int | None = None,
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
) -> float:
    train = optimizer is not None
    total, ndoc = 0.0, 0
    for s in tqdm(range(0, len(docs), doc_bs), desc="train" if train else "val"):
        batch = docs[s : s + doc_bs]
        if train:
            optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
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


def predict_clusters(d: dict, encoder, scorer, cls_id, sep_id, device, all_pairs: bool = False, window: int = CONTENT) -> list:
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(f"\nBuilding Stage 2 nominal data... (finetune={finetune}, predicted={predicted}, window={window})")
    if predicted:
        print("Evaluation setting: PREDICTED mentions (end-to-end), CoNLL only; KEY = true gold clusters")
    else:
        print("Evaluation setting: gold mentions (not end-to-end; not comparable to predicted-mention systems)")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    docs = build_conll_predicted_docs(device=device) if predicted else build_docs(device=device)
    for d in docs:  # retain mention subtoken start positions before precompute drops span_sub
        d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    tokenizer = load_tokenizer()
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    conll_val_docs = [d for d in docs if d["split"] == "validation" and d["name"].startswith("conll2012/")]
    conll_test_docs = [d for d in docs if d["split"] == "test" and d["name"].startswith("conll2012/")]
    train_docs = [d for d in docs if d["split"] == "train"]
    if predicted:
        val_sets = {"conll2012": conll_val_docs}
    else:
        val_sets = {
            "conll2012": conll_val_docs,
            "litbank": [d for d in docs if d["split"] == "validation" and d["name"].startswith("litbank/")],
            "preco": [d for d in docs if d["split"] == "validation" and d["name"].startswith("preco/")],
            "corefud": [d for d in docs if d["split"] == "validation" and d["name"].startswith("corefud/")],
        }

    ctx_dir, frozen_base, _, _ = _win_names(window)
    frozen_name = frozen_base
    if predicted:
        frozen_name += "_pred"
    if all_pairs:
        frozen_name += "_ap"
    scorer = AntecedentScorer().to(device)
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
            weight_decay=0.1,
        )
        enc_loop, ckpt_path, tag = encoder, MODELS_DIR / f"{frozen_name}_ft.pt", "finetune"
    else:
        for p in encoder.parameters():
            p.requires_grad_(False)
        precompute_span_ctx(
            encoder, docs, cls_id, sep_id, device,
            cache_dir=SPAN_CTX_CACHE_PRED if predicted else ctx_dir, window=window,
        )
        del encoder
        if device != "cpu":
            torch.cuda.empty_cache()
        max_epochs, patience, doc_bs = max_epochs or 60, patience or 8, doc_bs or 8
        optimizer = optim.AdamW(list(scorer.parameters()), lr=head_lr, weight_decay=0.1)
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
        print(f"Warm-started from {ckpt_path.name} (best_val_f1 {disk_best_f1:.4f}); training fresh from epoch 0")

    # During fine-tune training the live encoder's activations co-reside with all-pairs scoring;
    # cap the all-pairs group so the rare long docs (M up to ~660) window instead of OOM. Eval is uncapped.
    ft_cap = 160 if finetune else None
    for epoch in range(max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        random.shuffle(train_docs)
        if enc_loop is not None:
            enc_loop.train()
        scorer.train()
        tr_loss = run_epoch(enc_loop, scorer, train_docs, optimizer, cls_id, sep_id, device, doc_bs, all_pairs, ft_cap, window)
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
                f"  {ds:10s} loss {val_losses[ds]:.4f}  CoNLL {sc['CoNLL']:.4f} (MUC {sc['muc']:.4f} B3 {sc['bcub']:.4f} CEAFe {sc['ceafe']:.4f})"
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
                print(f"✓ Best model saved (val CoNLL F1 {disk_best_f1:.4f})")
            else:
                print(f"Improved this run to {best_f1:.4f} (all-time best {disk_best_f1:.4f}; not overwriting)")
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"\n⊘ Early stopping. Best val CoNLL F1 {best_f1:.4f}")
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
        f"CoNLL {metrics['CoNLL']:.4f} | MUC {metrics['muc']:.4f} | B3 {metrics['bcub']:.4f} | CEAFe {metrics['ceafe']:.4f}"
    )
    if "by_type" in metrics:
        print("\nType breakdown (official CoNLL F1 on cluster subsets):")
        for bucket, sc in sorted(metrics["by_type"].items()):
            print(
                f"  {bucket:12s}  CoNLL {sc['CoNLL']:.4f}  MUC {sc['muc']:.4f}  B3 {sc['bcub']:.4f}  CEAFe {sc['ceafe']:.4f}"
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
            f"  val/{ds:10s} CoNLL {sc['CoNLL']:.4f} (MUC {sc['muc']:.4f} B3 {sc['bcub']:.4f} CEAFe {sc['ceafe']:.4f})"
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


def _precompute_stage_a_clusters(docs: list, scorer: AntecedentScorer, device: str, window: int = CONTENT) -> list[dict[int, dict]]:
    cache_path = MODELS_DIR / _win_names(window)[3]
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
    print(f"  conll_AB {summary['conll_AB']} A-alone {summary['conll_Aalone']} | "
          f"edgeP {summary['edge_precision']} edgeR {summary['edge_recall']} | "
          f"within-K {summary['frac_ante_within_K']} purity {summary['stageA_purity']}")


def _diagnostic_eval_sets(docs: list, all_stage_a: list) -> list[tuple[str, str, list, list]]:
    # conll/litbank have test splits; preco/corefud only have validation
    plan = [("conll2012", "test"), ("litbank", "test"), ("preco", "validation"), ("corefud", "validation")]
    sets = []
    for ds, split in plan:
        idx = [i for i, d in enumerate(docs) if d["split"] == split and d["name"].startswith(ds + "/")]
        if idx:
            sets.append((ds, split, [docs[i] for i in idx], [all_stage_a[i] for i in idx]))
    return sets


def train_stage_b(
    head_lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    doc_bs: int = 8,
    window: int = CONTENT,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_only: bool = False,
) -> None:
    print(f"\nTraining Stage B cluster matcher... (window={window})")
    ctx_dir, frozen_base, ckpt_b, _ = _win_names(window)
    docs = build_docs(device=device)
    for d in docs:
        if "tok_pos" not in d:
            d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)

    stage_a = AntecedentScorer().to(device)
    stage_a.load_state_dict(torch.load(MODELS_DIR / f"{frozen_base}.pt", map_location=device)["scorer"])
    for p in stage_a.parameters():
        p.requires_grad_(False)
    stage_a.eval()

    for i, d in enumerate(docs):
        p = ctx_dir / f"{i:06d}.npy"
        if p.exists() and "ctx_vecs" not in d:
            d["ctx_vecs"] = np.load(p).astype(np.float32)

    train_docs = [d for d in docs if d["split"] == "train"]
    val_docs = [d for d in docs if d["split"] == "validation" and d["name"].startswith("conll2012/")]
    test_docs = [d for d in docs if d["split"] == "test" and d["name"].startswith("conll2012/")]

    # precompute Stage A clusters once — avoids re-running Stage A every epoch
    all_stage_a = _precompute_stage_a_clusters(docs, stage_a, device, window)
    train_clusters = [all_stage_a[i] for i, d in enumerate(docs) if d["split"] == "train"]
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
    cluster_matcher = ClusterMatcher().to(device)
    if eval_only:
        cluster_matcher.load_state_dict(torch.load(MODELS_DIR / ckpt_b, map_location=device)["cluster_matcher"])
        cluster_matcher.eval()
        print("\n=== Stage B Test (eval only) ===")
        test_scores = eval_stage_b(test_docs, test_clusters, cluster_matcher, device, "test", type_breakdown=True)
        print(
            f"Test CoNLL {test_scores['CoNLL']:.4f} MUC {test_scores['muc']:.4f} B3 {test_scores['bcub']:.4f} CEAFe {test_scores['ceafe']:.4f}"
        )
        for ds, split, dd, dc in _diagnostic_eval_sets(docs, all_stage_a):
            write_diagnostics(dd, dc, stage_a, cluster_matcher, device, window, ds, split)
        return
    optimizer = optim.AdamW(cluster_matcher.parameters(), lr=head_lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=head_lr * 0.1)
    ckpt_path = MODELS_DIR / ckpt_b
    best_val_f1, patience_ctr = -1.0, 0

    for epoch in range(max_epochs):
        print(f"\n=== Stage B Epoch {epoch + 1}/{max_epochs} ===")
        order = list(range(len(train_docs)))
        random.shuffle(order)
        cluster_matcher.train()
        total, n_pairs = 0.0, 0
        # accumulate (left_clusters, right_clusters, left_cids, right_cids) across docs
        pending: list[tuple] = []
        for bi in tqdm(order, desc="train_b"):
            d = train_docs[bi]
            per_window = train_clusters[bi]
            if not per_window:
                continue
            ctx_all, bge_all, cid = d["ctx_vecs"], d["mention_bge"], d["cluster_id"]
            windows = sorted(per_window.keys())
            for i in range(len(windows)):
                for j in range(i + 1, len(windows)):
                    wi, wj = per_window[windows[i]], per_window[windows[j]]
                    lc, rc = wi["clusters"], wj["clusters"]
                    if not lc or not rc:
                        continue
                    ctx_i = ctx_all[wi["global_idx"]]
                    bge_i = bge_all[wi["global_idx"]]
                    ctx_j = ctx_all[wj["global_idx"]]
                    bge_j = bge_all[wj["global_idx"]]
                    left = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(cl, ctx_i, bge_i) for cl in lc]]
                    right = [(c.to(device), b.to(device)) for c, b in [_cluster_reps(cl, ctx_j, bge_j) for cl in rc]]
                    left_cids = [_cluster_gold_cids([wi["global_idx"][li] for li in cl], cid) for cl in lc]
                    right_cids = [_cluster_gold_cids([wj["global_idx"][li] for li in cl], cid) for cl in rc]
                    pending.append((left, right, left_cids, right_cids))
                    n_pairs += 1
            if len(pending) >= doc_bs:
                optimizer.zero_grad()
                losses = [
                    cluster_match_loss(cluster_matcher(l, r), lc, rc, cluster_matcher.null_bias)
                    for l, r, lc, rc in pending
                ]
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(cluster_matcher.parameters(), 1.0)
                optimizer.step()
                total += sum(lo.item() for lo in losses)
                pending = []
        if pending:
            optimizer.zero_grad()
            losses = [
                cluster_match_loss(cluster_matcher(l, r), lc, rc, cluster_matcher.null_bias) for l, r, lc, rc in pending
            ]
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
                f"  {ds:10s} CoNLL {sc['CoNLL']:.4f} (MUC {sc['muc']:.4f} B3 {sc['bcub']:.4f} CEAFe {sc['ceafe']:.4f})"
            )

        if val_scores["CoNLL"] > best_val_f1 + 1e-4:
            best_val_f1, patience_ctr = val_scores["CoNLL"], 0
            torch.save({"cluster_matcher": cluster_matcher.state_dict(), "best_val_f1": best_val_f1}, ckpt_path)
            print(f"✓ Best cluster matcher saved (val CoNLL F1 {best_val_f1:.4f})")
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"\n⊘ Early stopping. Best val CoNLL F1 {best_val_f1:.4f}")
                break

    print("\n✓ Stage B training complete")
    cluster_matcher.load_state_dict(torch.load(ckpt_path, map_location=device)["cluster_matcher"])
    cluster_matcher.eval()
    print("\n=== Stage B Test ===")
    test_scores = eval_stage_b(test_docs, test_clusters, cluster_matcher, device, "test", type_breakdown=True)
    print(
        f"CoNLL {test_scores['CoNLL']:.4f} | MUC {test_scores['muc']:.4f} | B3 {test_scores['bcub']:.4f} | CEAFe {test_scores['ceafe']:.4f}"
    )
    for ds, split, dd, dc in _diagnostic_eval_sets(docs, all_stage_a):
        write_diagnostics(dd, dc, stage_a, cluster_matcher, device, window, ds, split)
    if "by_type" in test_scores:
        for bucket, sc in sorted(test_scores["by_type"].items()):
            print(
                f"  {bucket:12s}  CoNLL {sc['CoNLL']:.4f}  MUC {sc['muc']:.4f}  B3 {sc['bcub']:.4f}  CEAFe {sc['ceafe']:.4f}"
            )


if __name__ == "__main__":
    import gc

    for k in (128, 256, 512):
        train_stage2(window=k)
        train_stage_b(window=k)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
