import datetime
import json
import pickle
import random
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
    CKPT_NAME,
    CONTENT,
    AntecedentScorer,
    ContextEncoder,
    compute_mention_ctx_vecs,
    decode_antecedents,
    encode_document_ctx,
    load_tokenizer,
    mll_loss,
)

BGE_MODEL = "BAAI/bge-large-en-v1.5"
CONLL_SPLITS = ["train", "validation", "test"]
PRECO_SUBSAMPLE = 8000  # RAM cap: (M,2,1024) ctx_vecs; 8000 docs fits comfortably in ~10 GB
RANDOM_SEED = 42


def _doc_structure_conll(sample: dict, spacy_doc: spacy.tokens.Doc) -> tuple:
    # Returns (sents, offsets, spans, cluster_id, mention_surfaces)
    # spans: (sent_idx, start, end) end-exclusive; mention_surfaces: full span text for BGE
    sents = sample["sentences"]
    offsets, off = [], 0
    for s in sents:
        offsets.append(off)
        off += len(s)
    spacy_sents = list(spacy_doc.sents)
    spans, cluster_id, mention_surfaces = [], [], []
    for cid, cluster in enumerate(sample["mention_clusters"]):
        for si, a, b in cluster:
            if si >= len(spacy_sents):
                continue
            sent_span = spacy_sents[si]
            if a >= len(sent_span):
                continue
            head_local = sent_span[a : min(b, len(sent_span))].root.i - sent_span.start
            if head_local >= len(sents[si]):
                continue
            spans.append((si, a, b))
            cluster_id.append(cid)
            mention_surfaces.append(" ".join(sents[si][a:b]))
    return sents, offsets, spans, cluster_id, mention_surfaces


def _doc_structure_generic(sents: list[list[str]], clusters: list[list[tuple]]) -> tuple:
    # clusters: list of clusters, each a list of (sent_idx, start, end) end-exclusive
    offsets, off = [], 0
    for s in sents:
        offsets.append(off)
        off += len(s)
    spans, cluster_id, mention_surfaces = [], [], []
    for cid, cluster in enumerate(clusters):
        for si, a, b in cluster:
            if si >= len(sents) or a >= len(sents[si]):
                continue
            spans.append((si, a, b))
            cluster_id.append(cid)
            mention_surfaces.append(" ".join(sents[si][a:b]))
    return sents, offsets, spans, cluster_id, mention_surfaces


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
    name: str, split: str, sents: list, offsets: list, spans: list, cluster_id: list, mention_surfaces: list, tokenizer
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
            sents, offsets, spans, cluster_id, mention_surfaces = _doc_structure_conll(sample, sdoc)
            rec = _raw_from_doc(
                f"conll2012/{sample['doc_id']}#{len(raw)}",
                split,
                sents,
                offsets,
                spans,
                cluster_id,
                mention_surfaces,
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
            sents_str, offsets, spans, cluster_id, mention_surfaces = _doc_structure_generic(sents_list, clusters)
            rec = _raw_from_doc(
                f"litbank/{sample['doc_name']}#{len(raw)}",
                split,
                sents_str,
                offsets,
                spans,
                cluster_id,
                mention_surfaces,
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
        sents_str, offsets, spans, cluster_id, mention_surfaces = _doc_structure_generic(sents, clusters)
        rec = _raw_from_doc(
            f"preco/{sample['id']}#{len(raw)}",
            "train",
            sents_str,
            offsets,
            spans,
            cluster_id,
            mention_surfaces,
            tokenizer,
        )
        if rec:
            raw.append(rec)
    for sample in tqdm(preco_ds["validation"], desc="preco/validation"):
        sents = sample["sentences"]
        clusters = _clusters_preco(sample)
        sents_str, offsets, spans, cluster_id, mention_surfaces = _doc_structure_generic(sents, clusters)
        rec = _raw_from_doc(
            f"preco/{sample['id']}#{len(raw)}",
            "validation",
            sents_str,
            offsets,
            spans,
            cluster_id,
            mention_surfaces,
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
            sents_str, offsets, spans, cluster_id, mention_surfaces = _doc_structure_generic(sents, clusters)
            rec = _raw_from_doc(
                f"corefud/{sample['doc_id']}#{len(raw)}",
                split,
                sents_str,
                offsets,
                spans,
                cluster_id,
                mention_surfaces,
                tokenizer,
            )
            if rec:
                raw.append(rec)

    # --- Encode all unique mention surfaces with BGE ---
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
            }
        )
    with NOM_CACHE.open("wb") as f:
        pickle.dump([{**d, "mention_bge": d["mention_bge"].astype(np.float16)} for d in docs], f)
    print(f"Cached {len(docs)} docs to {NOM_CACHE.name}")
    return docs


def gather_spans_np(ctx: np.ndarray, span_sub: np.ndarray) -> np.ndarray:
    # Returns (M, 2, CTX_DIM) float16: (ctx_start, ctx_end) per mention.
    M = len(span_sub)
    out = np.zeros((M, 2, ctx.shape[1]), dtype=np.float16)
    for k, (s, e) in enumerate(span_sub):
        out[k, 0] = ctx[int(s)]
        out[k, 1] = ctx[int(e)]
    return out


def _ctx_path(i: int) -> Path:
    return SPAN_CTX_CACHE / f"{i:06d}.npy"


def precompute_span_ctx(encoder, docs, cls_id, sep_id, device) -> None:
    # Writes each doc's (M, 2, CTX_DIM) float16 ctx_vecs (start, end) to an individual .npy file immediately.
    # Never accumulates more than one doc in RAM. Resumable: skips already-written files.
    SPAN_CTX_CACHE.mkdir(parents=True, exist_ok=True)
    n_existing = sum(1 for i in range(len(docs)) if _ctx_path(i).exists())
    if n_existing == len(docs):
        for i, d in enumerate(docs):
            d["ctx_vecs"] = np.load(_ctx_path(i)).astype(np.float32)
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
            p = _ctx_path(i)
            if p.exists():
                d["ctx_vecs"] = np.load(p).astype(np.float32)
                d.pop("content_ids", None)
                d.pop("span_sub", None)
                d.pop("sent_sub_offsets", None)
                d.pop("sent_sub_lengths", None)
                continue
            ctx = encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device).float().cpu().numpy()
            ctx_vecs = gather_spans_np(ctx, d["span_sub"])
            np.save(p, ctx_vecs)
            d["ctx_vecs"] = ctx_vecs.astype(np.float32)
            d.pop("content_ids", None)
            d.pop("span_sub", None)
            d.pop("sent_sub_offsets", None)
            d.pop("sent_sub_lengths", None)
    print(f"Cached span ctx to {SPAN_CTX_CACHE.name}/")


def doc_scores(
    d: dict, encoder, scorer, cls_id, sep_id, device
) -> list[tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    if encoder is None:
        ctx = torch.from_numpy(d["ctx_vecs"]).to(device).float()
    else:
        ctx_doc = encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device)
        ctx = compute_mention_ctx_vecs(ctx_doc, d["span_sub"])
    bge = torch.from_numpy(d["mention_bge"]).to(device).float()
    win_ids = d["tok_pos"] // CONTENT
    results = []
    for w in np.unique(win_ids):
        idx = np.where(win_ids == w)[0]
        if len(idx) < 2:
            continue
        w_scores, w_mask = scorer(ctx[idx], bge[idx])
        results.append((w_scores, w_mask, idx))
    return results


def run_epoch(
    encoder, scorer, docs, optimizer, cls_id, sep_id, device, doc_bs
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
                for w_scores, w_mask, idx in doc_scores(d, encoder, scorer, cls_id, sep_id, device):
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


def predict_clusters(d: dict, encoder, scorer, cls_id, sep_id, device) -> list:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
        windows = doc_scores(d, encoder, scorer, cls_id, sep_id, device)
    clusters = []
    null_bias = float(scorer.null_bias.item())
    for w_scores, w_mask, idx in windows:
        groups = decode_antecedents(w_scores.float().cpu().numpy(), w_mask.cpu().numpy(), null_bias)
        clusters.extend([d["spans"][idx[i]] for i in g] for g in groups)
    return clusters


def _key_clusters(d: dict, window: int | None = None) -> list:
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
    encoder, scorer, docs, cls_id, sep_id, device, tag, type_breakdown=False,
    window: int | None = None,
) -> dict:
    key_docs = [(d["name"], d["sentences"], _key_clusters(d, window)) for d in docs]
    key_path = MODELS_DIR / f"stage2_{tag}_key.conll"
    write_conll(key_path, key_docs)
    resp_docs = [
        (
            d["name"],
            d["sentences"],
            predict_clusters(d, encoder, scorer, cls_id, sep_id, device),
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(f"\nBuilding Stage 2 nominal data... (finetune={finetune})")
    print("Evaluation setting: gold mentions (not end-to-end; not comparable to predicted-mention systems)")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    docs = build_docs(device=device)
    for d in docs:  # retain mention subtoken start positions before precompute drops span_sub
        d["tok_pos"] = np.asarray([s for s, _ in d["span_sub"]], dtype=np.int64)
    tokenizer = load_tokenizer()
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    conll_val_docs = [d for d in docs if d["split"] == "validation" and d["name"].startswith("conll2012/")]
    conll_test_docs = [d for d in docs if d["split"] == "test" and d["name"].startswith("conll2012/")]
    train_docs = [d for d in docs if d["split"] == "train"]
    val_sets = {
        "conll2012": conll_val_docs,
        "litbank": [d for d in docs if d["split"] == "validation" and d["name"].startswith("litbank/")],
        "preco": [d for d in docs if d["split"] == "validation" and d["name"].startswith("preco/")],
        "corefud": [d for d in docs if d["split"] == "validation" and d["name"].startswith("corefud/")],
    }

    scorer = AntecedentScorer().to(device)
    encoder = ContextEncoder().to(device)
    if finetune:
        encoder.roberta.gradient_checkpointing_enable()
        max_epochs, patience, doc_bs = max_epochs or 3, patience or 2, doc_bs or 1
        optimizer = optim.AdamW(
            [
                {"params": encoder.parameters(), "lr": roberta_lr},
                {"params": list(scorer.parameters()), "lr": head_lr},
            ],
            weight_decay=0.1,
        )
        enc_loop, ckpt_path, tag = encoder, MODELS_DIR / CKPT_NAME, "finetune"
    else:
        for p in encoder.parameters():
            p.requires_grad_(False)
        precompute_span_ctx(encoder, docs, cls_id, sep_id, device)
        del encoder
        if device != "cpu":
            torch.cuda.empty_cache()
        max_epochs, patience, doc_bs = max_epochs or 60, patience or 8, doc_bs or 8
        optimizer = optim.AdamW(
            list(scorer.parameters()), lr=head_lr, weight_decay=0.1
        )
        enc_loop, ckpt_path, tag = None, MODELS_DIR / "stage2_frozen_head.pt", "frozen"
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

    for epoch in range(max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        random.shuffle(train_docs)
        if enc_loop is not None:
            enc_loop.train()
        scorer.train()
        tr_loss = run_epoch(
            enc_loop, scorer, train_docs, optimizer, cls_id, sep_id, device, doc_bs
        )
        scheduler.step()
        if enc_loop is not None:
            enc_loop.eval()
        scorer.eval()
        with torch.inference_mode():
            val_loss = run_epoch(
                enc_loop, scorer, conll_val_docs, None, cls_id, sep_id, device, doc_bs
            )
        val_scores = {
            ds: eval_conll(
                enc_loop, scorer, vdocs, cls_id, sep_id, device, f"{tag}_val_{ds}", window=CONTENT
            )
            for ds, vdocs in val_sets.items()
            if vdocs
        }
        val = val_scores["conll2012"]
        print(f"Loss: {tr_loss:.6f} | Val loss: {val_loss:.6f}")
        for ds, sc in val_scores.items():
            print(
                f"  {ds:10s} CoNLL {sc['CoNLL']:.4f} (MUC {sc['muc']:.4f} B3 {sc['bcub']:.4f} CEAFe {sc['ceafe']:.4f})"
            )
        tb.add_scalar("loss/train", tr_loss, epoch + 1)
        tb.add_scalar("loss/val", val_loss, epoch + 1)
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
        window=CONTENT,
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
            enc_loop, scorer, vdocs, cls_id, sep_id, device, f"{tag}_final_val_{ds}", window=CONTENT
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


if __name__ == "__main__":
    train_stage2()
