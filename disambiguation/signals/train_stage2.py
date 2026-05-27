import datetime
import json
import pickle
import random

import numpy as np
import spacy
import spacy.tokens
import torch
import torch.optim as optim
from datasets import load_from_disk
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from disambiguation.signals.conll_scorer import write_conll, conll_f1
from disambiguation.signals.stage2_context_encoder import (
    ContextEncoder, MentionEncoder, AntecedentScorer, mll_loss, decode_antecedents,
    encode_document_ctx, gather_spans_tensor, load_tokenizer,
    MAX_SPAN_SUB, MODELS_DIR, CKPT_NAME,
)

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
DATA_DIR = CACHE_DIR.parent / "data"
SPACY_TRF_DIR = DATA_DIR / "spacy_trf"
NOM_CACHE = DATA_DIR / "stage2_conll_nominals_v2.pkl"
SPAN_CTX_CACHE = DATA_DIR / "stage2_span_ctx.pkl"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
SPLITS = ["train", "validation", "test"]


def _doc_structure(sample: dict, spacy_doc: spacy.tokens.Doc) -> tuple:
    sents = sample["sentences"]
    offsets, off = [], 0
    for s in sents:
        offsets.append(off)
        off += len(s)
    spacy_sents = list(spacy_doc.sents)
    spans, cluster_id, head_words = [], [], []
    for cid, cluster in enumerate(sample["mention_clusters"]):
        for si, a, b in cluster:
            if si >= len(spacy_sents):
                continue
            sent_span = spacy_sents[si]
            if a >= len(sent_span):
                continue
            head_local = sent_span[a:min(b, len(sent_span))].root.i - sent_span.start
            if head_local >= len(sents[si]):
                continue
            spans.append((si, a, b))
            cluster_id.append(cid)
            head_words.append(sents[si][head_local])
    return sents, offsets, spans, cluster_id, head_words


def _word_to_subtok(words: list, tokenizer) -> tuple:
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    word_ids = enc.word_ids()
    w2s: dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in w2s:
            w2s[wid] = pos
    return np.asarray(enc["input_ids"], dtype=np.int64), w2s


def build_docs(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> list:
    # Per-doc record: roberta content ids (no specials), and per gold mention its span subtoken range,
    # word width, frozen BGE head vector, span and cluster id. Mentions are sorted into document order
    # (by span start then end) so antecedent indexing j < i is left-to-right. No roberta forward here.
    if NOM_CACHE.exists():
        with NOM_CACHE.open("rb") as f:
            docs = pickle.load(f)
        for d in docs:
            d["head_bge"] = d["head_bge"].astype(np.float32)
        print(f"Loaded cached nominal docs: {len(docs)}")
        return docs

    from sentence_transformers import SentenceTransformer

    vocab = spacy.blank("en").vocab
    tokenizer = load_tokenizer()
    raw = []
    for split in SPLITS:
        ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
        db = spacy.tokens.DocBin().from_disk(SPACY_TRF_DIR / f"conll2012_{split}.spacy")
        for sample, sdoc in tqdm(zip(ds, db.get_docs(vocab), strict=True), total=len(ds), desc=f"conll2012/{split}"):
            sents, offsets, spans, cluster_id, head_words = _doc_structure(sample, sdoc)
            if len(spans) < 2:
                continue
            words_flat = [w for s in sents for w in s]
            content_ids, w2s = _word_to_subtok(words_flat, tokenizer)
            last = len(content_ids) - 1
            span_sub, width = [], []
            for si, a, b in spans:
                gw_start, gw_end = offsets[si] + a, offsets[si] + b - 1
                ss = w2s.get(gw_start, min(gw_start, last))
                se = min(max(w2s.get(gw_end + 1, len(content_ids)) - 1, ss), last)
                span_sub.append((ss, se))
                width.append(b - a)
            order = sorted(range(len(spans)), key=lambda k: span_sub[k])
            spans = [spans[k] for k in order]
            cluster_id = [cluster_id[k] for k in order]
            head_words = [head_words[k] for k in order]
            span_sub = [span_sub[k] for k in order]
            width = [width[k] for k in order]
            name = f"{sample['doc_id']}#{len(raw)}"
            raw.append((name, split, sents, spans, cluster_id, head_words, content_ids, span_sub, width))

    head_vocab = sorted({w for r in raw for w in r[5]})
    print(f"Encoding {len(head_vocab)} unique head words with BGE...")
    bge = SentenceTransformer(BGE_MODEL, device=device)
    embs = np.asarray(bge.encode(head_vocab, normalize_embeddings=True, batch_size=512, show_progress_bar=True), dtype=np.float32)
    word2bge = {w: embs[i] for i, w in enumerate(head_vocab)}
    del bge

    docs = []
    for name, split, sents, spans, cluster_id, head_words, content_ids, span_sub, width in raw:
        docs.append({
            "name": name, "split": split, "sentences": sents, "spans": spans,
            "cluster_id": np.asarray(cluster_id, dtype=np.int64),
            "content_ids": content_ids,
            "span_sub": np.asarray(span_sub, dtype=np.int64),
            "width": np.asarray(width, dtype=np.int64),
            "head_bge": np.stack([word2bge[w] for w in head_words]).astype(np.float32),
        })
    with NOM_CACHE.open("wb") as f:
        pickle.dump([{**d, "head_bge": d["head_bge"].astype(np.float16)} for d in docs], f)
    print(f"Cached {len(docs)} docs to {NOM_CACHE.name}")
    return docs


def gather_spans_np(ctx: np.ndarray, span_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lens = [min(int(e) - int(s) + 1, MAX_SPAN_SUB, ctx.shape[0] - int(s)) for s, e in span_sub]
    S = max(lens)
    out = np.zeros((len(span_sub), S, ctx.shape[1]), dtype=np.float16)
    for k, (s, _e) in enumerate(span_sub):
        out[k, :lens[k]] = ctx[int(s):int(s) + lens[k]]
    return out, np.asarray(lens, dtype=np.int64)


def precompute_span_ctx(encoder, docs, cls_id, sep_id, device) -> None:
    # Frozen-roberta path: sliding-window encode each doc once, slice out each mention's span subtoken
    # context vectors, pad per doc, cache to disk so re-runs skip the roberta pass.
    if SPAN_CTX_CACHE.exists():
        with SPAN_CTX_CACHE.open("rb") as f:
            cache = pickle.load(f)
        for d in docs:
            d["span_ctx"], d["span_len"] = cache[d["name"]]["span_ctx"], cache[d["name"]]["span_len"]
        print(f"Loaded cached span ctx: {len(cache)} docs")
        return
    encoder.eval()
    cache = {}
    with torch.inference_mode():
        for d in tqdm(docs, desc="span ctx precompute"):
            ctx = encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device).float().cpu().numpy()
            span_ctx, span_len = gather_spans_np(ctx, d["span_sub"])
            d["span_ctx"], d["span_len"] = span_ctx, span_len
            cache[d["name"]] = {"span_ctx": span_ctx, "span_len": span_len}
    with SPAN_CTX_CACHE.open("wb") as f:
        pickle.dump(cache, f)
    print(f"Cached span ctx to {SPAN_CTX_CACHE.name}")


def doc_scores(d: dict, encoder, mention_enc, scorer, cls_id, sep_id, device) -> torch.Tensor:
    if encoder is None:  # frozen: span context comes from the cache
        span_ctx = torch.from_numpy(d["span_ctx"]).to(device).float()
        span_len = torch.from_numpy(d["span_len"]).to(device)
    else:                # finetune: encode the document live so gradients reach roberta
        ctx = encode_document_ctx(d["content_ids"], encoder, cls_id, sep_id, device)
        span_ctx, span_len = gather_spans_tensor(ctx, d["span_sub"], device)
    head_bge = torch.from_numpy(d["head_bge"]).to(device).float()
    width = torch.from_numpy(d["width"]).to(device)
    reps = mention_enc(span_ctx, span_len, head_bge, width)
    return scorer(reps)  # (M, M) antecedent logits


def _sent_id(d: dict, device=None):
    sent = np.asarray([sp[0] for sp in d["spans"]], dtype=np.int64)
    return sent if device is None else torch.from_numpy(sent).to(device)


def run_epoch(encoder, mention_enc, scorer, docs, optimizer, cls_id, sep_id, device, doc_bs, intra_sentence=False) -> float:
    train = optimizer is not None
    total, ndoc = 0.0, 0
    for s in tqdm(range(0, len(docs), doc_bs), desc="train" if train else "val"):
        batch = docs[s:s + doc_bs]
        if train:
            optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            losses = [mll_loss(doc_scores(d, encoder, mention_enc, scorer, cls_id, sep_id, device),
                               torch.from_numpy(d["cluster_id"]).to(device),
                               _sent_id(d, device) if intra_sentence else None) for d in batch]
            loss = torch.stack(losses).mean()
        if train:
            loss.backward()
            params = list(mention_enc.parameters()) + list(scorer.parameters())
            if encoder is not None:
                params += list(encoder.parameters())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        total += loss.item() * len(batch)
        ndoc += len(batch)
    return total / max(ndoc, 1)


def predict_clusters(d: dict, encoder, mention_enc, scorer, cls_id, sep_id, device, intra_sentence=False) -> list:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
        scores = doc_scores(d, encoder, mention_enc, scorer, cls_id, sep_id, device)
    groups = decode_antecedents(scores.float().cpu().numpy(), _sent_id(d) if intra_sentence else None)
    return [[d["spans"][i] for i in g] for g in groups]


def _key_clusters(d: dict, intra_sentence: bool) -> list:
    if not intra_sentence:
        return [[d["spans"][i] for i in np.where(d["cluster_id"] == c)[0]] for c in np.unique(d["cluster_id"])]
    groups: dict = {}  # gold clusters split per sentence; an intra-sentence cluster needs >= 2 same-sentence members
    for i, c in enumerate(d["cluster_id"]):
        groups.setdefault((int(c), d["spans"][i][0]), []).append(i)
    return [[d["spans"][i] for i in members] for members in groups.values() if len(members) >= 2]


def eval_conll(encoder, mention_enc, scorer, docs, cls_id, sep_id, device, tag, intra_sentence=False) -> dict:
    key_docs = [(d["name"], d["sentences"], _key_clusters(d, intra_sentence)) for d in docs]
    key_path = MODELS_DIR / f"stage2_{tag}_key.conll"
    write_conll(key_path, key_docs)
    resp = [(d["name"], d["sentences"], predict_clusters(d, encoder, mention_enc, scorer, cls_id, sep_id, device, intra_sentence))
            for d in tqdm(docs, desc=f"eval/{tag}")]
    resp_path = MODELS_DIR / f"stage2_{tag}_response.conll"
    write_conll(resp_path, resp)
    return conll_f1(key_path, resp_path)


def train_stage2(
    finetune: bool = False,
    intra_sentence: bool = False,
    roberta_lr: float = 2e-5,
    head_lr: float = 1e-3,
    max_epochs: int | None = None,
    patience: int | None = None,
    doc_bs: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print(f"\nBuilding Stage 2 nominal data... (finetune={finetune})")
    docs = build_docs(device=device)
    tokenizer = load_tokenizer()
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    train_docs = [d for d in docs if d["split"] == "train"]
    val_docs = [d for d in docs if d["split"] == "validation"]
    test_docs = [d for d in docs if d["split"] == "test"]

    mention_enc = MentionEncoder().to(device)
    scorer = AntecedentScorer().to(device)
    encoder = ContextEncoder().to(device)
    if finetune:
        encoder.roberta.gradient_checkpointing_enable()
        max_epochs, patience, doc_bs = max_epochs or 3, patience or 2, doc_bs or 1
        optimizer = optim.AdamW(
            [{"params": encoder.parameters(), "lr": roberta_lr},
             {"params": list(mention_enc.parameters()) + list(scorer.parameters()), "lr": head_lr}],
            weight_decay=1e-2,
        )
        enc_loop, ckpt_path, tag = encoder, MODELS_DIR / CKPT_NAME, "finetune"
    else:
        for p in encoder.parameters():
            p.requires_grad_(False)
        precompute_span_ctx(encoder, docs, cls_id, sep_id, device)
        del encoder
        if device != "cpu":
            torch.cuda.empty_cache()
        max_epochs, patience, doc_bs = max_epochs or 30, patience or 5, doc_bs or 8
        optimizer = optim.AdamW(list(mention_enc.parameters()) + list(scorer.parameters()), lr=head_lr, weight_decay=1e-2)
        enc_loop, ckpt_path, tag = None, MODELS_DIR / "stage2_frozen_head.pt", "frozen"
    if intra_sentence:  # separate checkpoint/metrics so this never clobbers the global run
        tag = f"{tag}_intra"
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_intra{ckpt_path.suffix}")
    print(f"intra_sentence={intra_sentence}")
    head_params = sum(p.numel() for p in list(mention_enc.parameters()) + list(scorer.parameters()))
    print(f"head params: {head_params:,} | doc_bs={doc_bs} epochs={max_epochs}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(CACHE_DIR / "tensorboard" / f"stage2_{tag}_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
    best_f1, patience_ctr = -1.0, 0

    for epoch in range(max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        random.shuffle(train_docs)
        if enc_loop is not None:
            enc_loop.train()
        mention_enc.train()
        scorer.train()
        tr_loss = run_epoch(enc_loop, mention_enc, scorer, train_docs, optimizer, cls_id, sep_id, device, doc_bs, intra_sentence)
        if enc_loop is not None:
            enc_loop.eval()
        mention_enc.eval()
        scorer.eval()
        with torch.inference_mode():
            val_loss = run_epoch(enc_loop, mention_enc, scorer, val_docs, None, cls_id, sep_id, device, doc_bs, intra_sentence)
        val = eval_conll(enc_loop, mention_enc, scorer, val_docs, cls_id, sep_id, device, f"{tag}_val", intra_sentence)
        print(f"Loss: {tr_loss:.6f} | Val loss: {val_loss:.6f} | Val CoNLL F1: {val['CoNLL']:.4f} "
              f"(MUC {val['muc']:.4f} B3 {val['bcub']:.4f} CEAFe {val['ceafe']:.4f})")
        tb.add_scalar("loss/train", tr_loss, epoch + 1)
        tb.add_scalar("loss/val", val_loss, epoch + 1)
        for k in ("CoNLL", "muc", "bcub", "ceafe"):
            tb.add_scalar(f"val_f1/{k}", val[k], epoch + 1)
        if val["CoNLL"] > best_f1 + 1e-4:
            best_f1, patience_ctr = val["CoNLL"], 0
            state = {"mention_enc": mention_enc.state_dict(), "scorer": scorer.state_dict(), "best_val_f1": best_f1}
            if enc_loop is not None:
                state["encoder"] = enc_loop.state_dict()
            torch.save(state, ckpt_path)
            print(f"✓ Best model saved (val CoNLL F1 {best_f1:.4f})")
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"\n⊘ Early stopping. Best val CoNLL F1 {best_f1:.4f}")
                break

    print("\n✓ Training complete")
    tb.close()
    ckpt = torch.load(ckpt_path, map_location=device)
    mention_enc.load_state_dict(ckpt["mention_enc"])
    scorer.load_state_dict(ckpt["scorer"])
    mention_enc.eval()
    scorer.eval()
    if enc_loop is not None:
        enc_loop.load_state_dict(ckpt["encoder"])
        enc_loop.eval()
    print("\n=== Test ===")
    metrics = eval_conll(enc_loop, mention_enc, scorer, test_docs, cls_id, sep_id, device, f"{tag}_test", intra_sentence)
    print(f"CoNLL {metrics['CoNLL']:.4f} | MUC {metrics['muc']:.4f} | B3 {metrics['bcub']:.4f} | CEAFe {metrics['ceafe']:.4f}")
    metrics["best_val_f1"] = best_f1
    with (MODELS_DIR / f"stage2_eval_metrics_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    train_stage2()
