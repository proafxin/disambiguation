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
from disambiguation.signals.stage1_intrasentence import decode_clusters
from disambiguation.signals.stage2_context_encoder import (
    ContextEncoder, GlobalCorefHead, LogisticCorefHead, pair_bce_loss, encode_document, load_tokenizer,
    BGE_DIM, MODELS_DIR, CKPT_NAME,
)

HEADS = {"mlp": (GlobalCorefHead, CKPT_NAME), "logistic": (LogisticCorefHead, "stage2_logistic_coref.pt")}

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
DATA_DIR = CACHE_DIR.parent / "data"
SPACY_TRF_DIR = DATA_DIR / "spacy_trf"
DOCS_CACHE = DATA_DIR / "stage2_conll_docs.pkl"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
SPLITS = ["train", "validation", "test"]


def _doc_structure(sample: dict, spacy_doc: spacy.tokens.Doc) -> tuple:
    sents = sample["sentences"]
    offsets, off = [], 0
    for s in sents:
        offsets.append(off)
        off += len(s)
    spacy_sents = list(spacy_doc.sents)
    spans: list = []
    cluster_id: list = []
    head_words: list = []
    head_gpos: list = []
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
            head_gpos.append(offsets[si] + head_local)
    return sents, spans, cluster_id, head_words, head_gpos


def _word_to_subtok(words: list, tokenizer) -> tuple:
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    word_ids = enc.word_ids()
    w2s: dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in w2s:
            w2s[wid] = pos
    return np.asarray(enc["input_ids"], dtype=np.int64), w2s


def build_stage2_data(limit: int | None = None, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> list:
    if limit is None and DOCS_CACHE.exists():
        with DOCS_CACHE.open("rb") as f:
            docs = pickle.load(f)
        for d in docs:
            d["head_bge"] = d["head_bge"].astype(np.float32)
            d["head_ctx"] = d["head_ctx"].astype(np.float32)
        print(f"Loaded cached Stage 2 docs: {len(docs)}")
        return docs

    from sentence_transformers import SentenceTransformer

    vocab = spacy.blank("en").vocab
    tokenizer = load_tokenizer()
    raw: list = []  # (name, split, sents, spans, cluster_id, head_words, head_gpos, words_flat)
    for split in SPLITS:
        ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
        db = spacy.tokens.DocBin().from_disk(SPACY_TRF_DIR / f"conll2012_{split}.spacy")
        spacy_docs = db.get_docs(vocab)
        for sample, sdoc in tqdm(zip(ds, spacy_docs, strict=True), total=len(ds), desc=f"conll2012/{split}"):
            sents, spans, cluster_id, head_words, head_gpos = _doc_structure(sample, sdoc)
            if len(spans) < 2:
                continue
            words_flat = [w for s in sents for w in s]
            # conll2012 splits one document across multiple rows ("parts") sharing doc_id, each with
            # row-local clusters; suffix a running index so each part is a distinct unit for the scorer.
            name = f"{sample['doc_id']}#{len(raw)}"
            raw.append((name, split, sents, spans, cluster_id, head_words, head_gpos, words_flat))
            if limit is not None and len(raw) >= limit:
                break
        if limit is not None and len(raw) >= limit:
            break

    head_vocab = sorted({w for r in raw for w in r[5]})
    print(f"Encoding {len(head_vocab)} unique head words with BGE...")
    bge = SentenceTransformer(BGE_MODEL, device=device)
    embs = np.asarray(bge.encode(head_vocab, normalize_embeddings=True, batch_size=512, show_progress_bar=True), dtype=np.float32)
    word2bge = {w: embs[i] for i, w in enumerate(head_vocab)}
    del bge

    encoder = ContextEncoder().to(device)
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    docs: list = []
    for name, split, sents, spans, cluster_id, head_words, head_gpos, words_flat in tqdm(raw, desc="ctx encode"):
        content_ids, w2s = _word_to_subtok(words_flat, tokenizer)
        ctx = encode_document(content_ids, encoder, cls_id, sep_id, device)
        sub_idx = np.asarray([w2s.get(g, min(g, len(content_ids) - 1)) for g in head_gpos], dtype=np.int64)
        head_ctx = ctx[sub_idx]
        head_bge = np.stack([word2bge[w] for w in head_words]).astype(np.float32)
        docs.append({
            "name": name, "split": split, "sentences": sents, "spans": spans,
            "cluster_id": np.asarray(cluster_id, dtype=np.int64),
            "head_bge": head_bge, "head_ctx": head_ctx,
        })
    del encoder

    if limit is None:
        with DOCS_CACHE.open("wb") as f:
            pickle.dump([{**d, "head_bge": d["head_bge"].astype(np.float16), "head_ctx": d["head_ctx"].astype(np.float16)} for d in docs], f)
        print(f"Cached {len(docs)} docs to {DOCS_CACHE.name}")
    return docs


def _gold_matrix(cluster_id: np.ndarray) -> np.ndarray:
    same = (cluster_id[:, None] == cluster_id[None, :])
    np.fill_diagonal(same, False)
    return same.astype(np.float32)


def run_epoch(model, docs, optimizer, device) -> float:
    train = optimizer is not None
    total, n = 0.0, 0
    for d in tqdm(docs, desc="train" if train else "val"):
        bge = torch.from_numpy(d["head_bge"]).to(device)
        ctx = torch.from_numpy(d["head_ctx"]).to(device)
        gold = torch.from_numpy(_gold_matrix(d["cluster_id"])).to(device)
        if train:
            optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(bge, ctx)
            loss = pair_bce_loss(logits, gold)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def predict_clusters(model, d: dict, threshold: float, device: str) -> list:
    bge = torch.from_numpy(d["head_bge"]).to(device)
    ctx = torch.from_numpy(d["head_ctx"]).to(device)
    with torch.inference_mode():
        logits = model(bge, ctx).float().cpu().numpy()
    idx = np.arange(len(d["spans"]))
    groups = decode_clusters(idx, logits, threshold)  # frozensets of mention indices
    return [[d["spans"][i] for i in g] for g in groups]


def evaluate(model, docs, device, thresholds=(0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9)) -> list:
    key_docs = [(d["name"], d["sentences"],
                 [[d["spans"][i] for i in np.where(d["cluster_id"] == c)[0]]
                  for c in np.unique(d["cluster_id"])]) for d in docs]
    key_path = MODELS_DIR / "stage2_key.conll"
    write_conll(key_path, key_docs)
    results = []
    print(f"\n{'thr':>5} {'CoNLL':>7} {'MUC':>7} {'B3':>7} {'CEAFe':>7}")
    for thr in thresholds:
        resp_docs = [(d["name"], d["sentences"], predict_clusters(model, d, thr, device)) for d in docs]
        resp_path = MODELS_DIR / "stage2_response.conll"
        write_conll(resp_path, resp_docs)
        m = conll_f1(key_path, resp_path)
        m["threshold"] = thr
        results.append(m)
        print(f"{thr:>5.1f} {m['CoNLL']:>7.4f} {m['muc']:>7.4f} {m['bcub']:>7.4f} {m['ceafe']:>7.4f}")
    return results


def train_stage2(
    head: str = "mlp",
    learning_rate: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 6,
    resume: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    head_cls, ckpt_name = HEADS[head]
    print(f"\nBuilding Stage 2 data... (head={head})")
    docs = build_stage2_data(device=device)
    train_docs = [d for d in docs if d["split"] == "train"]
    val_docs = [d for d in docs if d["split"] == "validation"]
    test_docs = [d for d in docs if d["split"] == "test"]
    print(f"Docs: {len(train_docs)} train, {len(val_docs)} val, {len(test_docs)} test")

    print(f"\nTraining on {device}")
    model = head_cls().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(CACHE_DIR / "tensorboard" / f"stage2_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
    best_val, patience_ctr, best_epoch = float("inf"), 0, 0
    disk_best = float("inf")
    ckpt_path = MODELS_DIR / ckpt_name
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        disk_best = ckpt.get("best_val_loss", float("inf"))
        print(f"Warm-started from saved model (best_val_loss {disk_best:.6f}); training fresh from epoch 0")

    for epoch in range(max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        random.shuffle(train_docs)
        model.train()
        tr_loss = run_epoch(model, train_docs, optimizer, device)
        print(f"Loss: {tr_loss:.6f}")
        model.eval()
        with torch.inference_mode():
            val_loss = run_epoch(model, val_docs, None, device)
        print(f"Val loss: {val_loss:.6f}")
        scheduler.step()
        tb.add_scalar("loss/train", tr_loss, epoch + 1)
        tb.add_scalar("loss/val", val_loss, epoch + 1)

        if val_loss < best_val - 1e-4:
            best_val, patience_ctr, best_epoch = val_loss, 0, epoch
            if val_loss < disk_best - 1e-4:
                disk_best = val_loss
                torch.save({"epoch": epoch, "model": model.state_dict(), "best_val_loss": disk_best}, ckpt_path)
                print(f"✓ Best model saved (all-time best {disk_best:.6f})")
            else:
                print(f"Improved this run to {val_loss:.6f} (all-time best {disk_best:.6f}; not overwriting)")
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print(f"\n⊘ Early stopping. Best this run epoch {best_epoch + 1}, val_loss {best_val:.6f}")
                break

    print("\n✓ Training complete")
    tb.close()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("\n=== Validation ===")
    val_results = evaluate(model, val_docs, device)
    print("\n=== Test ===")
    test_results = evaluate(model, test_docs, device)
    with (MODELS_DIR / f"stage2_eval_metrics_{head}.json").open("w", encoding="utf-8") as f:
        json.dump({"validation": val_results, "test": test_results}, f, indent=2)


if __name__ == "__main__":
    train_stage2()
