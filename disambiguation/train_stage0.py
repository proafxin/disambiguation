import pickle
import random

import numpy as np
import torch
from datasets import load_from_disk
from torch import nn, optim
from tqdm import tqdm

from disambiguation.paths import DATA_DIR, MODELS_DIR
from disambiguation.stage2_context_encoder import (
    CTX_DIM,
    ContextEncoder,
    load_tokenizer,
)
from disambiguation.train_stage2 import RANDOM_SEED, _word_to_subtok

STAGE0_CACHE = DATA_DIR / "stage0_conll_detect_v2.pkl"
STAGE0_CKPT = MODELS_DIR / "stage0_detector.pt"
CONLL_SPLITS = ["train", "validation", "test"]
MAX_MENTION_LEN = 30


def _sentence_spans(sample: dict) -> dict:
    # si -> list of (start, end_inclusive) gold mention spans, deduped (keeps nesting)
    sents = sample["sentences"]
    out: dict[int, set] = {}
    for cluster in sample["mention_clusters"]:
        for si, a, b in cluster:
            if si >= len(sents):
                continue
            n = len(sents[si])
            if a >= n or n == 0:
                continue
            out.setdefault(si, set()).add((int(a), int(min(b, n) - 1)))
    return {si: sorted(s) for si, s in out.items()}


def _encode_into(items: list, encoder, tok, device: str, budget: int = 16384) -> None:
    # Fills item["ctx"] (n_words, CTX_DIM fp16) for every item, batching sentences across
    # docs by a token budget: batch_size * padded_len <= budget, bounding VRAM without
    # stopping at doc boundaries. Sentences are sorted by length so padding waste stays low.
    cls_id, sep_id, pad_id = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    subs = [_word_to_subtok(it["tokens"], tok) for it in items]
    order = sorted(range(len(items)), key=lambda k: len(subs[k][0]))
    pbar = tqdm(total=len(items), desc="stage0 encode")

    def flush(batch: list, lmax: int) -> None:
        if not batch:
            return
        L, B = lmax + 2, len(batch)
        ids = np.full((B, L), pad_id, np.int64)
        mask = np.zeros((B, L), np.int64)
        for bi, k in enumerate(batch):
            sub = subs[k][0]
            ids[bi, 0] = cls_id
            ids[bi, 1 : 1 + len(sub)] = sub
            ids[bi, 1 + len(sub)] = sep_id
            mask[bi, : 2 + len(sub)] = 1
        with torch.inference_mode():
            h = encoder(torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)).float().cpu().numpy()
        for bi, k in enumerate(batch):
            sub, w2s = subs[k]
            last = len(sub) - 1
            words = items[k]["tokens"]
            items[k]["ctx"] = np.stack(
                [h[bi, 1 + min(w2s.get(i, last), last)] for i in range(len(words))]
            ).astype(np.float16)
        pbar.update(B)

    batch, lmax = [], 0
    for k in order:
        n = len(subs[k][0])
        if n == 0:
            items[k]["ctx"] = np.zeros((len(items[k]["tokens"]), CTX_DIM), np.float16)
            pbar.update(1)
            continue
        if batch and (len(batch) + 1) * max(lmax, n) > budget:
            flush(batch, lmax)
            batch, lmax = [], 0
        batch.append(k)
        lmax = max(lmax, n)
    flush(batch, lmax)
    pbar.close()


def build_stage0_data(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> list:
    if STAGE0_CACHE.exists():
        with STAGE0_CACHE.open("rb") as f:
            recs = pickle.load(f)
        print(f"Loaded cached Stage 0 sentences: {len(recs)}")
        return recs
    tok = load_tokenizer()
    enc = ContextEncoder().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    items = []
    for split in CONLL_SPLITS:
        ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
        for sample in tqdm(ds, desc=f"stage0 gather/{split}"):
            spans_by_sent = _sentence_spans(sample)
            for si, words in enumerate(sample["sentences"]):
                if len(words) < 1:
                    continue
                items.append({"split": split, "tokens": words, "spans": spans_by_sent.get(si, [])})
    _encode_into(items, enc, tok, device)
    del enc
    if device != "cpu":
        torch.cuda.empty_cache()
    recs = [it for it in items if "ctx" in it]
    with STAGE0_CACHE.open("wb") as f:
        pickle.dump(recs, f)
    print(f"Cached {len(recs)} sentences -> {STAGE0_CACHE.name}")
    return recs


class BiaffineDetector(nn.Module):
    def __init__(self, dim: int = CTX_DIM, proj: int = 384, dropout: float = 0.2):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(dim, proj), nn.ReLU(), nn.Dropout(dropout))
        self.end = nn.Sequential(nn.Linear(dim, proj), nn.ReLU(), nn.Dropout(dropout))
        self.U = nn.Parameter(torch.empty(proj + 1, proj + 1))
        nn.init.xavier_uniform_(self.U)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: (B, T, dim) -> (B, T, T) span logits; score[b,i,j] = span starting i ending j
        u = self.start(ctx)
        v = self.end(ctx)
        ones = torch.ones(*u.shape[:-1], 1, device=ctx.device, dtype=u.dtype)
        u1 = torch.cat([u, ones], dim=-1)
        v1 = torch.cat([v, ones], dim=-1)
        return torch.einsum("bip,pq,bjq->bij", u1, self.U, v1)


def _valid_mask(lengths: list[int], T: int, device: str) -> torch.Tensor:
    ar = torch.arange(T, device=device)
    i = ar.view(1, T, 1)
    j = ar.view(1, 1, T)
    lb = torch.tensor(lengths, device=device).view(-1, 1, 1)
    return (i <= j) & (j - i < MAX_MENTION_LEN) & (j < lb) & (i < lb)


def _batch(recs: list, device: str) -> tuple:
    T = max(len(r["tokens"]) for r in recs)
    B = len(recs)
    ctx = torch.zeros(B, T, CTX_DIM, device=device)
    tgt = torch.zeros(B, T, T, device=device)
    lengths = []
    for b, r in enumerate(recs):
        n = r["ctx"].shape[0]
        lengths.append(n)
        ctx[b, :n] = torch.from_numpy(r["ctx"].astype(np.float32)).to(device)
        for a, e in r["spans"]:
            if a < T and e < T:
                tgt[b, a, e] = 1.0
    return ctx, tgt, _valid_mask(lengths, T, device)


def decode(scores: np.ndarray, n: int, thr: float) -> set:
    out = set()
    for i in range(n):
        for j in range(i, min(i + MAX_MENTION_LEN, n)):
            if scores[i, j] > thr:
                out.add((i, j))
    return out


def eval_detect(model: BiaffineDetector, recs: list, device: str, thr: float) -> dict:
    model.eval()
    tp = fp = fn = 0
    with torch.inference_mode():
        for s in range(0, len(recs), 256):
            chunk = recs[s : s + 256]
            ctx, _, _ = _batch(chunk, device)
            sc = torch.sigmoid(model(ctx)).cpu().numpy()
            for b, r in enumerate(chunk):
                n = len(r["tokens"])
                pred = decode(sc[b], n, thr)
                gold = set(map(tuple, r["spans"]))
                tp += len(pred & gold)
                fp += len(pred - gold)
                fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"precision": p, "recall": rc, "f1": f1}


def train_stage0(
    lr: float = 1e-3, max_epochs: int = 40, patience: int = 6, sent_bs: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    recs = build_stage0_data(device)
    train = [r for r in recs if r["split"] == "train"]
    val = [r for r in recs if r["split"] == "validation"]
    test = [r for r in recs if r["split"] == "test"]
    print(f"Stage 0 biaffine — sentences: train {len(train)} | val {len(val)} | test {len(test)}")

    print(f"gold spans {sum(len(r['spans']) for r in train)} | pos_weight 1.0 (plain BCE, threshold left to the bias)")

    model = BiaffineDetector().to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.1)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best_f1, ctr = -1.0, 0
    for epoch in range(max_epochs):
        model.train()
        random.shuffle(train)
        tot = 0.0
        for s in range(0, len(train), sent_bs):
            chunk = train[s : s + sent_bs]
            ctx, tgt, mask = _batch(chunk, device)
            logits = model(ctx)
            loss = (bce(logits, tgt) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(chunk)
        sched.step()
        thr = 0.5
        vm = eval_detect(model, val, device, thr)
        print(f"epoch {epoch+1}/{max_epochs} loss {tot/len(train):.4f} | val P {vm['precision']:.3f} R {vm['recall']:.3f} F1 {vm['f1']:.4f} @thr 0.5")
        if vm["f1"] > best_f1 + 1e-4:
            best_f1, ctr = vm["f1"], 0
            torch.save({"model": model.state_dict(), "best_val_f1": best_f1, "thr": thr}, STAGE0_CKPT)
            print(f"  ✓ saved (val F1 {best_f1:.4f})")
        else:
            ctr += 1
            if ctr >= patience:
                print(f"  ⊘ early stop (best {best_f1:.4f})")
                break
    ckpt = torch.load(STAGE0_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    thr = ckpt["thr"]
    tm = eval_detect(model, test, device, thr)
    print(f"\nStage 0 biaffine detection (CoNLL test, exact-span) @thr {thr}: "
          f"P {tm['precision']:.4f} R {tm['recall']:.4f} F1 {tm['f1']:.4f}")


if __name__ == "__main__":
    train_stage0()
