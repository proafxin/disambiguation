import pickle
import random
from collections import Counter

import numpy as np
import spacy
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from disambiguation.paths import DATA_DIR, MODELS_DIR
from disambiguation.train_stage0 import MAX_MENTION_LEN, _sentence_spans

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BACKBONE = "SpanBERT/spanbert-base-cased"
STAGE0_FT_CKPT = MODELS_DIR / "stage0_detector_ft.pt"
STAGE0_SPANS_CACHE = DATA_DIR / "stage0_predicted_spans.pkl"
STAGE0_ITEMS_CACHE = DATA_DIR / "stage0_ft_items.pkl"
STAGE0_ITEMS_SEMREC_CACHE = DATA_DIR / "stage0_ft_items_semrec.pkl"
STAGE0_ITEMS_SALIENCE_CACHE = DATA_DIR / "stage0_ft_items_salience.pkl"
CONLL_SPLITS = ["train", "validation", "test"]
RANDOM_SEED = 42

DEF_DEFINITE  = 0
DEF_INDEFINITE = 1
DEF_BARE      = 2

NER_PERSON  = 0
NER_PLACE   = 1
NER_ORG     = 2
NER_OTHER   = 3
NER_NONE    = 4

DEP_SUBJ  = 0
DEP_OBJ   = 1
DEP_OTHER = 2

_NLP = None


def _get_nlp() -> spacy.language.Language:
    global _NLP  # noqa: PLW0603
    if _NLP is None:
        _NLP = spacy.load("en_core_web_lg", disable=["lemmatizer"])
    return _NLP


def _def_bin(word: str) -> int:
    w = word.lower()
    if w == "the":
        return DEF_DEFINITE
    if w in {"a", "an"}:
        return DEF_INDEFINITE
    return DEF_BARE


def _ner_bin(label: str) -> int:
    if label in {"PERSON"}:
        return NER_PERSON
    if label in {"GPE", "LOC", "FAC"}:
        return NER_PLACE
    if label in {"ORG", "NORP"}:
        return NER_ORG
    if label:
        return NER_OTHER
    return NER_NONE


def _dep_bin(dep: str) -> int:
    if dep in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
        return DEP_SUBJ
    if dep in {"dobj", "pobj", "iobj", "attr"}:
        return DEP_OBJ
    return DEP_OTHER


def _salience_for_doc(sentences: list[list[str]]) -> list[dict]:
    nlp = _get_nlp()
    seen_surfaces: set[str] = set()
    result = []
    for sent in sentences:
        if not sent:
            result.append({"def": [], "ner": [], "dep": [], "first": [], "pos": []})
            continue
        doc = nlp(" ".join(sent))
        tokens = list(doc)
        n = len(sent)
        # align spacy tokens to original words by index (best-effort, same count expected)
        def_bins, ner_bins, dep_bins, first_bins, pos_bins = [], [], [], [], []
        for i in range(n):
            t = tokens[i] if i < len(tokens) else None
            def_bins.append(_def_bin(sent[i]))
            ner_bins.append(_ner_bin(t.ent_type_ if t else ""))
            dep_bins.append(_dep_bin(t.dep_ if t else ""))
            surf = sent[i].lower()
            first_bins.append(0 if surf in seen_surfaces else 1)
            seen_surfaces.add(surf)
            pos_bins.append(min(i * 4 // max(n, 1), 3))  # quartile 0..3
        result.append({"def": def_bins, "ner": ner_bins, "dep": dep_bins, "first": first_bins, "pos": pos_bins})
    return result


def _sem_rec_for_doc(sentences: list[list[str]], bge: SentenceTransformer) -> list[list[float]]:
    all_words = [w for sent in sentences for w in sent]
    if not all_words:
        return [[] for _ in sentences]
    embs = bge.encode(all_words, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    embs = np.array(embs, dtype=np.float32)          # (N, d)
    sim = embs @ embs.T                              # (N, N) cosine sim (normalised)
    np.fill_diagonal(sim, -1.0)                      # exclude self
    max_sim = sim.max(axis=1)                        # (N,) per-word max similarity to any other word
    result, offset = [], 0
    for sent in sentences:
        n = len(sent)
        result.append(max_sim[offset : offset + n].tolist())
        offset += n
    return result


def load_items(split: str, bge: SentenceTransformer) -> list:
    ds = load_from_disk(str(DATA_DIR / "conll2012"))[split]
    items = []
    for s in tqdm(ds, desc=f"load_items/{split}"):
        sp = _sentence_spans(s)
        freq = Counter(w.lower() for sent in s["sentences"] for w in sent)
        sem_rec = _sem_rec_for_doc(s["sentences"], bge)
        salience = _salience_for_doc(s["sentences"])
        for si, words in enumerate(s["sentences"]):
            if len(words) < 1:
                continue
            rec = [freq[w.lower()] for w in words]
            items.append({"words": words, "spans": sp.get(si, []), "rec": rec, "sem_rec": sem_rec[si], **{k: salience[si][k] for k in salience[si]}})
    return items


def load_all_items(device: str) -> dict[str, list]:
    if STAGE0_ITEMS_CACHE.exists():
        with STAGE0_ITEMS_CACHE.open("rb") as f:
            items = pickle.load(f)
        print(f"Loaded cached items: train {len(items['train'])} | val {len(items['validation'])} | test {len(items['test'])}", flush=True)
        return items
    bge = SentenceTransformer(BGE_MODEL, device=device)
    items = {split: load_items(split, bge) for split in CONLL_SPLITS}
    with STAGE0_ITEMS_CACHE.open("wb") as f:
        pickle.dump(items, f)
    print(f"Cached items -> {STAGE0_ITEMS_CACHE.name}", flush=True)
    return items


class FTDetector(nn.Module):
    def __init__(self, proj: int = 512, dropout: float = 0.2, width_dim: int = 64, rec_dim: int = 32):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(BACKBONE).float()
        self.roberta.gradient_checkpointing_enable()
        d = self.roberta.config.hidden_size
        self.attn    = nn.Linear(d, 1)
        self.width   = nn.Embedding(MAX_MENTION_LEN + 1, width_dim)
        self.rec_emb = nn.Embedding(5, rec_dim)
        self.bound_start = nn.Linear(d, 1)   # per-token start boundary score
        self.bound_end   = nn.Linear(d, 1)   # per-token end boundary score
        self.register_buffer("rec_bounds", torch.tensor([1.5, 2.5, 4.5, 9.5]))
        self.ffnn = nn.Sequential(
            nn.Linear(3 * d + width_dim + 2 * rec_dim, proj), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(proj, 1),
        )

    def forward(self, input_ids, attn_mask, word_first, rec):
        h = self.roberta(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        idx = word_first.clamp(min=0).unsqueeze(-1).expand(-1, -1, h.size(-1))
        hw = torch.gather(h, 1, idx)
        B, W, d = hw.shape
        L = MAX_MENTION_LEN
        a = self.attn(hw).squeeze(-1)
        e = torch.exp(a - a.amax(dim=1, keepdim=True))
        z = torch.zeros(B, 1, device=hw.device, dtype=hw.dtype)
        zh = torch.zeros(B, 1, d, device=hw.device, dtype=hw.dtype)
        pe = torch.cat([z, e.cumsum(1)], dim=1)
        peh = torch.cat([zh, (e.unsqueeze(-1) * hw).cumsum(1)], dim=1)
        i_idx = torch.arange(W, device=hw.device).view(W, 1)
        k_idx = torch.arange(L, device=hw.device).view(1, L)
        end  = (i_idx + k_idx + 1).clamp(max=W)
        jcol = (i_idx + k_idx).clamp(max=W - 1)
        Z = (pe[:, end] - pe[:, :W].unsqueeze(-1)).clamp(min=1e-6)
        N = peh[:, end] - peh[:, :W].unsqueeze(2)
        content = N / Z.unsqueeze(-1)
        hi = hw.unsqueeze(2).expand(B, W, L, d)
        hj = hw[:, jcol]
        we = self.width(k_idx.expand(W, L)).unsqueeze(0).expand(B, W, L, -1).to(hw.dtype)
        rb = torch.bucketize(rec.float(), self.rec_bounds)
        re = self.rec_emb(rb).to(hw.dtype)
        re_i = re.unsqueeze(2).expand(B, W, L, -1)
        re_j = re[:, jcol]
        g = torch.cat([hi, hj, content, we, re_i, re_j], dim=-1)
        span_score  = self.ffnn(g).squeeze(-1)                        # (B, W, L)
        start_score = self.bound_start(hw).squeeze(-1)                # (B, W)
        end_score   = self.bound_end(hw).squeeze(-1)                  # (B, W)
        return span_score + start_score.unsqueeze(2) + end_score[:, jcol]  # (B, W, L)


def collate(batch: list, tok, device: str, max_words: int = 120) -> tuple:
    words = [b["words"][:max_words] for b in batch]
    enc = tok(words, is_split_into_words=True, padding=True, truncation=True, max_length=max_words + 2, return_tensors="pt")
    B = len(batch)
    Wmax = max(len(w) for w in words)
    word_first = torch.zeros(B, Wmax, dtype=torch.long)
    wlen = []
    for b in range(B):
        seen = {}
        for p, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid not in seen:
                seen[wid] = p
        nw = len(words[b])
        wlen.append(nw)
        for w in range(nw):
            word_first[b, w] = seen.get(w, 0)
    L = MAX_MENTION_LEN
    tgt = torch.zeros(B, Wmax, L)
    for b in range(B):
        for a, e in batch[b]["spans"]:
            k = e - a
            if 0 <= k < L and a < wlen[b] and e < wlen[b]:
                tgt[b, a, k] = 1.0
    ar_i = torch.arange(Wmax).view(1, Wmax, 1)
    ar_k = torch.arange(L).view(1, 1, L)
    lb = torch.tensor(wlen).view(B, 1, 1)
    mask = ar_i + ar_k < lb
    rec = torch.zeros(B, Wmax, dtype=torch.long)
    for b in range(B):
        n = min(wlen[b], Wmax)
        for w in range(min(n, len(batch[b]["rec"]))):
            rec[b, w] = batch[b]["rec"][w]
    return (enc["input_ids"].to(device), enc["attention_mask"].to(device), word_first.to(device),
            tgt.to(device), mask.to(device), wlen, rec.to(device))


def decode(scores: np.ndarray, n: int, thr: float = 0.5) -> set:
    out = set()
    for i in range(n):
        for k in range(min(MAX_MENTION_LEN, n - i)):
            if scores[i, k] > thr:
                out.add((i, i + k))
    return out


def _unpack(batch_out: tuple) -> tuple:
    ids, attn, wf, tgt, mask, wlen, rec = batch_out
    return ids, attn, wf, tgt, mask, wlen, rec


def evaluate(model, items, tok, device, bs=32, thr=0.5) -> dict:
    model.eval()
    tp = fp = fn = 0
    nlp = _get_nlp()
    type_stats: dict[str, list] = {t: [0, 0, 0] for t in ("PROPN", "NOUN", "PRON", "OTHER")}
    with torch.inference_mode():
        for s in range(0, len(items), bs):
            batch = items[s : s + bs]
            ids, attn, wf, _, _, wlen, rec = _unpack(collate(batch, tok, device))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logits = model(ids, attn, wf, rec)
            sc = torch.sigmoid(logits.float()).cpu().numpy()
            for b, r in enumerate(batch):
                pred = decode(sc[b], wlen[b], thr)
                gold = {(a, e) for a, e in r["spans"] if a < wlen[b] and e < wlen[b]}
                tp += len(pred & gold)
                fp += len(pred - gold)
                fn += len(gold - pred)
                doc = nlp(" ".join(r["words"]))
                pos_map = {i: t.pos_ for i, t in enumerate(doc) if i < wlen[b]}
                for a, e in gold:
                    pos = pos_map.get(a, "")
                    bucket = pos if pos in type_stats else "OTHER"
                    hit = (a, e) in pred
                    type_stats[bucket][0] += int(hit)       # tp
                    type_stats[bucket][2] += int(not hit)   # fn
                for a, e in pred - gold:
                    pos = pos_map.get(a, "")
                    bucket = pos if pos in type_stats else "OTHER"
                    type_stats[bucket][1] += 1              # fp
    p = tp / (tp + fp + 1e-9)
    rc = tp / (tp + fn + 1e-9)
    result = {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc + 1e-9)}
    for t, (ttp, tfp, tfn) in type_stats.items():
        tp_ = ttp / (ttp + tfp + 1e-9)
        rc_ = ttp / (ttp + tfn + 1e-9)
        result[f"{t}_f1"] = 2 * tp_ * rc_ / (tp_ + rc_ + 1e-9)
    return result


def train(max_epochs: int = 15, bs: int = 32, patience: int = 4, enc_lr: float = 2e-5, head_lr: float = 1e-3,
          device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    all_items = load_all_items(device)
    train_items, val_items, test_items = all_items["train"], all_items["validation"], all_items["test"]
    print(f"FT detector ({BACKBONE}) — train {len(train_items)} | val {len(val_items)} | test {len(test_items)}", flush=True)
    model = FTDetector().to(device)
    arch_sig = f"rec_dim={model.rec_emb.embedding_dim},boundary=True"
    if STAGE0_FT_CKPT.exists() and STAGE0_SPANS_CACHE.exists():
        ckpt = torch.load(STAGE0_FT_CKPT, map_location=device)
        if ckpt.get("arch") == arch_sig:
            print(f"Checkpoint matches architecture ({arch_sig}) — skipping training.", flush=True)
            return
        print(f"Checkpoint architecture mismatch ({ckpt.get('arch')} vs {arch_sig}) — retraining.", flush=True)
    head_params = (
        list(model.attn.parameters()) + list(model.width.parameters()) +
        list(model.rec_emb.parameters()) + list(model.bound_start.parameters()) +
        list(model.bound_end.parameters()) + list(model.ffnn.parameters())
    )
    opt = optim.AdamW([
        {"params": model.roberta.parameters(), "lr": enc_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=0.01)
    steps_per_epoch = (len(train_items) + bs - 1) // bs
    total_steps = steps_per_epoch * max_epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best, patience_ctr = -1.0, 0
    for ep in range(max_epochs):
        model.train()
        random.shuffle(train_items)
        tot, nb = 0.0, 0
        for s in tqdm(range(0, len(train_items), bs), desc=f"ep{ep+1}"):
            ids, attn, wf, tgt, mask, _, rec = _unpack(collate(train_items[s : s + bs], tok, device))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logits = model(ids, attn, wf, rec)
                loss = (bce(logits, tgt) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        tr_loss = tot / nb
        # val loss
        model.eval()
        val_tot, val_nb = 0.0, 0
        with torch.inference_mode():
            for s in range(0, len(val_items), bs):
                ids, attn, wf, tgt, mask, _, rec = _unpack(collate(val_items[s : s + bs], tok, device))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    logits = model(ids, attn, wf, rec)
                    val_loss_b = (bce(logits, tgt) * mask).sum() / mask.sum().clamp(min=1)
                val_tot += val_loss_b.item()
                val_nb += 1
        val_loss = val_tot / val_nb
        tm = evaluate(model, train_items[:2000], tok, device)
        vm = evaluate(model, val_items, tok, device)
        print(f"epoch {ep+1}/{max_epochs} loss {tr_loss:.4f} | val loss {val_loss:.4f}", flush=True)
        print(f"  train P {tm['precision']:.3f} R {tm['recall']:.3f} F1 {tm['f1']:.4f}", flush=True)
        print(f"  val   P {vm['precision']:.3f} R {vm['recall']:.3f} F1 {vm['f1']:.4f}", flush=True)
        if vm["f1"] > best + 1e-4:
            best, patience_ctr = vm["f1"], 0
            torch.save({"model": model.state_dict(), "val_f1": best, "arch": arch_sig}, STAGE0_FT_CKPT)
            print(f"  ✓ saved (val F1 {best:.4f})", flush=True)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  ⊘ early stop (best val F1 {best:.4f})", flush=True)
                break
    model.load_state_dict(torch.load(STAGE0_FT_CKPT, map_location=device)["model"])
    tm = evaluate(model, test_items, tok, device)
    print(f"\nFT detector test (best-val ckpt, exact-span): P {tm['precision']:.4f} R {tm['recall']:.4f} F1 {tm['f1']:.4f}", flush=True)
    for t in ("PROPN", "NOUN", "PRON", "OTHER"):
        print(f"  {t:6s} F1 {tm[t+'_f1']:.4f}", flush=True)
    cache_predictions(model, {"train": train_items, "validation": val_items, "test": test_items}, tok, device)  # type: ignore[arg-type]


def cache_predictions(model, splits: dict, tok, device, bs: int = 16, thr: float = 0.5) -> None:  # noqa: PLR0913
    # Dump the (best) detector's predicted spans + gold spans per sentence, so Stage A/B can
    # consume them without re-running the encoder.
    model.eval()
    out = []
    with torch.inference_mode():
        for split, items in splits.items():
            for s in range(0, len(items), bs):
                batch = items[s : s + bs]
                ids, attn, wf, _, _, wlen, rec = _unpack(collate(batch, tok, device))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    logits = model(ids, attn, wf, rec)
                sc = torch.sigmoid(logits.float()).cpu().numpy()
                for b, r in enumerate(batch):
                    out.append({
                        "split": split, "tokens": r["words"],
                        "gold_spans": r["spans"],
                        "pred_spans": sorted(decode(sc[b], wlen[b], thr)),
                    })
    with STAGE0_SPANS_CACHE.open("wb") as f:
        pickle.dump(out, f)
    print(f"Cached detector predictions for {len(out)} sentences -> {STAGE0_SPANS_CACHE.name}", flush=True)


if __name__ == "__main__":
    train()
