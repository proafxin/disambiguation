import numpy as np
import spacy
import spacy.tokens
import torch
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from datasets import load_from_disk

from disambiguation.signals.stage1_intrasentence import DepGraphTransformer
from disambiguation.signals.train_stage1 import build_stage1_data, kfold_split, DATASET_CONFIG, SPACY_TRF_DIR, DATA_DIR
from disambiguation.signals.train_full import _clusters

CACHE_DIR = Path(__file__).parent / "cache"
MODELS_DIR = CACHE_DIR / "models"

# Mentions are head-token indices (sentence-relative), matching the training unit
# (gold spans mapped to span.root). Clusters are sets of those indices.
Cluster = frozenset[int]


def _pred_clusters(nom_idx: np.ndarray, scores: np.ndarray, null_margin: float) -> list[Cluster]:
    M = len(nom_idx)
    parent = list(range(M))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(M):
        masked = scores[i].copy()
        masked[i:M] = float("-inf")
        masked[M] -= null_margin
        ante = int(np.argmax(masked))
        if ante < M:
            parent[find(i)] = find(ante)

    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(find(i), []).append(int(nom_idx[i]))
    return [frozenset(members) for members in groups.values() if len(members) >= 2]


def _gold_clusters(doc_clusters: list, sent_idx: int, sent: spacy.tokens.Span) -> list[Cluster]:
    by_cluster: dict[int, set[int]] = {}
    for cid, cluster in enumerate(doc_clusters):
        for m_sent, m_start, m_end in cluster:
            if m_sent != sent_idx or m_start >= len(sent):
                continue
            span = sent[m_start:min(m_end, len(sent))]
            head_local = span.root.i - sent.start
            by_cluster.setdefault(cid, set()).add(head_local)
    return [frozenset(members) for members in by_cluster.values() if len(members) >= 2]


def _muc(pred: list[Cluster], gold: list[Cluster]) -> tuple[int, int, int, int]:
    def _score(a: list[Cluster], b: list[Cluster]) -> tuple[int, int]:
        num = den = 0
        for c in a:
            if len(c) < 2:
                continue
            den += len(c) - 1
            num += len(c) - sum(1 for bc in b if c & bc)
        return num, den
    pn, pd = _score(pred, gold)
    rn, rd = _score(gold, pred)
    return pn, pd, rn, rd


def _b3(pred: list[Cluster], gold: list[Cluster]) -> tuple[float, float, int]:
    pred_map: dict[Span, Cluster] = {m: c for c in pred for m in c}
    gold_map: dict[Span, Cluster] = {m: c for c in gold for m in c}
    mentions = set(pred_map) | set(gold_map)
    if not mentions:
        return 0.0, 0.0, 0
    p = sum(len(pred_map.get(m, frozenset({m})) & gold_map.get(m, frozenset({m}))) / len(pred_map.get(m, frozenset({m}))) for m in mentions)
    r = sum(len(pred_map.get(m, frozenset({m})) & gold_map.get(m, frozenset({m}))) / len(gold_map.get(m, frozenset({m}))) for m in mentions)
    return p, r, len(mentions)


def _ceafe(pred: list[Cluster], gold: list[Cluster]) -> tuple[float, float]:
    if not pred or not gold:
        return 0.0, 0.0
    cost = np.array([[2 * len(p & g) / (len(p) + len(g)) for g in gold] for p in pred])
    ri, ci = linear_sum_assignment(-cost)
    score = cost[ri, ci].sum()
    return score / len(pred), score / len(gold)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _collect_predictions(model: DepGraphTransformer, device: str, val_data: list) -> list[tuple[np.ndarray, np.ndarray, list[Cluster]]]:
    # Run the model once per val sentence; gold clusters don't depend on null_margin.
    by_doc: dict[tuple, list[tuple[int, tuple]]] = {}
    for row in val_data:
        key, cat, cont, edges, etypes, nom_idx, nom_lex, gold_ante = row
        ds_name, split_name, doc_id, sent_idx = key
        by_doc.setdefault((ds_name, split_name, doc_id), []).append((sent_idx, (cat, cont, edges, etypes, nom_idx, nom_lex)))
    needed_splits = {(key[0], key[1]) for key, *_ in val_data}

    vocab = spacy.blank("en").vocab
    collected: list[tuple[np.ndarray, np.ndarray, list[Cluster]]] = []

    for ds_name, splits in DATASET_CONFIG:
        ds_dict = None
        for split_name in splits:
            if (ds_name, split_name) not in needed_splits:
                continue
            if ds_dict is None:
                ds_dict = load_from_disk(str(DATA_DIR / ds_name))
            if split_name not in ds_dict:
                continue
            split_ds = ds_dict[split_name]
            doc_bin = spacy.tokens.DocBin().from_disk(SPACY_TRF_DIR / f"{ds_name}_{split_name}.spacy")
            doc_iter = iter(doc_bin.get_docs(vocab))

            for doc_id, (sample, full_doc) in enumerate(tqdm(
                zip(split_ds, doc_iter, strict=True),
                desc=f"{ds_name}/{split_name}",
                total=len(split_ds),
            )):
                doc_key = (ds_name, split_name, doc_id)
                if doc_key not in by_doc:
                    continue

                doc_clusters = _clusters(ds_name, sample)
                sents = list(full_doc.sents)

                for sent_idx, (cat, cont, edges, etypes, nom_idx, nom_lex) in by_doc[doc_key]:
                    sent = sents[sent_idx]
                    M = len(nom_idx)

                    with torch.no_grad():
                        attn_bias = model._build_batch_attn_bias([(edges, etypes)], [cat.shape[0]], cat.shape[0], device)
                        scores = model(
                            torch.from_numpy(cat).unsqueeze(0).to(device),
                            torch.from_numpy(cont).unsqueeze(0).to(device),
                            torch.zeros(1, cat.shape[0], dtype=torch.bool, device=device),
                            torch.from_numpy(nom_idx).unsqueeze(0).to(device),
                            torch.ones(1, M, dtype=torch.bool, device=device),
                            torch.from_numpy(nom_lex).unsqueeze(0).to(device),
                            attn_bias,
                        )
                    scores_np = scores.squeeze(0).cpu().numpy()
                    gold = _gold_clusters(doc_clusters, sent_idx, sent)
                    collected.append((nom_idx, scores_np, gold))

    return collected


def _score_predictions(collected: list[tuple[np.ndarray, np.ndarray, list[Cluster]]], null_margin: float) -> dict[str, float]:
    muc_pn = muc_pd = muc_rn = muc_rd = 0
    b3_p = b3_r = 0.0
    b3_n = 0
    ceafe_p = ceafe_r = 0.0
    n_sents = 0

    for nom_idx, scores_np, gold in collected:
        pred = _pred_clusters(nom_idx, scores_np, null_margin)

        pn, pd, rn, rd = _muc(pred, gold)
        muc_pn += pn; muc_pd += pd; muc_rn += rn; muc_rd += rd

        bp, br, bn = _b3(pred, gold)
        b3_p += bp; b3_r += br; b3_n += bn

        cp, cr = _ceafe(pred, gold)
        ceafe_p += cp; ceafe_r += cr
        n_sents += 1

    muc_f = _f1(muc_pn / max(muc_pd, 1), muc_rn / max(muc_rd, 1))
    b3_f = _f1(b3_p / max(b3_n, 1), b3_r / max(b3_n, 1))
    ceafe_f = _f1(ceafe_p / max(n_sents, 1), ceafe_r / max(n_sents, 1))
    return {"MUC": muc_f, "B3": b3_f, "CEAFe": ceafe_f, "CoNLL": (muc_f + b3_f + ceafe_f) / 3, "n_sents": n_sents}


def evaluate_stage1() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    model = DepGraphTransformer(d_model=256, n_heads=8, n_layers=4).to(device)
    model_path = MODELS_DIR / "stage1_graph_transformer.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    data = build_stage1_data()
    _, val_data = kfold_split(data, n_folds=4, fold=0)
    print(f"Val rows: {len(val_data)}")

    collected = _collect_predictions(model, device, val_data)

    print("\n=== Null margin sweep ===")
    for null_margin in [0.0, 0.5, 1.0, 1.5, 2.0]:
        m = _score_predictions(collected, null_margin)
        print(f"  null_margin={null_margin:.1f}  MUC={m['MUC']:.4f}  B3={m['B3']:.4f}  CEAFe={m['CEAFe']:.4f}  CoNLL={m['CoNLL']:.4f}  ({m['n_sents']} sents)")


if __name__ == "__main__":
    evaluate_stage1()
