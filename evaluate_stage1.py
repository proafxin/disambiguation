import numpy as np
import torch
from pathlib import Path
from disambiguation.signals.stage1_intrasentence import DepGraphTransformer
from disambiguation.signals.train_stage1 import build_stage1_data

CACHE_DIR = Path(__file__).parent / "cache"
MODELS_DIR = CACHE_DIR / "models"


def _clusters_from_argmax(scores: np.ndarray, M: int) -> list[list[int]]:
    parent = list(range(M))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(M):
        masked = scores[i].copy()
        masked[i:M] = float("-inf")
        ante = int(np.argmax(masked))
        if ante < M:
            parent[find(i)] = find(ante)

    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def _gold_clusters_from_ante(gold_ante: np.ndarray, M: int) -> list[list[int]]:
    parent = list(range(M))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(M):
        for j in range(i):
            if gold_ante[i, j] > 0:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def _pairwise_sets(clusters: list[list[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                pairs.add((cluster[i], cluster[j]))
    return pairs


def evaluate_stage1() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    model = DepGraphTransformer(d_model=256, n_heads=8, n_layers=4).to(device)
    model_path = MODELS_DIR / "stage1_graph_transformer.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
    model.eval()

    _, val_data = build_stage1_data()

    tp = fp = fn = 0
    mention_correct = mention_total = 0

    with torch.no_grad():
        for cat, cont, edges, etypes, nom_idx, gold_ante in val_data:
            M = len(nom_idx)
            cat_t = torch.from_numpy(cat).unsqueeze(0).to(device)
            cont_t = torch.from_numpy(cont).unsqueeze(0).to(device)
            pad_mask = torch.zeros(1, cat.shape[0], dtype=torch.bool, device=device)
            nom_t = torch.from_numpy(nom_idx).unsqueeze(0).to(device)
            nom_mask = torch.ones(1, M, dtype=torch.bool, device=device)
            attn_bias = model._build_batch_attn_bias(
                [(edges, etypes)], [cat.shape[0]], cat.shape[0], device
            )
            scores = model(cat_t, cont_t, pad_mask, nom_t, nom_mask, attn_bias)
            scores_np = scores.squeeze(0).cpu().numpy()

            pred_clusters = _clusters_from_argmax(scores_np, M)
            gold_clusters = _gold_clusters_from_ante(gold_ante, M)

            pred_pairs = _pairwise_sets(pred_clusters)
            gold_pairs = _pairwise_sets(gold_clusters)

            tp += len(pred_pairs & gold_pairs)
            fp += len(pred_pairs - gold_pairs)
            fn += len(gold_pairs - pred_pairs)

            # Mention-level antecedent accuracy (excluding null)
            for i in range(M):
                gold_ants = [j for j in range(i) if gold_ante[i, j] > 0]
                if gold_ants:
                    masked = scores_np[i].copy()
                    masked[i:M] = float("-inf")
                    pred_ante = int(np.argmax(masked))
                    mention_correct += int(pred_ante in gold_ants)
                    mention_total += 1

    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    mention_acc = mention_correct / max(mention_total, 1)

    print(f"\n=== Stage 1 Evaluation (Mention Ranking) ===")
    print(f"  Pairwise Precision: {p:.4f}")
    print(f"  Pairwise Recall:    {r:.4f}")
    print(f"  Pairwise F1:        {f1:.4f}")
    print(f"  Mention Ante Acc:   {mention_acc:.4f}  ({mention_correct}/{mention_total})")


if __name__ == "__main__":
    evaluate_stage1()
