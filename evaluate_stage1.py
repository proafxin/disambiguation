import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from disambiguation.signals.stage1_intrasentence import MiniTransformer
from disambiguation.signals.train_stage1 import build_stage1_data

CACHE_DIR = Path(__file__).parent / "cache"
MODELS_DIR = CACHE_DIR / "models"

def evaluate_stage1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    model = MiniTransformer(n_heads=4, n_layers=1).to(device)
    model_path = MODELS_DIR / "stage1_mini_transformer.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true = []
    y_prob = []

    _, val_data = build_stage1_data()

    with torch.no_grad():
        for feature_matrix, nom_idx, pair_labels in val_data:
            feature_t = torch.from_numpy(feature_matrix).float().unsqueeze(0).to(device)
            nom_t = torch.from_numpy(nom_idx).long().to(device)

            pair_scores = model(feature_t, nom_t).squeeze(0)
            pair_probs = torch.sigmoid(pair_scores)

            off = ~np.eye(pair_labels.shape[0], dtype=bool)
            y_true.append(pair_labels[off])
            y_prob.append(pair_probs.cpu().numpy()[off])

    if not y_true:
        print("No examples found!")
        return

    y_true_flat = np.concatenate(y_true)
    y_prob_flat = np.concatenate(y_prob)
    y_pred_flat = (y_prob_flat > 0.5).astype(int)
    y_true_binary = (y_true_flat > 0).astype(int)

    ap = average_precision_score(y_true_binary, y_prob_flat)
    p = precision_score(y_true_binary, y_pred_flat, zero_division=0)
    r = recall_score(y_true_binary, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_flat, zero_division=0)

    print(f"\n=== Stage 1 Evaluation ===")
    print(f"  AP:        {ap:.4f}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Pairs: {len(y_true_flat):,} (pos: {int(y_true_binary.sum()):,})")

if __name__ == "__main__":
    evaluate_stage1()
