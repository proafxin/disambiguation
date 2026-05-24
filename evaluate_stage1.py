import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from disambiguation.signals.stage1_intrasentence import MiniTransformer
from disambiguation.signals.train_stage1 import generate_stage1_training_data

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

    with torch.no_grad():
        for feature_matrix, pair_labels in generate_stage1_training_data():
            if pair_labels.shape[0] < 2:
                continue

            feature_t = torch.from_numpy(feature_matrix).float().unsqueeze(0).to(device)
            labels_t = torch.from_numpy(pair_labels).float().to(device)

            pair_scores = model(feature_t).squeeze(0)

            if pair_scores.shape != labels_t.shape:
                continue

            pair_probs = torch.sigmoid(pair_scores)

            y_true.append(labels_t.cpu().numpy())
            y_prob.append(pair_probs.cpu().numpy())

    if not y_true:
        print("No examples found!")
        return

    y_true_flat = np.concatenate([y.flatten() for y in y_true])
    y_prob_flat = np.concatenate([y.flatten() for y in y_prob])
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
