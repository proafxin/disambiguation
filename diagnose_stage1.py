import numpy as np
import torch
from pathlib import Path
from disambiguation.signals.stage1_intrasentence import MiniTransformer
from disambiguation.signals.train_stage1 import generate_stage1_training_data

CACHE_DIR = Path(__file__).parent / "cache"
MODELS_DIR = CACHE_DIR / "models"

def diagnose():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Diagnosing Stage 1 on {device}\n")

    model = MiniTransformer(n_heads=4, n_layers=1).to(device)
    model_path = MODELS_DIR / "stage1_mini_transformer.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    label_counts = {}
    pred_stats = []
    n_examples = 0

    with torch.no_grad():
        for feature_matrix, pair_labels in generate_stage1_training_data():
            if n_examples >= 10:
                break

            feature_t = torch.from_numpy(feature_matrix).float().unsqueeze(0).to(device)
            labels_t = torch.from_numpy(pair_labels).float().to(device)

            pair_scores = model(feature_t).squeeze(0)
            pair_probs = torch.sigmoid(pair_scores)

            labels_flat = pair_labels.flatten()
            probs_flat = pair_probs.cpu().numpy().flatten()

            for val in labels_flat:
                label_counts[float(val)] = label_counts.get(float(val), 0) + 1

            pred_stats.append({
                'mean_prob': probs_flat.mean(),
                'max_prob': probs_flat.max(),
                'min_prob': probs_flat.min(),
                'std_prob': probs_flat.std(),
                'n_high_prob': (probs_flat > 0.5).sum(),
            })

            if n_examples == 0:
                print(f"Example {n_examples}:")
                print(f"  Feature matrix shape: {feature_matrix.shape}")
                print(f"  Pair labels shape: {pair_labels.shape}")
                print(f"  Label distribution: {dict(zip(*np.unique(labels_flat, return_counts=True)))}")
                print(f"  Prob mean: {probs_flat.mean():.6f}, std: {probs_flat.std():.6f}")
                print()

            n_examples += 1

    if n_examples == 0:
        print("No examples found!")
        return

    print(f"Total labels (first {n_examples} examples):")
    total = sum(label_counts.values())
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = count / total * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    print(f"\nPrediction stats:")
    for i, stats in enumerate(pred_stats):
        print(f"  Ex {i}: mean={stats['mean_prob']:.4f}, std={stats['std_prob']:.4f}, range=[{stats['min_prob']:.4f}, {stats['max_prob']:.4f}]")

if __name__ == "__main__":
    diagnose()
