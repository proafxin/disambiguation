import numpy as np
import torch
from pathlib import Path
from disambiguation.signals.stage1_intrasentence import DepGraphTransformer
from disambiguation.signals.train_stage1 import build_stage1_data

CACHE_DIR = Path(__file__).parent / "cache"
MODELS_DIR = CACHE_DIR / "models"


def diagnose() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Diagnosing Stage 1 on {device}\n")

    model = DepGraphTransformer(d_model=256, n_heads=8, n_layers=4).to(device)
    model_path = MODELS_DIR / "stage1_graph_transformer.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
    model.eval()

    _, val_data = build_stage1_data()

    with torch.no_grad():
        for n_ex, (cat, cont, edges, etypes, nom_idx, gold_ante) in enumerate(val_data):
            if n_ex >= 10:
                break

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
            scores_np = scores.squeeze(0).cpu().numpy()  # (M, M+1)

            # Softmax over allowed antecedents (causal + null)
            probs = np.zeros_like(scores_np)
            for i in range(M):
                allowed = list(range(i)) + [M]
                s = scores_np[i, allowed]
                s = s - s.max()
                e = np.exp(s)
                probs[i, allowed] = e / e.sum()

            print(f"Example {n_ex}: {M} nominals, {cat.shape[0]} tokens, {edges.shape[1]} edges")
            for i in range(M):
                gold_ants = [j for j in range(i) if gold_ante[i, j] > 0]
                masked = scores_np[i].copy()
                masked[i:M] = float("-inf")
                pred_ante = int(np.argmax(masked))
                pred_label = "null" if pred_ante == M else str(pred_ante)
                gold_label = "null" if not gold_ants else str(gold_ants)
                correct = "✓" if (not gold_ants and pred_ante == M) or (gold_ants and pred_ante in gold_ants) else "✗"
                print(f"  mention {i}: pred={pred_label} gold={gold_label} {correct}  null_prob={probs[i, M]:.3f}")
            print()


if __name__ == "__main__":
    diagnose()
