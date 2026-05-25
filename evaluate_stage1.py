import torch
from pathlib import Path

from disambiguation.signals.stage1_intrasentence import DepGraphTransformer
from disambiguation.signals.train_stage1 import build_stage1_data, kfold_split, run_stage1_eval

CACHE_DIR = Path(__file__).parent / "cache"
MODELS_DIR = CACHE_DIR / "models"


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

    run_stage1_eval(model, device, val_data)


if __name__ == "__main__":
    evaluate_stage1()
