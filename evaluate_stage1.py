import torch

from disambiguation.signals.stage1_intrasentence import NominalCorefScorer, MODELS_DIR, CKPT_NAME
from disambiguation.signals.train_stage1 import build_stage1_data, build_bge_cache, kfold_split, run_eval


def evaluate_stage1() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    data = build_stage1_data()
    word2id, matrix = build_bge_cache(data)
    _, val_data = kfold_split(data, n_folds=4, fold=0)
    print(f"Val rows: {len(val_data)}")

    model = NominalCorefScorer().to(device)
    path = MODELS_DIR / CKPT_NAME
    if not path.exists():
        print(f"Model not found at {path}")
        return
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    run_eval(model, val_data, word2id, matrix, device)


if __name__ == "__main__":
    evaluate_stage1()
