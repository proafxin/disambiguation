import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from disambiguation.signals.train_full import CachedData, generate_doc_episodes
from disambiguation.signals.abstract_features import POS_IDS, FEATURE_NAMES, NUM_FEATURES

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def evaluate_split(
    cache: CachedData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    window_tokens: int = 150,
    max_train_docs: int = 2000,
    max_test_docs: int = 500,
) -> dict:
    train_idx = train_indices[:max_train_docs]
    test_idx = test_indices[:max_test_docs]

    # Generate train episodes
    train_features, train_labels = [], []
    for doc_idx in train_idx:
        feats, labels = generate_doc_episodes(cache, int(doc_idx), window_tokens)
        train_features.extend(feats)
        train_labels.extend(labels)

    if not train_features:
        return {"error": "no train episodes"}

    X_train = np.array(train_features)
    y_train = np.array(train_labels, dtype=np.int32)
    del train_features, train_labels

    # Generate test episodes
    test_features, test_labels = [], []
    for doc_idx in test_idx:
        feats, labels = generate_doc_episodes(cache, int(doc_idx), window_tokens)
        test_features.extend(feats)
        test_labels.extend(labels)

    if not test_features:
        return {"error": "no test episodes"}

    X_test = np.array(test_features)
    y_test = np.array(test_labels, dtype=np.int32)
    del test_features, test_labels

    # Train
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.1,
        subsample=0.8, min_child_weight=10, device="cuda",
        tree_method="hist", random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    ap = float(average_precision_score(y_test, y_prob))
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    # By hop type
    pron_id = POS_IDS["PRON"]
    propn_id = POS_IDS["PROPN"]
    noun_id = POS_IDS["NOUN"]

    hop_metrics = {}
    for origin_name, origin_id in [("PRON", pron_id), ("NOUN", noun_id)]:
        for cand_name, cand_id in [("PRON", pron_id), ("NOUN", noun_id), ("PROPN", propn_id)]:
            mask = (X_test[:, 0] == origin_id) & (X_test[:, 12] == cand_id)
            if mask.sum() < 10 or y_test[mask].sum() == 0:
                continue
            sub_y = y_test[mask]
            sub_pred = model.predict(X_test[mask])
            sub_prob = model.predict_proba(X_test[mask])[:, 1]
            sub_ap = float(average_precision_score(sub_y, sub_prob))
            sub_r = float(sub_y[sub_pred == 1].sum() / max(sub_y.sum(), 1))
            sub_p = float(sub_y[sub_pred == 1].sum() / max(sub_pred.sum(), 1)) if sub_pred.sum() > 0 else 0.0
            hop_metrics[f"{origin_name}->{cand_name}"] = {
                "ap": sub_ap, "precision": sub_p, "recall": sub_r,
                "positive_count": int(sub_y.sum()),
            }

    # Feature importance
    importances = model.feature_importances_
    top_features = {FEATURE_NAMES[i]: float(importances[i]) for i in np.argsort(importances)[::-1][:10]}

    return {
        "train_docs": int(len(train_idx)),
        "test_docs": int(len(test_idx)),
        "train_episodes": int(X_train.shape[0]),
        "test_episodes": int(X_test.shape[0]),
        "window_tokens": window_tokens,
        "overall": {"ap": ap, "precision": float(p), "recall": float(r), "f1": float(f1)},
        "by_hop_type": hop_metrics,
        "top_features": top_features,
    }


def run_all_experiments() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = CachedData()

    with open("cache/dataset_ranges.json") as f:
        ranges = json.load(f)

    preco_docs = np.arange(ranges["preco"]["start_doc"], ranges["preco"]["end_doc"])
    litbank_docs = np.arange(ranges["litbank"]["start_doc"], ranges["litbank"]["end_doc"])
    corefud_docs = np.arange(ranges["corefud"]["start_doc"], ranges["corefud"]["end_doc"])

    rng = np.random.default_rng(42)
    window_lengths = [100, 150, 200]

    all_results = []

    # === Table 1: Individual datasets (2:1 split) ===
    print("=" * 60)
    print("TABLE 1: Individual datasets (2:1 train/test)")
    print("=" * 60)

    for name, docs in [("preco", preco_docs), ("litbank", litbank_docs), ("corefud", corefud_docs)]:
        shuffled = rng.permutation(docs)
        split = int(len(shuffled) * 2 / 3)
        train_idx = shuffled[:split]
        test_idx = shuffled[split:]

        for window in window_lengths:
            print(f"\n  {name}, window={window}...")
            result = evaluate_split(cache, train_idx, test_idx, window_tokens=window, max_train_docs=1000, max_test_docs=300)
            result["experiment"] = f"individual_{name}"
            result["train_source"] = name
            result["test_source"] = name
            all_results.append(result)
            if "error" not in result:
                print(f"    AP={result['overall']['ap']:.4f} P={result['overall']['precision']:.4f} R={result['overall']['recall']:.4f} F1={result['overall']['f1']:.4f}")

    # === Table 2: Cross-dataset (train on 2, test on 1) ===
    print("\n" + "=" * 60)
    print("TABLE 2: Cross-dataset transfer (train 2, test 1)")
    print("=" * 60)

    dataset_map = {"preco": preco_docs, "litbank": litbank_docs, "corefud": corefud_docs}
    for test_name in ["preco", "litbank", "corefud"]:
        train_names = [n for n in dataset_map if n != test_name]
        train_idx = np.concatenate([dataset_map[n] for n in train_names])
        train_idx = rng.permutation(train_idx)
        test_idx = rng.permutation(dataset_map[test_name])

        for window in window_lengths:
            print(f"\n  Train({'+'.join(train_names)}) -> Test({test_name}), window={window}...")
            result = evaluate_split(cache, train_idx, test_idx, window_tokens=window, max_train_docs=1000, max_test_docs=300)
            result["experiment"] = f"cross_{test_name}"
            result["train_source"] = "+".join(train_names)
            result["test_source"] = test_name
            all_results.append(result)
            if "error" not in result:
                print(f"    AP={result['overall']['ap']:.4f} P={result['overall']['precision']:.4f} R={result['overall']['recall']:.4f} F1={result['overall']['f1']:.4f}")

    # === Table 3: Unified (all 3, 2:1 split) ===
    print("\n" + "=" * 60)
    print("TABLE 3: Unified (all datasets, 2:1 split)")
    print("=" * 60)

    all_docs = np.arange(len(cache.doc_boundaries))
    shuffled = rng.permutation(all_docs)
    split = int(len(shuffled) * 2 / 3)
    train_idx = shuffled[:split]
    test_idx = shuffled[split:]

    for window in window_lengths:
        print(f"\n  Unified, window={window}...")
        result = evaluate_split(cache, train_idx, test_idx, window_tokens=window, max_train_docs=2000, max_test_docs=500)
        result["experiment"] = "unified"
        result["train_source"] = "all"
        result["test_source"] = "all"
        all_results.append(result)
        if "error" not in result:
            print(f"    AP={result['overall']['ap']:.4f} P={result['overall']['precision']:.4f} R={result['overall']['recall']:.4f} F1={result['overall']['f1']:.4f}")

    # Save all results
    with open(RESULTS_DIR / "all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Export as CSV for charting
    with open(RESULTS_DIR / "metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "train_source", "test_source", "window",
            "ap", "precision", "recall", "f1",
            "pron_pron_recall", "pron_noun_recall", "pron_propn_recall",
            "train_docs", "test_docs", "train_episodes", "test_episodes",
        ])
        for r in all_results:
            if "error" in r:
                continue
            hops = r.get("by_hop_type", {})
            writer.writerow([
                r["experiment"], r["train_source"], r["test_source"], r["window_tokens"],
                f"{r['overall']['ap']:.4f}", f"{r['overall']['precision']:.4f}",
                f"{r['overall']['recall']:.4f}", f"{r['overall']['f1']:.4f}",
                f"{hops.get('PRON->PRON', {}).get('recall', 0):.4f}",
                f"{hops.get('PRON->NOUN', {}).get('recall', 0):.4f}",
                f"{hops.get('PRON->PROPN', {}).get('recall', 0):.4f}",
                r["train_docs"], r["test_docs"], r["train_episodes"], r["test_episodes"],
            ])

    print(f"\n\nResults saved to {RESULTS_DIR}/")
    print(f"  all_metrics.json (full details)")
    print(f"  metrics_summary.csv (for charts)")


if __name__ == "__main__":
    run_all_experiments()
