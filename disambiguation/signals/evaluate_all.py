import csv
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

from disambiguation.signals.abstract_features import POS_IDS, FEATURE_NAMES, NUM_FEATURES
from disambiguation.signals.generate_episodes import load_episodes_for_docs

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

N_FOLDS = 3
GULLIVERS_DOC_IDX = 36676


def _load(
    docs: np.ndarray,
    source: str,
    window: int,
    ranges: dict,
) -> tuple[np.ndarray, np.ndarray]:
    parts = source.split("+")
    if len(parts) == 1 and parts[0] in ("preco", "litbank", "corefud"):
        return load_episodes_for_docs(parts[0], docs, window)
    all_X, all_y = [], []
    for name, r in ranges.items():
        mask = (docs >= r["start_doc"]) & (docs < r["end_doc"])
        if mask.sum() > 0:
            X, y = load_episodes_for_docs(name, docs[mask], window)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
    if all_X:
        return np.concatenate(all_X), np.concatenate(all_y)
    return np.empty((0, NUM_FEATURES), dtype=np.float32), np.empty(0, dtype=np.int32)


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    if len(X_train) == 0 or len(X_test) == 0:
        return {"error": "empty split"}

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.1,
        subsample=0.8, min_child_weight=10, device="cuda",
        tree_method="hist", random_state=42,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    ap = float(average_precision_score(y_test, y_prob))
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

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
            hop_metrics[f"{origin_name}->{cand_name}"] = {
                "ap": float(average_precision_score(sub_y, sub_prob)),
                "precision": float(sub_y[sub_pred == 1].sum() / max(sub_pred.sum(), 1)),
                "recall": float(sub_y[sub_pred == 1].sum() / max(sub_y.sum(), 1)),
                "positive_count": int(sub_y.sum()),
            }

    importances = model.feature_importances_
    top_features = {FEATURE_NAMES[i]: float(importances[i]) for i in np.argsort(importances)[::-1][:10]}

    return {
        "train_episodes": int(len(X_train)),
        "test_episodes": int(len(X_test)),
        "overall": {"ap": ap, "precision": float(p), "recall": float(r), "f1": float(f1)},
        "by_hop_type": hop_metrics,
        "top_features": top_features,
    }


def average_fold_results(fold_results: list[dict]) -> dict:
    valid = [r for r in fold_results if "error" not in r]
    if not valid:
        return {"error": "all folds failed"}

    def avg(key_path: list) -> float:
        vals = []
        for r in valid:
            v = r
            for k in key_path:
                v = v.get(k, {})
            if isinstance(v, float):
                vals.append(v)
        return float(np.mean(vals)) if vals else 0.0

    def std(key_path: list) -> float:
        vals = []
        for r in valid:
            v = r
            for k in key_path:
                v = v.get(k, {})
            if isinstance(v, float):
                vals.append(v)
        return float(np.std(vals)) if vals else 0.0

    metrics = ["ap", "precision", "recall", "f1"]
    overall = {m: avg(["overall", m]) for m in metrics}
    overall_std = {f"{m}_std": std(["overall", m]) for m in metrics}

    hop_types = set()
    for r in valid:
        hop_types.update(r.get("by_hop_type", {}).keys())

    by_hop = {}
    for ht in hop_types:
        by_hop[ht] = {
            "ap": avg(["by_hop_type", ht, "ap"]),
            "precision": avg(["by_hop_type", ht, "precision"]),
            "recall": avg(["by_hop_type", ht, "recall"]),
        }

    all_features: dict[str, list[float]] = {}
    for r in valid:
        for feat, imp in r.get("top_features", {}).items():
            all_features[feat] = all_features.get(feat, []) + [imp]
    top_features = {k: float(np.mean(v)) for k, v in sorted(all_features.items(), key=lambda x: -np.mean(x[1]))[:10]}

    return {
        "n_folds": len(valid),
        "train_episodes_avg": int(np.mean([r["train_episodes"] for r in valid])),
        "test_episodes_avg": int(np.mean([r["test_episodes"] for r in valid])),
        "overall": {**overall, **overall_std},
        "by_hop_type": by_hop,
        "top_features": top_features,
    }


def _write_table_csv(
    table_name: str,
    experiments: list[str],
    window: int,
    all_results: list[dict],
) -> None:
    path = RESULTS_DIR / f"{table_name}_w{window}.csv"
    rows = [r for r in all_results
            if r.get("experiment", "") in experiments
            and r.get("window_tokens") == window
            and "error" not in r]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "train_source", "test_source",
                         "ap", "ap_std", "precision", "recall", "f1", "f1_std",
                         "pron_pron_r", "pron_noun_r", "pron_propn_r", "noun_propn_r",
                         "n_folds", "train_eps", "test_eps"])
        for r in rows:
            o = r.get("overall", {})
            hops = r.get("by_hop_type", {})
            writer.writerow([
                r["experiment"], r["train_source"], r["test_source"],
                f"{o.get('ap',0):.4f}", f"{o.get('ap_std',0):.4f}",
                f"{o.get('precision',0):.4f}", f"{o.get('recall',0):.4f}",
                f"{o.get('f1',0):.4f}", f"{o.get('f1_std',0):.4f}",
                f"{hops.get('PRON->PRON',{}).get('recall',0):.4f}",
                f"{hops.get('PRON->NOUN',{}).get('recall',0):.4f}",
                f"{hops.get('PRON->PROPN',{}).get('recall',0):.4f}",
                f"{hops.get('NOUN->PROPN',{}).get('recall',0):.4f}",
                r.get("n_folds", 1),
                r.get("train_episodes_avg", 0), r.get("test_episodes_avg", 0),
            ])


def run_all_experiments() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "all_metrics.json"

    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        done_keys = {
            (r["experiment"], r.get("train_source"), r.get("test_source"), r.get("window_tokens"))
            for r in all_results if "error" not in r
        }
        print(f"Resuming: {len(done_keys)} experiments already done")
    else:
        all_results = []
        done_keys = set()

    with open(CACHE_DIR / "dataset_ranges.json") as f:
        ranges = json.load(f)

    preco_docs = np.arange(ranges["preco"]["start_doc"], ranges["preco"]["end_doc"])
    litbank_docs = np.arange(ranges["litbank"]["start_doc"], ranges["litbank"]["end_doc"])
    corefud_docs = np.arange(ranges["corefud"]["start_doc"], ranges["corefud"]["end_doc"])
    all_doc_indices = np.arange(sum(r["end_doc"] - r["start_doc"] for r in ranges.values()))

    rng = np.random.default_rng(42)
    window_lengths = [100, 150, 200]

    def _save() -> None:
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    def _run_kfold(
        experiment: str,
        train_source: str,
        test_source: str,
        train_docs: np.ndarray,
        test_docs: np.ndarray,
        window: int,
        cross_dataset: bool = False,
    ) -> None:
        key = (experiment, train_source, test_source, window)
        if key in done_keys:
            print(f"  Skipping {key} (done)")
            return

        print(f"\n  {experiment}: train={train_source} test={test_source} w={window}")
        start = time.time()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_results = []

        if cross_dataset:
            for fold_idx, (tr_idx, _) in enumerate(kf.split(train_docs)):
                fold_train = train_docs[tr_idx]
                print(f"    Fold {fold_idx+1}/{N_FOLDS}: {len(fold_train)} train docs, {len(test_docs)} test docs")
                X_train, y_train = _load(fold_train, train_source, window, ranges)
                X_test, y_test = _load(test_docs, test_source, window, ranges)
                fold_result = train_and_evaluate(X_train, y_train, X_test, y_test)
                fold_result["fold"] = fold_idx
                fold_results.append(fold_result)
                print(f"      AP={fold_result.get('overall', {}).get('ap', 0):.4f} "
                      f"R={fold_result.get('overall', {}).get('recall', 0):.4f}")
        else:
            for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(train_docs)):
                fold_train = train_docs[tr_idx]
                fold_test = train_docs[te_idx]
                print(f"    Fold {fold_idx+1}/{N_FOLDS}: {len(fold_train)} train, {len(fold_test)} test docs")
                X_train, y_train = _load(fold_train, train_source, window, ranges)
                X_test, y_test = _load(fold_test, test_source, window, ranges)
                fold_result = train_and_evaluate(X_train, y_train, X_test, y_test)
                fold_result["fold"] = fold_idx
                fold_results.append(fold_result)
                print(f"      AP={fold_result.get('overall', {}).get('ap', 0):.4f} "
                      f"R={fold_result.get('overall', {}).get('recall', 0):.4f}")

        averaged = average_fold_results(fold_results)
        averaged.update({
            "experiment": experiment,
            "train_source": train_source,
            "test_source": test_source,
            "window_tokens": window,
            "elapsed_seconds": time.time() - start,
            "fold_details": fold_results,
        })
        all_results.append(averaged)
        _save()

        if "error" not in averaged:
            o = averaged["overall"]
            print(f"    AVG: AP={o['ap']:.4f} P={o['precision']:.4f} "
                  f"R={o['recall']:.4f} F1={o['f1']:.4f} (±{o.get('f1_std', 0):.4f})")

    # TABLE 1: Per-dataset 3-fold CV
    print("\n" + "=" * 70)
    print("TABLE 1: Per-dataset 3-fold CV (ALL docs)")
    print("=" * 70)
    for name, docs in [("preco", preco_docs), ("litbank", litbank_docs), ("corefud", corefud_docs)]:
        shuffled = rng.permutation(docs)
        for window in window_lengths:
            _run_kfold(f"individual_{name}", name, name, shuffled, shuffled, window)

    # TABLE 2: Cross-dataset
    print("\n" + "=" * 70)
    print("TABLE 2: Cross-dataset")
    print("=" * 70)
    dataset_map = {"preco": preco_docs, "litbank": litbank_docs, "corefud": corefud_docs}
    for test_name in ["preco", "litbank", "corefud"]:
        train_names = [n for n in dataset_map if n != test_name]
        train_all = rng.permutation(np.concatenate([dataset_map[n] for n in train_names]))
        test_all = rng.permutation(dataset_map[test_name])
        label = "+".join(train_names)
        for window in window_lengths:
            _run_kfold(f"cross_{test_name}", label, test_name, train_all, test_all, window, cross_dataset=True)

    # TABLE 3: Unified 3-fold CV
    print("\n" + "=" * 70)
    print("TABLE 3: Unified (all 3 datasets, 3-fold CV)")
    print("=" * 70)
    all_docs = rng.permutation(all_doc_indices)
    for window in window_lengths:
        _run_kfold("unified", "all", "all", all_docs, all_docs, window)

    # TABLE 4: Gulliver's held-out
    print("\n" + "=" * 70)
    print("TABLE 4: Gulliver's Travels held-out (261-hop chain)")
    print("=" * 70)
    for window in window_lengths:
        key = ("gullivers_heldout", "litbank_minus_gullivers", "gullivers", window)
        if key in done_keys:
            print(f"  Skipping {key} (done)")
            continue
        print(f"  window={window}...")
        start = time.time()
        litbank_train = litbank_docs[litbank_docs != GULLIVERS_DOC_IDX]
        X_train, y_train = load_episodes_for_docs("litbank", litbank_train, window)
        X_test, y_test = load_episodes_for_docs("gullivers", np.array([GULLIVERS_DOC_IDX]), window)
        result = train_and_evaluate(X_train, y_train, X_test, y_test)
        result.update({
            "experiment": "gullivers_heldout",
            "train_source": "litbank_minus_gullivers",
            "test_source": "gullivers_261hop",
            "window_tokens": window,
            "elapsed_seconds": time.time() - start,
            "n_folds": 1,
            "train_episodes_avg": result.get("train_episodes", 0),
            "test_episodes_avg": result.get("test_episodes", 0),
        })
        all_results.append(result)
        _save()
        if "error" not in result:
            o = result["overall"]
            print(f"    AP={o['ap']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")

    # TABLE 5: Hop-count generalization
    print("\n" + "=" * 70)
    print("TABLE 5: Hop-count generalization (train ≤10 hops, test >50 hops)")
    print("=" * 70)
    for window in window_lengths:
        key = ("hop_generalization", "litbank_chains_le10hops", "litbank_chains_gt50hops", window)
        if key in done_keys:
            print(f"  Skipping {key} (done)")
            continue
        print(f"  window={window}...")
        start = time.time()
        litbank_train = litbank_docs[litbank_docs != GULLIVERS_DOC_IDX]
        X_train, y_train = load_episodes_for_docs("litbank", litbank_train, window, suffix="_maxhop10")
        X_test, y_test = load_episodes_for_docs("litbank_hihop", litbank_docs, window, suffix="_minhop50")
        result = train_and_evaluate(X_train, y_train, X_test, y_test)
        result.update({
            "experiment": "hop_generalization",
            "train_source": "litbank_chains_le10hops",
            "test_source": "litbank_chains_gt50hops",
            "window_tokens": window,
            "elapsed_seconds": time.time() - start,
            "n_folds": 1,
            "train_episodes_avg": result.get("train_episodes", 0),
            "test_episodes_avg": result.get("test_episodes", 0),
        })
        all_results.append(result)
        _save()
        if "error" not in result:
            o = result["overall"]
            print(f"    AP={o['ap']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")

    # Write CSVs
    for window in window_lengths:
        _write_table_csv("table1_individual", [f"individual_{n}" for n in ["preco", "litbank", "corefud"]], window, all_results)
        _write_table_csv("table2_cross", [f"cross_{n}" for n in ["preco", "litbank", "corefud"]], window, all_results)
        _write_table_csv("table3_unified", ["unified"], window, all_results)
        _write_table_csv("table4_gullivers", ["gullivers_heldout"], window, all_results)
        _write_table_csv("table5_hop_generalization", ["hop_generalization"], window, all_results)

    with open(RESULTS_DIR / "metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "train_source", "test_source", "window",
            "ap", "ap_std", "precision", "recall", "f1", "f1_std",
            "pron_pron_recall", "pron_noun_recall", "pron_propn_recall", "noun_propn_recall",
            "n_folds", "train_episodes_avg", "test_episodes_avg",
        ])
        for r in all_results:
            if "error" in r:
                continue
            o = r.get("overall", {})
            hops = r.get("by_hop_type", {})
            writer.writerow([
                r["experiment"], r["train_source"], r["test_source"], r["window_tokens"],
                f"{o.get('ap', 0):.4f}", f"{o.get('ap_std', 0):.4f}",
                f"{o.get('precision', 0):.4f}", f"{o.get('recall', 0):.4f}",
                f"{o.get('f1', 0):.4f}", f"{o.get('f1_std', 0):.4f}",
                f"{hops.get('PRON->PRON', {}).get('recall', 0):.4f}",
                f"{hops.get('PRON->NOUN', {}).get('recall', 0):.4f}",
                f"{hops.get('PRON->PROPN', {}).get('recall', 0):.4f}",
                f"{hops.get('NOUN->PROPN', {}).get('recall', 0):.4f}",
                r.get("n_folds", 0),
                r.get("train_episodes_avg", 0),
                r.get("test_episodes_avg", 0),
            ])

    print(f"\nAll done. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    run_all_experiments()
