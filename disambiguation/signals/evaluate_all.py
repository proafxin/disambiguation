import csv
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from disambiguation.signals.abstract_features import FEATURE_NAMES, NUM_FEATURES, POS_IDS
from disambiguation.signals.generate_episodes import HELD_OUT_DOC_INDICES, load_episodes_for_docs

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
MODELS_DIR = CACHE_DIR / "models"

DATASETS = ["preco", "litbank", "corefud", "conll2012"]
GULLIVERS_DOC_IDX = 36676
WINDOW_LENGTHS = [100, 150, 200]

XGB_PARAMS = dict(
    n_estimators=500, max_depth=10, learning_rate=0.1,
    subsample=0.8, min_child_weight=10, device="cuda",
    tree_method="hist", random_state=42,
)


def _evaluate(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    if len(X) == 0 or len(y) == 0:
        return {"error": "empty", "episodes": 0, "positives": 0}
    if y.sum() == 0:
        return {"error": "no positives", "episodes": int(len(X)), "positives": 0}

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    ap = float(average_precision_score(y, y_prob))
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary")

    pron_id = POS_IDS["PRON"]
    propn_id = POS_IDS["PROPN"]
    noun_id = POS_IDS["NOUN"]
    hop_metrics: dict = {}
    for on, oi in [("PRON", pron_id), ("NOUN", noun_id)]:
        for cn, ci in [("PRON", pron_id), ("NOUN", noun_id), ("PROPN", propn_id)]:
            mask = (X[:, 0] == oi) & (X[:, 9] == ci)
            if mask.sum() < 10 or y[mask].sum() == 0:
                continue
            sy = y[mask]
            sp = model.predict(X[mask])
            sb = model.predict_proba(X[mask])[:, 1]
            hop_metrics[f"{on}->{cn}"] = {
                "ap": float(average_precision_score(sy, sb)),
                "precision": float(sy[sp == 1].sum() / max(sp.sum(), 1)),
                "recall": float(sy[sp == 1].sum() / max(sy.sum(), 1)),
                "count": int(sy.sum()),
            }

    importances = model.feature_importances_
    top_features = {
        FEATURE_NAMES[i]: float(importances[i])
        for i in np.argsort(importances)[::-1][:10]
    }

    return {
        "episodes": int(len(X)),
        "positives": int(y.sum()),
        "overall": {"ap": ap, "precision": float(p), "recall": float(r), "f1": float(f1)},
        "by_hop_type": hop_metrics,
        "top_features": top_features,
    }


def _train_or_load(model_key: str, window: int, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    path = MODELS_DIR / f"{model_key}_w{window}.ubj"
    model = xgb.XGBClassifier(**XGB_PARAMS)
    if path.exists():
        model.load_model(str(path))
        print(f"    Loaded: {path.name}")
    else:
        print(f"    Training on {len(X_train):,} episodes...")
        model.fit(X_train, y_train)
        model.save_model(str(path))
        print(f"    Saved: {path.name}")
    return model


def _get_docs(ds: str, ranges: dict, exclude_gullivers: bool = False) -> np.ndarray:
    r = ranges.get(ds)
    if r is None:
        return np.empty(0, dtype=np.int64)
    docs = np.arange(r["start_doc"], r["end_doc"])
    if exclude_gullivers and ds == "litbank":
        docs = docs[docs != GULLIVERS_DOC_IDX]
    return docs


def _load_eps(ds: str, docs: np.ndarray, window: int, suffix: str = "") -> tuple[np.ndarray, np.ndarray]:
    if len(docs) == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float32), np.empty(0, dtype=np.int32)
    return load_episodes_for_docs(ds, docs, window, suffix=suffix)


def _shuffle_concat(Xs: list, ys: list, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _write_table_csv(table_num: int, all_results: list, window: int) -> None:
    path = RESULTS_DIR / f"table{table_num}_w{window}.csv"
    rows = [r for r in all_results
            if r.get("table") == table_num and r.get("window_tokens") == window
            and "error" not in r]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "experiment", "train_source", "test_source",
            "ap", "precision", "recall", "f1",
            "pron_pron_r", "pron_noun_r", "pron_propn_r", "noun_propn_r",
            "test_episodes",
            "gull_ap", "gull_precision", "gull_recall", "gull_f1", "gull_episodes",
        ])
        for r in rows:
            o = r.get("overall", {})
            h = r.get("by_hop_type", {})
            g = r.get("gullivers", {})
            go = g.get("overall", {})
            w.writerow([
                r["experiment"], r.get("train_source", ""), r.get("test_source", ""),
                f"{o.get('ap', 0):.4f}", f"{o.get('precision', 0):.4f}",
                f"{o.get('recall', 0):.4f}", f"{o.get('f1', 0):.4f}",
                f"{h.get('PRON->PRON', {}).get('recall', 0):.4f}",
                f"{h.get('PRON->NOUN', {}).get('recall', 0):.4f}",
                f"{h.get('PRON->PROPN', {}).get('recall', 0):.4f}",
                f"{h.get('NOUN->PROPN', {}).get('recall', 0):.4f}",
                r.get("episodes", 0),
                f"{go.get('ap', 0):.4f}", f"{go.get('precision', 0):.4f}",
                f"{go.get('recall', 0):.4f}", f"{go.get('f1', 0):.4f}",
                g.get("episodes", 0),
            ])


def run_all_experiments() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "all_metrics.json"

    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        done_keys = {
            (r["experiment"], r.get("window_tokens"))
            for r in all_results
            if "error" not in r and "gullivers" in r
        }
        print(f"Resuming: {len(done_keys)} experiments done")
    else:
        all_results = []
        done_keys = set()

    with open(CACHE_DIR / "dataset_ranges.json") as f:
        ranges = json.load(f)

    rng = np.random.default_rng(42)

    def _save() -> None:
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    def _run(
        experiment: str,
        model_key: str,
        table: int,
        train_source: str,
        test_source: str,
        window: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_gull: np.ndarray,
        y_gull: np.ndarray,
    ) -> None:
        key = (experiment, window)
        if key in done_keys:
            print(f"  Skip {experiment} w={window}")
            return
        start = time.time()
        print(f"\n  [T{table}] {experiment} w={window}: "
              f"{len(X_train):,} train, {len(X_test):,} test eps")

        if len(X_train) == 0 or y_train.sum() == 0:
            print(f"    SKIP: empty training data")
            return

        model = _train_or_load(model_key, window, X_train, y_train)
        test_m = _evaluate(model, X_test, y_test)
        gull_m = _evaluate(model, X_gull, y_gull)

        if "error" not in test_m:
            o = test_m["overall"]
            go = gull_m.get("overall", {})
            print(f"    test: AP={o['ap']:.4f} P={o['precision']:.4f} "
                  f"R={o['recall']:.4f} F1={o['f1']:.4f}")
            print(f"    gull: AP={go.get('ap', 0):.4f} F1={go.get('f1', 0):.4f} "
                  f"({gull_m.get('episodes', 0)} eps)")

        record = {
            "experiment": experiment,
            "table": table,
            "train_source": train_source,
            "test_source": test_source,
            "window_tokens": window,
            "elapsed": time.time() - start,
            **test_m,
            "gullivers": gull_m,
        }
        all_results.append(record)
        done_keys.add(key)
        _save()

    for window in WINDOW_LENGTHS:
        print(f"\n{'='*70}\nWINDOW = {window} tokens\n{'='*70}")

        X_gull, y_gull = _load_eps("gullivers", np.array([GULLIVERS_DOC_IDX]), window)
        if len(X_gull) == 0:
            print(f"  WARNING: no Gulliver's episodes for w={window} — benchmark will be empty")

        # --- Table 1: Individual 2:1 split (4 datasets) ---
        print("\n--- Table 1: Individual 2:1 split ---")
        for ds in DATASETS:
            if ds not in ranges:
                continue
            docs = rng.permutation(_get_docs(ds, ranges, exclude_gullivers=True))
            n = len(docs) * 2 // 3
            X_tr, y_tr = _load_eps(ds, docs[:n], window)
            X_te, y_te = _load_eps(ds, docs[n:], window)
            _run(f"t1_{ds}", f"t1_{ds}", 1, ds, ds, window,
                 X_tr, y_tr, X_te, y_te, X_gull, y_gull)

        # --- Table 2: Cross-dataset (train on 3, test on 1) ---
        print("\n--- Table 2: Cross-dataset (3→1) ---")
        for test_ds in DATASETS:
            if test_ds not in ranges:
                continue
            train_Xs, train_ys = [], []
            for td in DATASETS:
                if td == test_ds or td not in ranges:
                    continue
                X, y = _load_eps(td, _get_docs(td, ranges, exclude_gullivers=True), window)
                if len(X) > 0:
                    train_Xs.append(X)
                    train_ys.append(y)
            if not train_Xs:
                continue
            X_tr, y_tr = _shuffle_concat(train_Xs, train_ys, rng)
            X_te, y_te = _load_eps(test_ds, _get_docs(test_ds, ranges), window)
            train_label = "+".join(d for d in DATASETS if d != test_ds and d in ranges)
            _run(f"t2_test_{test_ds}", f"t2_test_{test_ds}", 2,
                 train_label, test_ds, window,
                 X_tr, y_tr, X_te, y_te, X_gull, y_gull)

        # --- Table 3: Unified stratified (2:1 per dataset, then mix) ---
        print("\n--- Table 3: Unified stratified ---")
        tr_Xs, tr_ys, te_Xs, te_ys = [], [], [], []
        for ds in DATASETS:
            if ds not in ranges:
                continue
            docs = rng.permutation(_get_docs(ds, ranges, exclude_gullivers=True))
            n = len(docs) * 2 // 3
            X_tr, y_tr = _load_eps(ds, docs[:n], window)
            X_te, y_te = _load_eps(ds, docs[n:], window)
            if len(X_tr) > 0:
                tr_Xs.append(X_tr)
                tr_ys.append(y_tr)
            if len(X_te) > 0:
                te_Xs.append(X_te)
                te_ys.append(y_te)
        if tr_Xs and te_Xs:
            X_tr, y_tr = _shuffle_concat(tr_Xs, tr_ys, rng)
            _run("t3_unified", "t3_unified", 3,
                 "all_stratified_2:1", "all_stratified_2:1", window,
                 X_tr, y_tr, np.concatenate(te_Xs), np.concatenate(te_ys),
                 X_gull, y_gull)

        # --- Table 4: Hop-count generalization (≤50 train, >50 test, all datasets) ---
        print("\n--- Table 4: Hop-count gen (≤50 hops train, >50 hops test) ---")
        tr_Xs, tr_ys, te_Xs, te_ys = [], [], [], []
        for ds in DATASETS:
            if ds not in ranges:
                continue
            train_docs = _get_docs(ds, ranges, exclude_gullivers=True)
            all_docs = _get_docs(ds, ranges)
            X_tr, y_tr = _load_eps(ds, train_docs, window, suffix="_maxhop50")
            X_te, y_te = _load_eps(ds, all_docs, window, suffix="_minhop51")
            if len(X_tr) > 0:
                tr_Xs.append(X_tr)
                tr_ys.append(y_tr)
            if len(X_te) > 0:
                te_Xs.append(X_te)
                te_ys.append(y_te)
        if tr_Xs and te_Xs:
            X_tr, y_tr = _shuffle_concat(tr_Xs, tr_ys, rng)
            _run("t4_hopgen", "t4_hopgen", 4,
                 "all_le50hops", "all_gt50hops", window,
                 X_tr, y_tr, np.concatenate(te_Xs), np.concatenate(te_ys),
                 X_gull, y_gull)

    # Write per-table CSVs
    for window in WINDOW_LENGTHS:
        for t in [1, 2, 3, 4]:
            _write_table_csv(t, all_results, window)

    # Summary CSV
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "experiment", "window", "train_source", "test_source",
                    "ap", "f1", "gull_ap", "gull_f1", "test_eps", "gull_eps"])
        for r in sorted(all_results, key=lambda x: (x.get("table", 0), x.get("window_tokens", 0))):
            if "error" in r:
                continue
            o = r.get("overall", {})
            go = r.get("gullivers", {}).get("overall", {})
            w.writerow([
                r.get("table"), r["experiment"], r.get("window_tokens"),
                r.get("train_source"), r.get("test_source"),
                f"{o.get('ap', 0):.4f}", f"{o.get('f1', 0):.4f}",
                f"{go.get('ap', 0):.4f}", f"{go.get('f1', 0):.4f}",
                r.get("episodes", 0), r.get("gullivers", {}).get("episodes", 0),
            ])

    print(f"\nAll done. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    run_all_experiments()
