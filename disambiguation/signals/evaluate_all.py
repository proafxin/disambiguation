import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from disambiguation.signals.abstract_features import FEATURE_NAMES, NUM_FEATURES, POS_IDS
from disambiguation.signals.generate_episodes import (
    GULLIVERS_DOC_IDX, episode_dir, manifest_path,
)

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
MODELS_DIR = CACHE_DIR / "models"

DATASETS = ["preco", "litbank", "corefud", "conll2012"]
WINDOW_LENGTHS = [100, 150, 200]

XGB_PARAMS = dict(
    n_estimators=2000, max_depth=10, learning_rate=0.1,
    subsample=0.8, min_child_weight=10, max_bin=1024,
    device="cuda", tree_method="hist", random_state=42,
)

XGB_PARAMS_QUICK = dict(
    n_estimators=20, max_depth=6, learning_rate=0.1,
    subsample=0.8, min_child_weight=5, max_bin=256,
    device="cuda", tree_method="hist", random_state=42,
)


def _build_flat_store(window: int, ranges: dict) -> dict:
    store: dict = {}
    t0 = time.time()
    total_mb = 0.0
    for ds in ranges:
        mpath = manifest_path(ds, window)
        if not mpath.exists():
            continue
        with open(mpath) as f:
            m = json.load(f)
        out_dir = episode_dir(ds, window)
        X_flat = np.load(out_dir / "X.npy")
        y_flat = np.load(out_dir / "y.npy")
        r_flat = np.load(out_dir / "ranks.npy")
        doc_rows = {int(k): (v[0], v[1]) for k, v in m["doc_map"].items()}
        store[ds] = {"X": X_flat, "y": y_flat, "ranks": r_flat, "doc_rows": doc_rows}
        total_mb += X_flat.nbytes / 1024 / 1024
    print(f"  Loaded {total_mb:.0f} MB into RAM in {time.time() - t0:.1f}s")
    return store


def _slice(
    store: dict,
    ds: str,
    doc_indices: np.ndarray,
    min_rank: int | None = None,
    max_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if ds not in store or len(doc_indices) == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float32), np.empty(0, dtype=np.int32)
    entry = store[ds]
    X_flat, y_flat, r_flat, doc_rows = entry["X"], entry["y"], entry["ranks"], entry["doc_rows"]
    need_ranks = min_rank is not None or max_rank is not None
    X_parts, y_parts, r_parts = [], [], []
    for doc_idx in doc_indices:
        rng_val = doc_rows.get(int(doc_idx))
        if rng_val is not None:
            rs, re = rng_val
            X_parts.append(X_flat[rs:re])
            y_parts.append(y_flat[rs:re])
            if need_ranks:
                r_parts.append(r_flat[rs:re])
    if not X_parts:
        return np.empty((0, NUM_FEATURES), dtype=np.float32), np.empty(0, dtype=np.int32)
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    if need_ranks:
        r = np.concatenate(r_parts)
        mask = np.ones(len(r), dtype=bool)
        if min_rank is not None:
            mask &= r >= min_rank
        if max_rank is not None:
            mask &= r <= max_rank
        X, y = X[mask], y[mask]
    return X, y


def _evaluate(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    if len(X) == 0 or len(y) == 0:
        return {"error": "empty", "episodes": 0, "positives": 0}
    if y.sum() == 0:
        return {"error": "no positives", "episodes": int(len(X)), "positives": 0}

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    ap = float(average_precision_score(y, y_prob))
    p = float(precision_score(y, y_pred, average="binary", zero_division=0))
    r = float(recall_score(y, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y, y_pred, average="binary", zero_division=0))

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
                "precision": float(precision_score(sy, sp, average="binary", zero_division=0)),
                "recall": float(recall_score(sy, sp, average="binary", zero_division=0)),
                "f1": float(f1_score(sy, sp, average="binary", zero_division=0)),
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
        "overall": {"ap": ap, "precision": p, "recall": r, "f1": f1},
        "by_hop_type": hop_metrics,
        "top_features": top_features,
    }


def _train_or_load(
    model_key: str,
    window: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_dir: Path,
    quick: bool = False,
) -> xgb.XGBClassifier:
    path = models_dir / f"{model_key}_w{window}.ubj"
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)
    params = XGB_PARAMS_QUICK if quick else XGB_PARAMS
    early_stopping = None if quick else 30
    model = xgb.XGBClassifier(**params, scale_pos_weight=spw, early_stopping_rounds=early_stopping)
    if path.exists():
        model.load_model(str(path))
        print(f"    Loaded: {path.name}")
    else:
        print(f"    Training on {len(X_train):,} episodes (pos_rate={n_pos/max(n_pos+n_neg,1):.3f}, spw={spw:.1f})...")
        rng = np.random.default_rng(42)
        n_val = max(100 if quick else 1000, len(X_train) // 10)
        idx = rng.permutation(len(X_train))
        X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
        X_tr, y_tr = X_train[idx[n_val:]], y_train[idx[n_val:]]
        fit_kwargs = {} if quick else {"eval_set": [(X_val, y_val)], "verbose": 50}
        model.fit(X_tr, y_tr, **fit_kwargs)
        model.save_model(str(path))
        trees = getattr(model, "best_iteration", model.n_estimators - 1) + 1
        print(f"    Saved: {path.name} ({trees} trees)")
    return model


def _get_docs(ds: str, ranges: dict, exclude_gullivers: bool = False) -> np.ndarray:
    r = ranges.get(ds)
    if r is None:
        return np.empty(0, dtype=np.int64)
    docs = np.arange(r["start_doc"], r["end_doc"])
    if exclude_gullivers and ds == "litbank":
        docs = docs[docs != GULLIVERS_DOC_IDX]
    return docs


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


def run_all_experiments(quick: bool = False) -> None:
    models_dir = MODELS_DIR / ("quick" if quick else "full")
    results_dir = RESULTS_DIR / ("quick" if quick else "full")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "all_metrics.json"

    if results_path.exists() and not quick:
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

    if quick:
        print("QUICK TEST MODE — 200 docs/dataset, 20 trees, w=100 only")

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

        model = _train_or_load(model_key, window, X_train, y_train, models_dir, quick)
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

    windows = [100] if quick else WINDOW_LENGTHS
    quick_cap = 200

    for window in windows:
        print(f"\n{'='*70}\nWINDOW = {window} tokens\n{'='*70}")

        store = _build_flat_store(window, ranges)

        X_gull, y_gull = _slice(store, "litbank", np.array([GULLIVERS_DOC_IDX]))
        if len(X_gull) == 0:
            print(f"  WARNING: no Gulliver's episodes for w={window} — benchmark will be empty")

        def _cap(docs: np.ndarray) -> np.ndarray:
            return docs[:quick_cap] if quick else docs

        # --- Table 1: Individual 2:1 split (4 datasets) ---
        # Splits cached for T3 reuse — T3 train/test = union of T1 splits per dataset
        print("\n--- Table 1: Individual 2:1 split ---")
        t1_train: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        t1_test: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ds in DATASETS:
            if ds not in ranges:
                continue
            docs = _cap(rng.permutation(_get_docs(ds, ranges, exclude_gullivers=True)))
            n = len(docs) * 2 // 3
            X_tr, y_tr = _slice(store, ds, docs[:n])
            X_te, y_te = _slice(store, ds, docs[n:])
            t1_train[ds] = (X_tr, y_tr)
            t1_test[ds] = (X_te, y_te)
            _run(f"t1_{ds}", f"t1_{ds}", 1, ds, ds, window,
                 X_tr, y_tr, X_te, y_te, X_gull, y_gull)

        # --- Table 2: Blind cross-dataset validation (train on 3, test on 1) ---
        print("\n--- Table 2: Blind cross-dataset validation ---")
        for test_ds in DATASETS:
            if test_ds not in ranges:
                continue
            train_Xs, train_ys = [], []
            for td in DATASETS:
                if td == test_ds or td not in ranges:
                    continue
                X, y = _slice(store, td, _cap(_get_docs(td, ranges, exclude_gullivers=True)))
                if len(X) > 0:
                    train_Xs.append(X)
                    train_ys.append(y)
            if not train_Xs:
                continue
            X_tr, y_tr = _shuffle_concat(train_Xs, train_ys, rng)
            X_te, y_te = _slice(store, test_ds, _cap(_get_docs(test_ds, ranges)))
            train_label = "+".join(d for d in DATASETS if d != test_ds and d in ranges)
            _run(f"t2_test_{test_ds}", f"t2_test_{test_ds}", 2,
                 train_label, test_ds, window,
                 X_tr, y_tr, X_te, y_te, X_gull, y_gull)

        # --- Table 3: Unified stratified — reuses T1 splits directly ---
        print("\n--- Table 3: Unified stratified ---")
        tr_Xs = [t1_train[ds][0] for ds in DATASETS if ds in t1_train and len(t1_train[ds][0]) > 0]
        tr_ys = [t1_train[ds][1] for ds in DATASETS if ds in t1_train and len(t1_train[ds][0]) > 0]
        te_Xs = [t1_test[ds][0] for ds in DATASETS if ds in t1_test and len(t1_test[ds][0]) > 0]
        te_ys = [t1_test[ds][1] for ds in DATASETS if ds in t1_test and len(t1_test[ds][0]) > 0]
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
            train_docs = _cap(_get_docs(ds, ranges, exclude_gullivers=True))
            all_docs = _cap(_get_docs(ds, ranges))
            X_tr, y_tr = _slice(store, ds, train_docs, max_rank=50)
            X_te, y_te = _slice(store, ds, all_docs, min_rank=51)
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

        del store

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


def probe_features(n_train_docs: int = 500, n_test_docs: int = 250, window: int = 100) -> None:
    with open(CACHE_DIR / "dataset_ranges.json") as f:
        ranges = json.load(f)

    store = _build_flat_store(window, ranges)
    available = [ds for ds in DATASETS if ds in store]
    if not available:
        print("No episodes found — run generate_episodes first")
        return

    rng = np.random.default_rng(42)
    tr_Xs, tr_ys, te_Xs, te_ys = [], [], [], []
    for ds in available:
        docs = rng.permutation(_get_docs(ds, ranges, exclude_gullivers=True))
        n_tr = min(n_train_docs, len(docs) * 2 // 3)
        n_te = min(n_test_docs, len(docs) - n_tr)
        X_tr_ds, y_tr_ds = _slice(store, ds, docs[:n_tr])
        X_te_ds, y_te_ds = _slice(store, ds, docs[n_tr : n_tr + n_te])
        if len(X_tr_ds) > 0:
            tr_Xs.append(X_tr_ds)
            tr_ys.append(y_tr_ds)
        if len(X_te_ds) > 0:
            te_Xs.append(X_te_ds)
            te_ys.append(y_te_ds)

    X_tr, y_tr = _shuffle_concat(tr_Xs, tr_ys, rng)
    X_te, y_te = np.concatenate(te_Xs), np.concatenate(te_ys)
    print(f"Probe: {len(X_tr):,} train eps (pos={int(y_tr.sum())}), "
          f"{len(X_te):,} test eps (pos={int(y_te.sum())})")

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    spw = n_neg / max(n_pos, 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, min_child_weight=5, max_bin=512,
        device="cuda", tree_method="hist", random_state=42,
        scale_pos_weight=spw,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    print(f"Test: AP={average_precision_score(y_te, y_prob):.4f}  "
          f"P={precision_score(y_te, y_pred, average='binary', zero_division=0):.4f}  "
          f"R={recall_score(y_te, y_pred, average='binary', zero_division=0):.4f}  "
          f"F1={f1_score(y_te, y_pred, average='binary', zero_division=0):.4f}")

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    print(f"\n{'Rank':>4}  {'Feature':<35}  {'Importance':>10}")
    print("-" * 55)
    for rank, i in enumerate(order, 1):
        print(f"{rank:>4}  {FEATURE_NAMES[i]:<35}  {importances[i]:>10.4f}")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe_features()
    else:
        run_all_experiments(quick="--quick" in sys.argv)
