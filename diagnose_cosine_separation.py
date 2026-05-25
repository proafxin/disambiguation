import pickle
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm

DATA = Path(__file__).parent / "data" / "stage1_graph_features.pkl"
COS = 8  # pair_synt channel holding contextual trf cosine
PS_SAME_CLAUSE, PS_DOMINATES, PS_APPOS, PS_CONJ, PS_POSS, PS_RELCL = 0, 1, 3, 4, 5, 6
NEG_RATE = 0.02  # subsample non-coref pairs to keep memory bounded
RNG = np.random.default_rng(42)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    with DATA.open("rb") as f:
        data = pickle.load(f)
    print(f"Rows: {len(data)}")

    buckets = ["all", "diff_lemma", "diff_lemma_unrelated"]
    pos = {b: [] for b in buckets}
    neg = {b: [] for b in buckets}
    skipped = 0

    for row in tqdm(data, desc="scan"):
        _, _, _, _, _, nom_idx, nom_lex, pair_synt, gold_ante, _ = row
        M = len(nom_idx)
        if M < 2:
            continue
        cosm = pair_synt[:, :, COS]
        if not np.any(cosm != 0):  # cosine cache miss → all-zero, skip
            skipped += 1
            continue

        ii, jj = np.tril_indices(M, -1)  # i > j
        cos = cosm[ii, jj].astype(np.float32)
        coref = gold_ante[ii, jj] == 1.0
        lemma_diff = nom_lex[ii, 0] != nom_lex[jj, 0]
        ps = pair_synt[ii, jj]
        unrelated = (
            (ps[:, PS_SAME_CLAUSE] == 0) & (ps[:, PS_DOMINATES] == 0)
            & (ps[:, PS_APPOS] == 0) & (ps[:, PS_CONJ] == 0)
            & (ps[:, PS_POSS] == 0) & (ps[:, PS_RELCL] == 0)
        )

        masks = {
            "all": np.ones(len(cos), dtype=bool),
            "diff_lemma": lemma_diff,
            "diff_lemma_unrelated": lemma_diff & unrelated,
        }
        keep_neg = RNG.random(len(cos)) < NEG_RATE
        for b, m in masks.items():
            pos[b].append(cos[m & coref])
            neg[b].append(cos[m & ~coref & keep_neg])

    print(f"Skipped (cosine cache miss): {skipped}\n")
    print(f"{'bucket':<22} {'n_pos':>8} {'n_neg*':>8} {'mean_pos':>9} {'mean_neg':>9} {'AUC':>6}")
    print("-" * 64)
    for b in buckets:
        p = np.concatenate(pos[b]) if pos[b] else np.array([])
        n = np.concatenate(neg[b]) if neg[b] else np.array([])
        mp = float(p.mean()) if len(p) else float("nan")
        mn = float(n.mean()) if len(n) else float("nan")
        print(f"{b:<22} {len(p):>8} {len(n):>8} {mp:>9.4f} {mn:>9.4f} {auc(p, n):>6.3f}")
    print("\n* n_neg subsampled at NEG_RATE; AUC is unaffected by negative count.")


if __name__ == "__main__":
    main()
