import pickle
from pathlib import Path

import numpy as np
import spacy
import spacy.tokens
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA = Path("data")
SPACY_TRF_DIR = DATA / "spacy_trf"
FEATURES = DATA / "stage1_graph_features.pkl"
PS_SAME_CLAUSE, PS_DOMINATES, PS_APPOS, PS_CONJ, PS_POSS, PS_RELCL = 0, 1, 3, 4, 5, 6
N_SENTS = 3000
NEG_PER_POS = 3
RNG = np.random.default_rng(42)


def bucket_mask(nom_lex: np.ndarray, pair_synt: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    lemma_diff = nom_lex[ii, 0] != nom_lex[jj, 0]
    ps = pair_synt[ii, jj]
    unrelated = (
        (ps[:, PS_SAME_CLAUSE] == 0) & (ps[:, PS_DOMINATES] == 0)
        & (ps[:, PS_APPOS] == 0) & (ps[:, PS_CONJ] == 0)
        & (ps[:, PS_POSS] == 0) & (ps[:, PS_RELCL] == 0)
    )
    return lemma_diff & unrelated


def main() -> None:
    vocab_blank = spacy.blank("en").vocab
    with FEATURES.open("rb") as f:
        data = pickle.load(f)
    print(f"Rows: {len(data)}")

    candidates = []
    for r, row in enumerate(tqdm(data, desc="scan")):
        _, _, _, _, _, nom_idx, nom_lex, pair_synt, gold_ante, _ = row
        M = len(nom_idx)
        if M < 2:
            continue
        ii, jj = np.tril_indices(M, -1)
        if np.any(bucket_mask(nom_lex, pair_synt, ii, jj) & (gold_ante[ii, jj] == 1.0)):
            candidates.append(r)
    print(f"Candidate sentences: {len(candidates)}")
    sampled = [candidates[i] for i in RNG.permutation(len(candidates))[:N_SENTS]]

    # Load spaCy docbins (the authoritative token source — raw datasets have
    # different sentence boundaries due to chunking in run_spacy_cache).
    docbin_cache: dict[tuple[str, str], list] = {}
    rows_meta = []
    bge_vocab: set = set()
    for r in tqdm(sampled, desc="load docs"):
        ds_name, split_name, doc_id, sent_idx = data[r][0]
        key = (ds_name, split_name)
        if key not in docbin_cache:
            path = SPACY_TRF_DIR / f"{ds_name}_{split_name}.spacy"
            db = spacy.tokens.DocBin().from_disk(path)
            docbin_cache[key] = list(db.get_docs(vocab_blank))
        doc = docbin_cache[key][doc_id]
        sent = list(doc.sents)[sent_idx]
        nom_idx = data[r][5]
        words = [sent[int(i)].text for i in nom_idx]
        rows_meta.append((r, words))
        bge_vocab.update(words)

    bge_vocab_sorted = sorted(bge_vocab)
    print(f"Unique head words: {len(bge_vocab_sorted)}")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embs = model.encode(bge_vocab_sorted, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    word2vec = {w: embs[i] for i, w in enumerate(bge_vocab_sorted)}

    X_parts, y_parts, grp_parts, cos_parts, wi_parts, wj_parts = [], [], [], [], [], []
    gid = 0
    for r, words in rows_meta:
        _, _, _, _, _, nom_idx, nom_lex, pair_synt, gold_ante, _ = data[r]
        nom_vecs = np.stack([word2vec[w] for w in words])
        M = len(nom_idx)
        ii, jj = np.tril_indices(M, -1)
        bmask = bucket_mask(nom_lex, pair_synt, ii, jj)
        coref = gold_ante[ii, jj] == 1.0
        pos = np.where(bmask & coref)[0]
        neg = np.where(bmask & ~coref)[0]
        if len(neg) > NEG_PER_POS * len(pos) + 2:
            neg = RNG.choice(neg, NEG_PER_POS * len(pos) + 2, replace=False)
        sel = np.concatenate([pos, neg])
        if len(sel) == 0:
            continue
        vi, vj = nom_vecs[ii[sel]], nom_vecs[jj[sel]]
        X_parts.append(np.concatenate([vi, vj, np.abs(vi - vj), vi * vj], axis=1).astype(np.float32))
        y_parts.append(coref[sel].astype(np.int64))
        grp_parts.append(np.full(len(sel), gid))
        cos_parts.append((vi * vj).sum(1).astype(np.float32))
        wi_parts.append(np.array([words[k] for k in ii[sel]], dtype=object))
        wj_parts.append(np.array([words[k] for k in jj[sel]], dtype=object))
        gid += 1

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(grp_parts)
    cos = np.concatenate(cos_parts)
    wi = np.concatenate(wi_parts)
    wj = np.concatenate(wj_parts)
    print(f"\nPairs: {len(y)}  pos: {int(y.sum())}  neg: {int((1 - y).sum())}  sentences: {gid}  feat_dim: {X.shape[1]}")

    def fit_auc(tr: np.ndarray, te: np.ndarray) -> tuple[float, float, int, int]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        return (
            roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]),
            roc_auc_score(y[te], cos[te]),
            len(te), int(y[te].sum()),
        )

    # Sentence-disjoint split — same words can appear in train and test (word-identity leakage).
    s_tr, s_te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, groups))
    s_probe, s_cos, s_n, s_pos = fit_auc(s_tr, s_te)

    # Word-disjoint split — held-out vocabulary: a pair is in test only if BOTH head words
    # are in the held-out vocab, in train only if both are in the train vocab. No word the
    # probe was trained on appears in a test pair, so it cannot memorise word-pair identity.
    all_words = sorted(set(wi) | set(wj))
    perm = np.random.default_rng(0).permutation(len(all_words))
    test_words = {all_words[i] for i in perm[: len(all_words) // 4]}
    wi_test = np.array([w in test_words for w in wi])
    wj_test = np.array([w in test_words for w in wj])
    w_te = np.where(wi_test & wj_test)[0]
    w_tr = np.where(~wi_test & ~wj_test)[0]
    w_probe, w_cos, w_n, w_pos = fit_auc(w_tr, w_te)

    print("\n=== diff-lemma-unrelated bucket, BGE-small-en STATIC ===")
    print(f"{'split':<20} {'probe AUC':>10} {'cos AUC':>9} {'test pairs':>11} {'pos':>7}")
    print(f"{'sentence-disjoint':<20} {s_probe:>10.3f} {s_cos:>9.3f} {s_n:>11} {s_pos:>7}")
    print(f"{'word-disjoint':<20} {w_probe:>10.3f} {w_cos:>9.3f} {w_n:>11} {w_pos:>7}")
    print(f"\n  reference (trf contextual, sentence-split): 0.912")
    print(f"  leakage estimate (sentence - word disjoint): {s_probe - w_probe:+.3f}")


if __name__ == "__main__":
    main()
