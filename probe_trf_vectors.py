import pickle
from pathlib import Path

import numpy as np
import spacy
import spacy.tokens
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA = Path("data")
FEATURES = DATA / "stage1_graph_features.pkl"
COS = 8
PS_SAME_CLAUSE, PS_DOMINATES, PS_APPOS, PS_CONJ, PS_POSS, PS_RELCL = 0, 1, 3, 4, 5, 6
N_SENTS = 3000          # diff-lemma-unrelated sentences to sample
NEG_PER_POS = 3         # cap bucket negatives per sentence relative to positives
RNG = np.random.default_rng(42)


def get_sentences(example: dict, ds_name: str) -> list:
    if ds_name == "corefud":
        return [[t["form"] for t in s["tokens"]] for s in example["sentences"]]
    return example["sentences"]


def make_doc(nlp: spacy.Language, tokens: list) -> spacy.tokens.Doc:
    n = len(tokens)
    return spacy.tokens.Doc(nlp.vocab, words=list(tokens), sent_starts=[True] + [False] * (n - 1))


def pool_tokens(doc: spacy.tokens.Doc, device: torch.device) -> torch.Tensor:
    lhs = doc._.trf_data.last_hidden_layer_state
    if hasattr(lhs.lengths, "__dlpack__"):
        lengths = torch.utils.dlpack.from_dlpack(lhs.lengths).to(torch.long)
        raw = torch.utils.dlpack.from_dlpack(lhs.dataXd).to(torch.float32)
    else:
        lengths = torch.as_tensor(np.asarray(lhs.lengths), dtype=torch.long, device=device)
        raw = torch.as_tensor(np.asarray(lhs.dataXd), dtype=torch.float32, device=device)
    n_tokens = lengths.shape[0]
    tok_ids = torch.repeat_interleave(torch.arange(n_tokens, device=raw.device), lengths)
    tok = torch.zeros(n_tokens, raw.shape[1], dtype=torch.float32, device=raw.device)
    tok.scatter_add_(0, tok_ids.unsqueeze(1).expand(-1, raw.shape[1]), raw)
    return F.normalize(tok, dim=1)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with FEATURES.open("rb") as f:
        data = pickle.load(f)
    print(f"Rows: {len(data)}  device: {device}")

    # First pass: rows with >=1 coref diff-lemma-unrelated pair (guarantees positives).
    candidates = []
    for r, row in enumerate(tqdm(data, desc="scan")):
        _, _, _, _, _, nom_idx, nom_lex, pair_synt, gold_ante, _ = row
        M = len(nom_idx)
        if M < 2:
            continue
        ii, jj = np.tril_indices(M, -1)
        bmask = bucket_mask(nom_lex, pair_synt, ii, jj)
        coref = gold_ante[ii, jj] == 1.0
        if np.any(bmask & coref):
            candidates.append(r)
    print(f"Candidate sentences (>=1 bucket coref pair): {len(candidates)}")

    pick = RNG.permutation(len(candidates))[:N_SENTS]
    sampled = [candidates[i] for i in pick]

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["senter"])
    trf = nlp.get_pipe("transformer")

    raw_splits: dict = {}
    docs, metas = [], []
    for r in sampled:
        key = data[r][0]
        ds_name, split_name, doc_id, sent_idx = key
        if (ds_name, split_name) not in raw_splits:
            raw_splits[(ds_name, split_name)] = load_from_disk(str(DATA / ds_name))[split_name]
        tokens = get_sentences(raw_splits[(ds_name, split_name)][doc_id], ds_name)[sent_idx]
        docs.append(make_doc(nlp, tokens))
        metas.append(r)

    X_parts, y_parts, grp_parts, cos_parts = [], [], [], []
    gid = 0
    for r, doc in tqdm(zip(metas, trf.pipe(docs, batch_size=64)), total=len(docs), desc="trf"):
        if doc._.trf_data is None:
            continue
        _, _, _, _, _, nom_idx, nom_lex, pair_synt, gold_ante, _ = data[r]
        tok = pool_tokens(doc, device)
        doc._.trf_data = None
        if int(nom_idx.max()) >= tok.shape[0]:
            continue
        nom_vecs = tok[torch.as_tensor(nom_idx, device=tok.device)].cpu().numpy()

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

        vi = nom_vecs[ii[sel]]
        vj = nom_vecs[jj[sel]]
        feat = np.concatenate([vi, vj, np.abs(vi - vj), vi * vj], axis=1)
        X_parts.append(feat.astype(np.float32))
        y_parts.append(coref[sel].astype(np.int64))
        grp_parts.append(np.full(len(sel), gid))
        cos_parts.append((vi * vj).sum(1).astype(np.float32))
        gid += 1

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(grp_parts)
    cos = np.concatenate(cos_parts)
    print(f"\nPairs: {len(y)}  pos: {int(y.sum())}  neg: {int((1 - y).sum())}  sentences: {gid}  feat_dim: {X.shape[1]}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X[tr], y[tr])
    probe_auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
    cos_auc = roc_auc_score(y[te], cos[te])

    print("\n=== diff-lemma-unrelated bucket, held-out test ===")
    print(f"  cosine (v_i . v_j) AUC:        {cos_auc:.3f}")
    print(f"  learned linear probe AUC:      {probe_auc:.3f}")
    print(f"  test pairs: {len(te)}  (pos {int(y[te].sum())} / neg {int((1 - y[te]).sum())})")


if __name__ == "__main__":
    main()
