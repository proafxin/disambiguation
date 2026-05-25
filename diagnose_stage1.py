import numpy as np
import torch
from collections import defaultdict

from disambiguation.signals.abstract_features import POS_IDS
from disambiguation.signals.stage1_intrasentence import (
    NominalCorefScorer, decode_clusters, MODELS_DIR, CKPT_NAME,
)
from disambiguation.signals.train_stage1 import (
    build_stage1_data, build_bge_cache, kfold_split, _gold_clusters, _links, _muc, _b3, _ceafe, _f1,
)

THRESHOLD = 0.5  # decode threshold (pick from the eval sweep, not re-tuned here)
ID2POS = {v: k for k, v in POS_IDS.items()}
NOM_POS = {"PRON", "NOUN", "PROPN"}


def _acc() -> dict:
    return {
        "n": 0, "gl": 0, "pl": 0, "tp": 0,
        "muc": [0, 0, 0, 0], "b3": [0.0, 0.0, 0], "ce": [0.0, 0.0],
        "pos_g": defaultdict(int), "pos_r": defaultdict(int),
        "noun_g": defaultdict(int), "noun_r": defaultdict(int),
        "propn_g": defaultdict(int), "propn_r": defaultdict(int),
    }


def diagnose() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Diagnosing Stage 1 on {device} (threshold={THRESHOLD})\n")

    data = build_stage1_data()
    word2id, matrix = build_bge_cache(data)
    _, val = kfold_split(data, n_folds=4, fold=0)
    print(f"Val rows: {len(val)}\n")

    model = NominalCorefScorer().to(device)
    path = MODELS_DIR / CKPT_NAME
    if not path.exists():
        print(f"Model not found at {path}")
        return
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    by_ds: dict[str, dict] = defaultdict(_acc)
    for row in val:
        key, words, nom_idx, gold_cid, pos_ids, nom_lex = row
        L = len(words)
        ids = np.fromiter((word2id[w] for w in words), dtype=np.int64, count=L)
        bge = torch.from_numpy(matrix[ids][None]).to(device)
        pad = torch.zeros(1, L, dtype=torch.bool, device=device)
        nt = torch.from_numpy(nom_idx[None]).to(device)
        nm = torch.ones(1, len(nom_idx), dtype=torch.bool, device=device)
        with torch.inference_mode():
            logits = model(bge, pad, nt, nm).squeeze(0).float().cpu().numpy()
        pred = decode_clusters(nom_idx, logits, THRESHOLD)
        gold = _gold_clusters(nom_idx, gold_cid)

        a = by_ds[key[0]]
        a["n"] += 1
        m = _muc(pred, gold)
        for i in range(4):
            a["muc"][i] += m[i]
        bp, br, bn = _b3(pred, gold)
        a["b3"][0] += bp; a["b3"][1] += br; a["b3"][2] += bn
        cp, cr = _ceafe(pred, gold)
        a["ce"][0] += cp; a["ce"][1] += cr
        gl, pl = _links(gold), _links(pred)
        a["gl"] += len(gl); a["pl"] += len(pl); a["tp"] += len(gl & pl)

        pos_of = {int(nom_idx[k]): ID2POS.get(int(pos_ids[k]), "OTHER") for k in range(len(nom_idx))}
        lem_of = {int(nom_idx[k]): int(nom_lex[k, 0]) for k in range(len(nom_idx))}
        sur_of = {int(nom_idx[k]): int(nom_lex[k, 1]) for k in range(len(nom_idx))}
        for (x, y) in gl:
            ana, ant = max(x, y), min(x, y)
            rec = (x, y) in pl
            pos = pos_of.get(ana, "OTHER")
            bucket = pos if pos in NOM_POS else "OTHER"
            a["pos_g"][bucket] += 1
            if rec:
                a["pos_r"][bucket] += 1
            if pos == "NOUN":
                k2 = "same_lemma" if lem_of.get(ana) == lem_of.get(ant) else "diff_lemma"
                a["noun_g"][k2] += 1
                if rec:
                    a["noun_r"][k2] += 1
            elif pos == "PROPN":
                k2 = "same_surface" if sur_of.get(ana) == sur_of.get(ant) else "diff_surface"
                a["propn_g"][k2] += 1
                if rec:
                    a["propn_r"][k2] += 1

    header = f"{'dataset':<11} {'sents':>6} {'gLinks':>7} {'linkP':>6} {'linkR':>6} {'linkF':>6}  {'MUC':>5} {'B3':>5} {'CEAFe':>5} {'CoNLL':>6}"
    print(header)
    print("-" * len(header))

    def report(name: str, a: dict) -> None:
        lp = a["tp"] / max(a["pl"], 1)
        lr = a["tp"] / max(a["gl"], 1)
        muc = _f1(a["muc"][0] / max(a["muc"][1], 1), a["muc"][2] / max(a["muc"][3], 1))
        b3 = _f1(a["b3"][0] / max(a["b3"][2], 1), a["b3"][1] / max(a["b3"][2], 1))
        ce = _f1(a["ce"][0] / max(a["n"], 1), a["ce"][1] / max(a["n"], 1))
        print(f"{name:<11} {a['n']:>6} {a['gl']:>7} {lp:>6.3f} {lr:>6.3f} {_f1(lp, lr):>6.3f}  {muc:>5.3f} {b3:>5.3f} {ce:>5.3f} {(muc + b3 + ce) / 3:>6.3f}")

    total = _acc()
    for ds in sorted(by_ds):
        report(ds, by_ds[ds])
        for k in ("n", "gl", "pl", "tp"):
            total[k] += by_ds[ds][k]
        for i in range(4):
            total["muc"][i] += by_ds[ds]["muc"][i]
        for i in range(3):
            total["b3"][i] += by_ds[ds]["b3"][i]
        for i in range(2):
            total["ce"][i] += by_ds[ds]["ce"][i]
        for field in ("pos_g", "pos_r", "noun_g", "noun_r", "propn_g", "propn_r"):
            for bk, v in by_ds[ds][field].items():
                total[field][bk] += v
    print("-" * len(header))
    report("ALL", total)

    print("\n=== Link recall by anaphor POS ===")
    for p in ("PRON", "NOUN", "PROPN", "OTHER"):
        g, r = total["pos_g"].get(p, 0), total["pos_r"].get(p, 0)
        print(f"  {p:<8} {r}/{g} ({r / max(g, 1):.2f})")

    print("\n=== NOUN-anaphor recall by lemma match with antecedent ===")
    for k in ("same_lemma", "diff_lemma"):
        g, r = total["noun_g"].get(k, 0), total["noun_r"].get(k, 0)
        print(f"  {k:<14} {r}/{g} ({r / max(g, 1):.2f})")

    print("\n=== PROPN-anaphor recall by surface match with antecedent ===")
    for k in ("same_surface", "diff_surface"):
        g, r = total["propn_g"].get(k, 0), total["propn_r"].get(k, 0)
        print(f"  {k:<14} {r}/{g} ({r / max(g, 1):.2f})")


if __name__ == "__main__":
    diagnose()
