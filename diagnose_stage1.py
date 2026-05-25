import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from disambiguation.signals.abstract_features import POS_IDS
from disambiguation.signals.stage1_intrasentence import DepGraphTransformer
from disambiguation.signals.train_stage1 import (
    build_stage1_data, kfold_split, _pred_clusters, _muc, _b3, _ceafe, _f1,
)

CACHE_DIR = Path(__file__).parent / "cache"
MODELS_DIR = CACHE_DIR / "models"
MARGIN = 2.0  # frozen decode threshold (selected on the sweep, not re-tuned here)
ID2POS = {v: k for k, v in POS_IDS.items()}
NOM_POS = {"PRON", "NOUN", "PROPN"}


def _links(clusters: list) -> set:
    pairs = set()
    for c in clusters:
        members = sorted(c)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def _new_acc() -> dict:
    return {
        "n_sents": 0, "n_gold_links": 0, "n_pred_links": 0, "tp_links": 0,
        "muc_pn": 0, "muc_pd": 0, "muc_rn": 0, "muc_rd": 0,
        "b3_p": 0.0, "b3_r": 0.0, "b3_n": 0,
        "ceafe_p": 0.0, "ceafe_r": 0.0,
        "pos_gold": defaultdict(int), "pos_recalled": defaultdict(int),
        # noun-anaphor links split by whether antecedent shares the lemma
        "noun_gold": defaultdict(int), "noun_recalled": defaultdict(int),
        # propn-anaphor links split by whether antecedent shares the surface form
        "propn_gold": defaultdict(int), "propn_recalled": defaultdict(int),
        # diff-lemma noun links split by syntactic relation between the pair
        "ndiff_synt_gold": defaultdict(int), "ndiff_synt_recalled": defaultdict(int),
    }


# pair_synt column indices (see _pair_syntactic_features)
PS_SAME_CLAUSE, PS_DOMINATES, PS_APPOS, PS_CONJ, PS_POSS, PS_RELCL = 0, 1, 3, 4, 5, 6


def _synt_bucket(ps_row: np.ndarray) -> str:
    if ps_row[PS_APPOS] or ps_row[PS_CONJ] or ps_row[PS_POSS] or ps_row[PS_RELCL]:
        return "marked_arc"      # appos/conj/poss/relcl — labeled structural relation
    if ps_row[PS_DOMINATES]:
        return "dominates"       # one head c-commands the other (incl. copular/predicative)
    if ps_row[PS_SAME_CLAUSE]:
        return "same_clause"     # co-clausal but no direct relation
    return "unrelated"           # different clauses, no syntactic link → needs semantics


def diagnose() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Diagnosing Stage 1 on {device} (margin={MARGIN})\n")

    model = DepGraphTransformer(d_model=256, n_heads=8, n_layers=4).to(device)
    model_path = MODELS_DIR / "stage1_graph_transformer.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
    model.eval()

    data = build_stage1_data()
    _, val_data = kfold_split(data, n_folds=4, fold=0)
    print(f"Val rows: {len(val_data)}\n")

    by_ds: dict[str, dict] = defaultdict(_new_acc)

    for row in tqdm(val_data, desc="diagnose"):
        key, cat, cont, edges, etypes, nom_idx, nom_lex, pair_synt, gold_ante, gold_clusters = row
        ds = key[0]
        M = len(nom_idx)
        with torch.inference_mode():
            attn_bias = model._build_batch_attn_bias([(edges, etypes)], [cat.shape[0]], cat.shape[0], device)
            scores = model(
                torch.from_numpy(cat).unsqueeze(0).to(device),
                torch.from_numpy(cont).unsqueeze(0).to(device),
                torch.zeros(1, cat.shape[0], dtype=torch.bool, device=device),
                torch.from_numpy(nom_idx).unsqueeze(0).to(device),
                torch.ones(1, M, dtype=torch.bool, device=device),
                torch.from_numpy(nom_lex).unsqueeze(0).to(device),
                torch.from_numpy(pair_synt).unsqueeze(0).to(device),
                attn_bias,
            )
        scores_np = scores.squeeze(0).cpu().numpy()
        pred = _pred_clusters(nom_idx, scores_np, MARGIN)
        gold = gold_clusters

        a = by_ds[ds]
        a["n_sents"] += 1

        pn, pd, rn, rd = _muc(pred, gold)
        a["muc_pn"] += pn; a["muc_pd"] += pd; a["muc_rn"] += rn; a["muc_rd"] += rd
        bp, br, bn = _b3(pred, gold)
        a["b3_p"] += bp; a["b3_r"] += br; a["b3_n"] += bn
        cp, cr = _ceafe(pred, gold)
        a["ceafe_p"] += cp; a["ceafe_r"] += cr

        gold_links = _links(gold)
        pred_links = _links(pred)
        a["n_gold_links"] += len(gold_links)
        a["n_pred_links"] += len(pred_links)
        a["tp_links"] += len(gold_links & pred_links)

        lemma_of = {int(nom_idx[k]): int(nom_lex[k, 0]) for k in range(M)}
        surf_of = {int(nom_idx[k]): int(nom_lex[k, 1]) for k in range(M)}
        nidx_of = {int(nom_idx[k]): k for k in range(M)}

        # Recall by anaphor POS (anaphor = later token in the pair).
        for (x, y) in gold_links:
            anaphor, ante = max(x, y), min(x, y)
            recalled = (x, y) in pred_links
            pos = ID2POS.get(int(cat[anaphor, 0]), "OTHER")
            bucket = pos if pos in NOM_POS else "OTHER"
            a["pos_gold"][bucket] += 1
            if recalled:
                a["pos_recalled"][bucket] += 1

            # Split noun/propn anaphora by exact match with the antecedent.
            if pos == "NOUN":
                if anaphor in lemma_of and ante in lemma_of:
                    key2 = "same_lemma" if lemma_of[anaphor] == lemma_of[ante] else "diff_lemma"
                else:
                    key2 = "ante_non_nominal"
                a["noun_gold"][key2] += 1
                if recalled:
                    a["noun_recalled"][key2] += 1
                if key2 == "diff_lemma":
                    sb = _synt_bucket(pair_synt[nidx_of[anaphor], nidx_of[ante]])
                    a["ndiff_synt_gold"][sb] += 1
                    if recalled:
                        a["ndiff_synt_recalled"][sb] += 1
            elif pos == "PROPN":
                if anaphor in surf_of and ante in surf_of:
                    key2 = "same_surface" if surf_of[anaphor] == surf_of[ante] else "diff_surface"
                else:
                    key2 = "ante_non_nominal"
                a["propn_gold"][key2] += 1
                if recalled:
                    a["propn_recalled"][key2] += 1

    header = f"{'dataset':<11} {'sents':>6} {'gLinks':>7} {'linkP':>6} {'linkR':>6} {'linkF':>6}  {'MUC':>5} {'B3':>5} {'CEAFe':>5} {'CoNLL':>6}"
    print("\n" + header)
    print("-" * len(header))

    def report(name: str, a: dict) -> None:
        lp = a["tp_links"] / max(a["n_pred_links"], 1)
        lr = a["tp_links"] / max(a["n_gold_links"], 1)
        lf = _f1(lp, lr)
        muc = _f1(a["muc_pn"] / max(a["muc_pd"], 1), a["muc_rn"] / max(a["muc_rd"], 1))
        b3 = _f1(a["b3_p"] / max(a["b3_n"], 1), a["b3_r"] / max(a["b3_n"], 1))
        ceafe = _f1(a["ceafe_p"] / max(a["n_sents"], 1), a["ceafe_r"] / max(a["n_sents"], 1))
        conll = (muc + b3 + ceafe) / 3
        print(f"{name:<11} {a['n_sents']:>6} {a['n_gold_links']:>7} {lp:>6.3f} {lr:>6.3f} {lf:>6.3f}  {muc:>5.3f} {b3:>5.3f} {ceafe:>5.3f} {conll:>6.3f}")

    total = _new_acc()
    for ds in sorted(by_ds):
        report(ds, by_ds[ds])
        for k in ("n_sents", "n_gold_links", "n_pred_links", "tp_links", "muc_pn", "muc_pd", "muc_rn", "muc_rd", "b3_p", "b3_r", "b3_n", "ceafe_p", "ceafe_r"):
            total[k] += by_ds[ds][k]
        for field in ("pos_gold", "pos_recalled", "noun_gold", "noun_recalled", "propn_gold", "propn_recalled", "ndiff_synt_gold", "ndiff_synt_recalled"):
            for bucket, v in by_ds[ds][field].items():
                total[field][bucket] += v
    print("-" * len(header))
    report("ALL", total)

    print("\n=== Link recall by anaphor POS (the later mention) ===")
    print(f"{'dataset':<11} " + " ".join(f"{p:>14}" for p in ("PRON", "NOUN", "PROPN", "OTHER")))
    for ds in sorted(by_ds) + ["ALL"]:
        a = by_ds[ds] if ds != "ALL" else total
        cells = []
        for p in ("PRON", "NOUN", "PROPN", "OTHER"):
            g = a["pos_gold"].get(p, 0)
            r = a["pos_recalled"].get(p, 0)
            cells.append(f"{r}/{g} ({r / max(g, 1):.2f})".rjust(14))
        print(f"{ds:<11} " + " ".join(cells))

    print("\n=== NOUN-anaphor recall split by exact lemma match with antecedent (ALL) ===")
    for k in ("same_lemma", "diff_lemma", "ante_non_nominal"):
        g = total["noun_gold"].get(k, 0)
        r = total["noun_recalled"].get(k, 0)
        print(f"  {k:<18} recall {r}/{g} ({r / max(g, 1):.2f})")

    print("\n=== PROPN-anaphor recall split by exact surface match with antecedent (ALL) ===")
    for k in ("same_surface", "diff_surface", "ante_non_nominal"):
        g = total["propn_gold"].get(k, 0)
        r = total["propn_recalled"].get(k, 0)
        print(f"  {k:<18} recall {r}/{g} ({r / max(g, 1):.2f})")

    print("\n=== DIFF-LEMMA noun links split by syntactic relation (ALL) ===")
    print("  (marked_arc/dominates/same_clause = structurally reachable; unrelated = needs semantics)")
    for k in ("marked_arc", "dominates", "same_clause", "unrelated"):
        g = total["ndiff_synt_gold"].get(k, 0)
        r = total["ndiff_synt_recalled"].get(k, 0)
        print(f"  {k:<14} recall {r}/{g} ({r / max(g, 1):.2f})")


if __name__ == "__main__":
    diagnose()
