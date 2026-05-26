import re
import subprocess
from pathlib import Path

SCORER_PL = Path.home() / "Projects" / "reference-coreference-scorers" / "scorer.pl"
METRICS = ("muc", "bcub", "ceafe")
_COREF_RE = re.compile(
    r"Coreference: Recall: \(([\d.]+) / ([\d.]+)\).*?Precision: \(([\d.]+) / ([\d.]+)\)",
    re.DOTALL,
)


def _coref_column(starts: list, ends: list, singles: list) -> str:
    # starts: (cid, gend) multi-token opens here; ends: (cid, gstart) multi-token closes here.
    # Open longest spans first, close innermost (latest-started) spans first for correct nesting.
    parts = [f"({cid}" for cid, _ in sorted(starts, key=lambda x: -x[1])]
    parts += [f"({cid})" for cid in singles]
    parts += [f"{cid})" for cid, _ in sorted(ends, key=lambda x: -x[1])]
    return "|".join(parts) if parts else "-"


def write_conll(path: Path, docs: list) -> None:
    # docs: list of (doc_name, sentences, clusters); sentences = list[list[str]];
    # clusters = list of clusters, each a list of (sent_idx, start, end) with end exclusive.
    lines: list[str] = []
    for doc_name, sentences, clusters in docs:
        offsets, off = [], 0
        for sent in sentences:
            offsets.append(off)
            off += len(sent)
        starts: dict[int, list] = {}
        ends: dict[int, list] = {}
        singles: dict[int, list] = {}
        for cid, cluster in enumerate(clusters):
            for sent_idx, start, end in cluster:
                g0 = offsets[sent_idx] + start
                g1 = offsets[sent_idx] + end - 1
                if g1 == g0:
                    singles.setdefault(g0, []).append(cid)
                else:
                    starts.setdefault(g0, []).append((cid, g1))
                    ends.setdefault(g1, []).append((cid, g0))
        lines.append(f"#begin document ({doc_name}); part 000")
        gtok = 0
        for sent in sentences:
            for ti, word in enumerate(sent):
                col = _coref_column(starts.get(gtok, []), ends.get(gtok, []), singles.get(gtok, []))
                lines.append(f"{doc_name}\t0\t{ti}\t{word}\t{col}")
                gtok += 1
            lines.append("")
        lines.append("#end document")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scorer(key_path: Path, response_path: Path, metric: str) -> dict:
    out = subprocess.run(
        ["perl", str(SCORER_PL), metric, str(key_path), str(response_path), "none"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = _COREF_RE.search(out)
    if m is None:
        raise RuntimeError(f"could not parse {metric} output:\n{out}")
    rn, rd, pn, pd = (float(x) for x in m.groups())
    r = rn / rd if rd else 0.0
    p = pn / pd if pd else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"recall": r, "precision": p, "f1": f1}


def conll_f1(key_path: Path, response_path: Path) -> dict:
    scores = {metric: run_scorer(key_path, response_path, metric) for metric in METRICS}
    avg = sum(scores[m]["f1"] for m in METRICS) / len(METRICS)
    return {"CoNLL": avg, **{m: scores[m]["f1"] for m in METRICS}}
