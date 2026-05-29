import re
import subprocess
from pathlib import Path

from disambiguation.paths import SCORER_PL

METRICS = ("muc", "bcub", "ceafe")
_COREF_RE = re.compile(
    r"Coreference: Recall: \(([\d.]+) / ([\d.]+)\).*?Precision: \(([\d.]+) / ([\d.]+)\)",
    re.DOTALL,
)
PRONOUNS = frozenset(
    {
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "who",
        "whom",
        "whose",
        "which",
        "that",
        "this",
        "these",
        "those",
        "there",
    }
)


def mention_type(words: list[str]) -> str:
    head = words[0].lower()
    if head in PRONOUNS:
        return "PRON"
    if words[0][0].isupper() and len(words) <= 5:
        return "PROPN"
    return "NOUN"


# Per-mention agreement class for the scorer: 0 NOUN, 1 PROPN, 2-9 pronoun agreement classes.
N_MENTION_CLASS = 10
_PRON_CLASS = {
    **dict.fromkeys(["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"], 2),  # 1st
    **dict.fromkeys(["you", "your", "yours", "yourself", "yourselves"], 3),  # 2nd
    **dict.fromkeys(["he", "him", "his", "himself"], 4),  # 3sg masc
    **dict.fromkeys(["she", "her", "hers", "herself"], 5),  # 3sg fem
    **dict.fromkeys(["it", "its", "itself"], 6),  # 3sg neuter
    **dict.fromkeys(["they", "them", "their", "theirs", "themselves"], 7),  # 3pl
    **dict.fromkeys(["who", "whom", "whose", "which", "that", "this", "these", "those", "there"], 8),  # rel/dem
}


def mention_class(words: list[str]) -> int:
    if not words:
        return 0
    t = mention_type(words)
    if t == "NOUN":
        return 0
    if t == "PROPN":
        return 1
    return _PRON_CLASS.get(words[0].lower(), 9)  # 9 = other pronoun


def cluster_type_label(cluster: list, sentences: list[list[str]]) -> str:
    # Returns a frozenset of mention types present in the cluster, e.g. frozenset({'PRON','NOUN'})
    types = set()
    for sent_idx, start, end in cluster:
        words = sentences[sent_idx][start:end]
        if words:
            types.add(mention_type(words))
    return frozenset(types)


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
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    m = _COREF_RE.search(out)
    if m is None:
        msg = f"could not parse {metric} output:\n{out}"
        raise RuntimeError(msg)
    rn, rd, pn, pd = (float(x) for x in m.groups())
    r = rn / rd if rd else 0.0
    p = pn / pd if pd else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"recall": r, "precision": p, "f1": f1}


def conll_f1(key_path: Path, response_path: Path) -> dict:
    scores = {metric: run_scorer(key_path, response_path, metric) for metric in METRICS}
    avg = sum(scores[m]["f1"] for m in METRICS) / len(METRICS)
    return {"CoNLL": avg, **{m: scores[m]["f1"] for m in METRICS}}


def conll_f1_by_type(key_docs: list, resp_docs: list, tmp_dir: Path) -> dict[str, dict]:
    # Runs the official scorer on type-filtered subsets of the full document set.
    # Filtering is at the cluster level: a cluster is included in bucket B if its
    # set of mention types matches or contains the types in B. This preserves full
    # cluster structure (all mentions kept) so MUC/B3/CEAFe remain well-defined.
    # key_docs / resp_docs: list of (doc_name, sentences, clusters)
    # Returns {bucket_label: {CoNLL, muc, bcub, ceafe}} for each non-empty bucket.
    buckets: dict[str, frozenset] = {
        "PRON-only": frozenset({"PRON"}),
        "PROPN-only": frozenset({"PROPN"}),
        "NOUN-only": frozenset({"NOUN"}),
        "PRON+NOUN": frozenset({"PRON", "NOUN"}),
        "PRON+PROPN": frozenset({"PRON", "PROPN"}),
        "NOUN+PROPN": frozenset({"NOUN", "PROPN"}),
        "all-mixed": frozenset({"PRON", "NOUN", "PROPN"}),
    }
    results = {}
    for label, target_types in buckets.items():
        filtered_key, filtered_resp = [], []
        for (kname, ksents, kclusters), (_, _, rclusters) in zip(key_docs, resp_docs, strict=False):
            kc = [c for c in kclusters if cluster_type_label(c, ksents) == target_types]
            rc = [c for c in rclusters if cluster_type_label(c, ksents) == target_types]
            if kc or rc:
                filtered_key.append((kname, ksents, kc))
                filtered_resp.append((kname, ksents, rc))
        if not any(kc for _, _, kc in filtered_key):
            continue
        key_path = tmp_dir / f"type_{label}_key.conll"
        resp_path = tmp_dir / f"type_{label}_resp.conll"
        write_conll(key_path, filtered_key)
        write_conll(resp_path, filtered_resp)
        results[label] = conll_f1(key_path, resp_path)
    return results
