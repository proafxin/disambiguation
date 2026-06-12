from disambiguation.data import _win_names
from disambiguation.paths import MODELS_DIR
from disambiguation.stage_a import train_stage_a
from disambiguation.stage_b import train_stage_b

if __name__ == "__main__":
    # sent_aligned=True runs the sentence-packed-window variant end-to-end (own ctx cache, head,
    # clusters); Stage A and Stage B share the exact same windows. Set False for the fixed-K system.
    window, channel, sent_aligned, subset = 256, "both", True, "all8k"
    # Canonical Stage A (raw=False, projected + distance = 89.48). raw is abandoned -- it overfits
    # (Stage A -1.58 vs projected); the projection regularizes. Instead we widen ONLY the GNN's
    # RoBERTa projection in Stage B from 512 to 1024 (ctx_proj), since RoBERTa is the dominant and
    # most-compressed channel (2048->512, a 4x cut). Stage A already runs ctx at 1024; this matches it.
    raw, hidden, use_distance = False, 1024, True
    ctx_proj = None  # ctx 512->1024 tested flat (86.26 vs 86.46); 512 is near-optimal, reverted
    # stage_a_error_analysis(window=window, subset=subset, channel=channel, sent_aligned=sent_aligned)
    # Dependence/causal check for the removed agreement channel: linear≈ceiling ⇒ agreement is already
    # linearly recoverable from RoBERTa, so concatenating it was redundant (RESEARCH.md §4.2).
    # agreement_dependence_probe(window=window, subset=subset, sent_aligned=sent_aligned)
    # Stage A frozen head for this (channel, sent_aligned[, raw][, nodist]) must exist; train it if missing.
    frozen_name = (
        _win_names(window, subset, channel, sent_aligned=sent_aligned)[1]
        + ("_raw" if raw else "")
        + ("_nodist" if not use_distance else "")
    )
    head_path = MODELS_DIR / f"{frozen_name}.pt"
    if not head_path.exists():
        train_stage_a(
            window=window,
            subset=subset,
            channel=channel,
            sent_aligned=sent_aligned,
            raw=raw,
            hidden=hidden,
            use_distance=use_distance,
        )

    train_stage_b(
        window=window,
        subset=subset,
        channel=channel,
        member_pool="lse",
        neg_ratio=5,
        lexical=True,
        sent_aligned=sent_aligned,
        raw=raw,
        hidden=hidden,
        use_distance=use_distance,
        ctx_proj=ctx_proj,
        calibrate=True,  # #1: per-dataset merge-threshold sweep (fixes LitBank/PreCo over-merge)
    )
