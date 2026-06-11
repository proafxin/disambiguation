from disambiguation.data import _win_names
from disambiguation.paths import MODELS_DIR
from disambiguation.stage_a import stage_a_error_analysis, train_stage_a
from disambiguation.stage_b import train_stage_b

if __name__ == "__main__":
    # sent_aligned=True runs the sentence-packed-window variant end-to-end (own ctx cache, head,
    # clusters); Stage A and Stage B share the exact same windows. Set False for the fixed-K system.
    window, channel, sent_aligned, subset = 256, "both", True, "all8k"
    stage_a_error_analysis(window=window, subset=subset, channel=channel, sent_aligned=sent_aligned)
    # Stage A frozen head for this (channel, sent_aligned) must exist; train it if missing.
    head_path = MODELS_DIR / f"{_win_names(window, subset, channel, sent_aligned=sent_aligned)[1]}.pt"
    if not head_path.exists():
        train_stage_a(window=window, subset=subset, channel=channel, sent_aligned=sent_aligned)
    # GNN Stage B — current best config (lse member pooling + concatenated lexical = 86.46 sent-aligned).
    # Reuses an existing matcher checkpoint for eval; pass force=True to retrain over it.
    train_stage_b(
        window=window,
        subset=subset,
        channel=channel,
        member_pool="lse",
        neg_ratio=5,
        lexical=True,
        sent_aligned=sent_aligned,
    )
