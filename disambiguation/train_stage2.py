from disambiguation.data import _win_names
from disambiguation.paths import MODELS_DIR
from disambiguation.stage2_context_encoder import BACKBONE, BIO_BACKBONE
from disambiguation.stage_a import train_bio_tagger, train_stage_a
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
    # Mention detection runs FIRST: the BIO tagger (fine-tuned RoBERTa, L depth heads) predicts the
    # spans the rest of the pipeline resolves. It runs at det_window=510 content tokens (RoBERTa's max
    # context: max_position_embeddings=514 minus <s>/</s> minus the +2 position offset; 512 content
    # overflows the position table). Detection is O(T) so the 256 resolution-window budget doesn't
    # apply. Independent of Stage A/B's 256 windows; sentence-aligned packing keeps every mention whole.
    # train_detector=True (re)trains/resumes the detector each run: train_bio_tagger(resume=True) loads
    # an existing checkpoint and continues from it (or trains fresh if none). Flip to False once the
    # detector is converged so the pipeline skips straight to Stage A/B.
    n_layers, det_window, train_detector = 3, 510, True
    # Detector knobs (init and depth are orthogonal). Measured exact F1: head-only=52, NER top-6=77
    # (plateau), vanilla full-FT(24)=85.6 — boundary quality scales with trainable depth. Current run:
    # NER-12, warm-started from the NER-6 checkpoint (same arch/backbone) so it continues from 77 rather
    # than re-climbing, to see if more depth on the NER init beats the plateau. To instead resume the
    # 85.6 vanilla full-FT run: det_backbone="vanilla", det_layers=24, det_win_bs=2,
    # det_init=bio_tagger_k510_L3_sent.pt (set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for it).
    det_backbone = {"ner": BIO_BACKBONE, "vanilla": BACKBONE}["ner"]
    det_layers, det_win_bs = 12, 4  # top-12 encoder layers + heads
    det_init = MODELS_DIR / "bio_tagger_k510_L3_ner_ft6_sent.pt"  # warm-start from NER-6 (77); None to start fresh
    if train_detector:
        train_bio_tagger(
            window=det_window, sent_aligned=sent_aligned, n_layers=n_layers,
            model_name=det_backbone, n_trainable_layers=det_layers, win_bs=det_win_bs,
            init_ckpt=det_init if det_init.exists() else None, resume=True,
        )
    # stage_a_error_analysis(window=window, subset=subset, channel=channel, sent_aligned=sent_aligned)
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
        calibrate=False,
    )
