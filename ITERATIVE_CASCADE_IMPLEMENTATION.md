# Iterative Cascade Implementation

## Overview

Implemented an **iterative disambiguation cascade** for Stage B that refines cluster predictions over multiple passes. This addresses the limitation where Stage B makes all merge decisions in a single pass without considering the full document-level context that emerges from earlier merges.

## Architecture

### Key Idea
Instead of one-shot merging, the model:
1. Starts with Stage A clusters (within-window)
2. Performs Stage B cross-window merging (pass 1)
3. Re-clusters the merged results into new per-window groups
4. Repeats Stage B merging on refined clusters (pass 2)
5. Continues for N passes (default: 3)

### Why This Works
- **Early passes** catch high-confidence merges (proper nouns, exact matches)
- **Later passes** benefit from richer cluster context (merged clusters have more evidence)
- **Iterative refinement** allows tentative decisions in early passes to inform later ones
- **No architectural changes** to the matcher itself—just multi-pass inference

## Implementation Details

### Modified Components

1. **MentionMatcher.__init__** (`stage2_context_encoder.py`)
   - Added `iterative: bool = False` parameter
   - Stored as `self.iterative` instance variable

2. **_predict_full_doc_clusters** (`train_stage2.py`)
   - Routes to `_predict_iterative_cascade()` when `cluster_matcher.iterative = True`
   - Standard single-pass logic remains unchanged for backward compatibility

3. **_predict_iterative_cascade** (`train_stage2.py`)
   - New function implementing the multi-pass refinement loop
   - For each pass:
     - Scores all cross-window cluster pairs via `_stage_b_merge_edges()`
     - Applies union-find to merge clusters
     - Reconstructs per-window cluster structure from merged results
     - Uses refined clusters as input for next pass
   - Early exits if no merges happen (convergence)

4. **train_stage_b** (`train_stage2.py`)
   - Added `iterative: bool = False` parameter
   - Checkpoint naming: appends `_iter` suffix when enabled
   - Passes `iterative` flag to MentionMatcher constructor

### Training Configuration

```python
train_stage_b(
    window=256,
    subset="all8k",
    head="mention",           # mention-level matcher (current best)
    dropout=0.2,
    iterative=True,           # enable cascade
    hard_neg=False,           # disabled (no proven benefit)
    use_structured=False,     # N/A for mention head
)
```

### Checkpoint
- Saves to: `stage2_cluster_matcher_k256_all8k_ment_iter.pt`
- Compatible with existing eval/diagnostic infrastructure

## Expected Improvements

From STAGE_B_FUTURE_IMPROVEMENTS.md conservative estimates:
- **Iterative Cascade**: +0.6 F1 improvement
- **Mechanism**: Better recall on multi-hop chains (A→B in pass 1, B→C in pass 2 ⇒ A→C merged)
- **Cost**: 3x inference time during eval (but training time unchanged—still one-pass loss)

## Design Decisions

### Why Mention-Level Base?
- Current best architecture (85.56 F1)
- Preserves per-mention evidence without pooling
- Explicit lexical channel for identity matching

### Why No Hard Negatives?
- Previous experiment showed -0.10 F1 degradation
- Lower edge recall (69.5% → 71.0%)
- Slower training with no benefit
- Start simple; can add later if cascade plateaus

### Why 3 Passes?
- Conservative choice balancing improvement vs. runtime
- Diminishing returns expected after pass 2-3
- Can tune based on validation performance

### Training vs. Inference
- **Training**: Still uses single-pass loss (not iterative)
  - Avoids gradient issues from multi-pass reconstruction
  - Learns to make good single-pass decisions
- **Inference**: Applies full iterative cascade
  - Test-time refinement without retraining

## Next Steps

1. **Train** the iterative cascade model:
   ```bash
   uv run python disambiguation/train_stage2.py
   ```

2. **Evaluate** on CoNLL test set (compare to 85.56 baseline)

3. **Analyze** diagnostics:
   - Edge recall by window gap (should improve gap≥3)
   - Per-doc runtime (3x slower expected)
   - Convergence behavior (how many passes needed?)

4. **If successful** (+0.4 F1 or better):
   - Document in RESEARCH.md
   - Consider combining with hybrid pooling next

5. **If unsuccessful** (no improvement):
   - Debug: check if clusters are actually being refined between passes
   - Fallback: investigate hybrid pooling instead (separate architecture)

## Open Questions

- **Optimal number of passes?** (may need < 3 or > 3)
- **Should training also be iterative?** (more complex gradients, worth trying if inference works)
- **Does it help long-range links?** (check recall_gap_ge3 metric)
- **Interaction with threshold tuning?** (cascade may prefer different null_bias)

## Code Locations

- Architecture: `disambiguation/stage2_context_encoder.py` (MentionMatcher)
- Training/Inference: `disambiguation/train_stage2.py` (_predict_iterative_cascade)
- Main script: `disambiguation/train_stage2.py` (__main__)
