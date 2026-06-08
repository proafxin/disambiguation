# Stage B Improvements: Structured Cluster Features + Hard Negative Mining

## Summary

Implemented two key improvements to Stage B based on external feedback:

1. **Structured Cluster Representations** - Enriches cluster encoding with explicit structural features
2. **Hard Negative Mining** - Focuses training on semantically confusable negatives

## 1. Structured Cluster Features

### Problem
The original `ClusterEncoder` collapsed each cluster into a single attention-pooled vector, throwing away:
- Canonical mention (most identity-bearing)
- First mention (discourse salience)
- Cluster statistics (size, specificity variance)

### Solution
Modified `ClusterEncoder` to concatenate multiple cluster views:

```python
class ClusterEncoder(nn.Module):
    def __init__(self, proj_dim=1024, dropout=0.3, use_structured=True):
        # ... existing pooling ...
        
    def forward(self, ctx, bge):
        # 1. Original: attention-pooled representation
        pooled = (attn @ m)  # (2*proj_dim,)
        
        if not use_structured:
            return pooled
        
        # 2. Canonical mention (highest BGE norm = most specific)
        canonical_idx = torch.argmax(torch.norm(bge, dim=-1))
        canonical = m[canonical_idx]  # (2*proj_dim,)
        
        # 3. First mention (discourse position matters)
        first = m[0]  # (2*proj_dim,)
        
        # 4. Cluster statistics
        stats = torch.cat([
            bge_norms.mean().unsqueeze(0),  # avg specificity
            bge_norms.std().unsqueeze(0),   # specificity variance
            torch.tensor([len(bge)]),        # cluster size
        ])  # (3,)
        
        # Output: pooled + canonical + first + stats
        return torch.cat([pooled, canonical, first, stats], dim=-1)
```

**Output dimension:** `4*proj_dim + 3` (e.g., 4099 for proj_dim=1024)

**FFNN adjusted:** Input changes from `2*g = 4096` to `2*(4*proj_dim+3) = 8198` for cluster pairs

### Why This Helps
- **Canonical mention**: Surfaces the best identity-bearing mention (what humans use: "Obama" not "he")
- **First mention**: Entity introduction position matters for discourse tracking
- **Statistics**: Size/variance help the matcher weight clusters appropriately

**Complexity:** Near-zero - just indexing and concatenation, no new learned parameters beyond FFNN input adjustment

## 2. Hard Negative Mining

### Problem
Most negatives are trivial ("Obama" vs "Microsoft"). The model wastes capacity on easy examples and doesn't learn to distinguish hard cases ("Obama" vs "Bush" - same entity type, similar contexts).

### Solution
During training, compute semantic hardness for each negative pair:

```python
def _meta_hardness(meta, bge_all):
    # Max cosine similarity between left/right cluster mean BGE vectors
    # High similarity + non-coreferent = hard negative
    left_means = [bge_all[cluster].mean(0) for cluster in left_clusters]
    right_means = [bge_all[cluster].mean(0) for cluster in right_clusters]
    return max_cosine_similarity(left_means, right_means)
```

**During negative subsampling:**
- If `hard_neg=True`: sort negatives by hardness, keep top-k
- If `hard_neg=False`: random sample negatives

**During loss computation:**
- Hard negatives (hardness > 0.5): weight = `hard_neg_weight` (default 3.0x)
- Easy negatives: weight = 1.0x
- Positives: weight = 1.0x

```python
# Per-pair weighted loss
weighted_loss = sum(loss[i] * weight[i] for i in pairs) / len(pairs)
```

### Why This Helps
- Model spends more gradient budget on actual mistakes
- Semantically similar but different entities (e.g., "Obama" vs "Bush") get more training signal
- Expected +0.5 to +1.5 F1 based on external analysis

## Usage

### Train with structured features only:
```python
from disambiguation.train_stage2 import train_stage_b

train_stage_b(
    window=256,
    subset="all8k",
    head="cluster",
    dropout=0.2,
    use_structured=True,   # Enable structured features
    hard_neg=False,
    neg_ratio=1.0,
)
```

### Train with structured features + hard negatives:
```python
train_stage_b(
    window=256,
    subset="all8k",
    head="cluster",
    dropout=0.2,
    use_structured=True,   # Enable structured features
    hard_neg=True,         # Enable hard negative mining
    neg_ratio=1.0,         # 1:1 neg sampling, but select hardest
    hard_neg_weight=3.0,   # 3x weight for hard negatives
)
```

### Quick start script:
```bash
uv run python train_stage_b_structured.py
```

## Checkpoints

Checkpoints are saved with naming convention:
- Baseline: `stage2_cluster_matcher_k256_all8k.pt`
- Structured: `stage2_cluster_matcher_k256_all8k_struct.pt`
- Structured + HN: `stage2_cluster_matcher_k256_all8k_struct_hn.pt`

## Expected Results

Based on external feedback analysis:

**High probability improvements:**
- Structured cluster features: +0.3 to +0.8 F1
- Hard negative mining: +0.5 to +1.5 F1
- Combined: +0.8 to +2.0 F1

**Current baseline:** 85.33 F1 (pooled ClusterMatcher)
**Target:** 86.0-87.0 F1

## Next Steps

After validating these improvements:

1. **Contrastive prototype learning** (medium-priority)
   - Add explicit entity identity space
   - Address AUC gap (0.57 cosine → 0.91 linear probe)
   
2. **GNN over clusters** (if needed)
   - Only if simpler approaches saturate
   - O(C²) = ~21² cluster nodes, still cheap

3. **Consistency loss for transitive closure**
   - Encourage scorer to anticipate union-find behavior

## Implementation Notes

- **Backward compatibility:** `use_structured=False` recovers original pooled behavior
- **No retraining needed:** Structured features add zero learned parameters to encoder
- **Batched implementation:** Both `forward()` and `forward_batched()` support structured mode
- **Hard neg caching:** Hardness scores could be precomputed and cached (not done for simplicity)

## Code Changes

Files modified:
1. `disambiguation/stage2_context_encoder.py`
   - `ClusterEncoder.__init__()`: add `use_structured` flag
   - `ClusterEncoder.forward()`: compute canonical/first/stats
   - `ClusterEncoder.forward_batched()`: batched structured feature extraction
   - `ClusterMatcher.__init__()`: adjust FFNN input dim based on `use_structured`

2. `disambiguation/train_stage2.py`
   - `train_stage_b()`: add `use_structured`, `hard_neg_weight` parameters
   - Training loop: compute hardness, weight losses by hardness
   - Checkpoint naming: add `_struct` and `_hn` suffixes

3. `train_stage_b_structured.py` (new)
   - Convenience script to run both experiments

## References

External feedback source: Agent analysis of RESEARCH.md emphasizing:
- "Stage A is saturated, Stage B has the headroom"
- "Learn a cluster state, not a cluster embedding"
- "Hard negatives: model spends capacity on actual mistakes"
- "Canonical mention + first mention + stats = structured entity representation"
