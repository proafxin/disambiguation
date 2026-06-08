# Stage B Future Improvements

## Current Performance Boundaries

**Current Best:** 85.56 F1 (mention-level matching)
- Stage A alone: 74.39 F1
- Stage B contribution: +11.17 points
- Remaining gap to hypothetical ceiling (~87 F1): ~1.5 points

**Key Bottlenecks:**
1. **PRON-only clusters:** 64.1 F1 — pronouns have no BGE semantic signal, cross-window matching relies entirely on decayed RoBERTa context
2. **Edge recall:** 71.0% — Stage B misses ~29% of cross-window gold pairs, concentrated in pronoun-heavy and long-gap cases
3. **Independent pair scoring:** Each cluster pair is scored in isolation without leveraging surrounding quotient-graph structure

---

## Proposed Architectural Extensions

### 1. Dynamic Message Passing on the Quotient Graph (GNN Layers)

**Motivation:** Current Stage B treats each cluster-pair match independently. A cluster containing only pronouns has no BGE signal and weak cross-window RoBERTa context. But if that cluster co-occurs in a window with a high-confidence proper-noun cluster, the structural context can inform the pronoun cluster's representation.

**Design:**
- After Stage A produces per-window clusters, construct a quotient graph where:
  - **Nodes:** Stage A clusters (mean ~20.7 per document)
  - **Edges:** Within-window co-occurrence (clusters from the same window are connected)
- Add 1–2 layers of Graph Attention Network (GAT) or GraphSAGE:
  ```
  For each cluster node c:
    neighbors = {clusters in same window as c}
    c_updated = c + Attention(c, {n for n in neighbors})
  ```
- Updated cluster representations are then passed to ClusterMatcher/MentionMatcher for cross-window scoring

**Expected Gains:**
- **PRON-only improvement:** +1.0–2.0 F1 by borrowing signal from co-occurring named entities
- **Edge recall:** +2–4% by enriching pronoun cluster representations with window-local structural context
- **Complexity:** O(C²) message-passing over ~21 clusters/doc is negligible compared to O(W²) cross-window matching

**Implementation Notes:**
- Keep GNN layers shallow (1–2 hops) to avoid over-smoothing
- Initialize node features as current ClusterEncoder output (pooled or structured)
- Train GNN layers jointly with ClusterMatcher, keeping Stage A frozen
- Consider residual connections: `c_final = c_original + GNN(c)`

---

### 2. Averaged/Max Hybrid Pooling in Mention Matcher

**Motivation:** Log-mean-exp aggregation (logsumexp - log N) removes cluster-size bias but can smooth out high-confidence singular links. If one mention pair has overwhelming evidence (e.g., exact surface match "Barack Obama" ↔ "Barack Obama") but the rest of the cluster pairs are weak, log-mean-exp dilutes this signal.

**Design:**
- Current MentionMatcher:
  ```
  score(C_L, C_R) = log-mean-exp over all (m_i, m_j) pairs
  ```
- Hybrid aggregation:
  ```
  score_avg = log-mean-exp over all pairs  (current)
  score_max = max_{i,j} score(m_i, m_j)    (strongest pair)
  final = MLP([score_avg, score_max])      (learned combination)
  ```

**Expected Gains:**
- **PROPN-only improvement:** +0.3–0.5 F1 by preserving exact-match signals
- **Edge precision:** +1–2% by allowing strong lexical matches to dominate noisy distributional evidence
- **Handles mixed clusters better:** all-mixed bucket (70.1 F1) benefits from explicit "best pair" channel

**Implementation Notes:**
- Concatenate [log-mean-exp, max-score] before final linear layer
- Max operation is differentiable via straight-through estimator or max-pooling gradient
- Alternative: use attention weights from MentionMatcher to identify top-K pairs, pool only those

---

### 3. Iterative Disambiguation Cascade

**Motivation:** Not all cluster pairs are equally hard. Proper-noun clusters with exact lexical overlap have near-perfect precision but current greedy decode treats them the same as ambiguous pronoun pairs. Resolving high-confidence matches first and updating representations before scoring hard cases mimics human incremental reasoning.

**Design:**

**Phase 1: High-Precision Lexical Matching**
- For each cross-window cluster pair (C_L, C_R):
  - Compute lexical features: exact surface match, IDF-weighted Jaccard, containment
  - If any feature > threshold (e.g., exact match = 1.0, Jaccard > 0.8):
    - Merge immediately (no neural scoring)
    - Mark as "resolved"
- Union-find over resolved pairs → updated quotient graph

**Phase 2: Representation Pooling**
- For each merged cluster from Phase 1:
  - Pool member mentions into single node
  - Recompute BGE/ctx representations (mean or attention-pooled)
- Shrinks quotient graph from ~21 to ~15 nodes (estimate: 30% of pairs are high-precision lexical)

**Phase 3: Neural Matching on Residual**
- Run MentionMatcher only on remaining unresolved cluster pairs
- These are pronoun-heavy, long-gap, or ambiguous noun pairs where neural signal is necessary
- Benefits from reduced search space and updated representations

**Expected Gains:**
- **Edge precision:** +2–3% by removing trivial errors on high-confidence pairs
- **Efficiency:** 30% fewer neural evaluations if Phase 1 resolves ~30% of pairs
- **PRON+PROPN improvement:** +0.5–1.0 F1 by correctly anchoring proper nouns before scoring pronouns
- **Interpretability:** Explicit separation of lexical vs. contextual reasoning

**Implementation Notes:**
- Thresholds can be learned via validation sweep or treated as hyperparameters
- Phase 1 is deterministic and costs O(C²) lexical feature computation (cheap)
- Phase 2 representation update is a single forward pass through ClusterEncoder
- Phase 3 reuses existing MentionMatcher with no architecture changes

---

## Orthogonal Improvements (Non-Architectural)

### Data Augmentation for Pronouns
- Current bottleneck: PRON-only 64.1 F1
- Approach: Synthetically replace proper nouns with pronouns in training data to increase pronoun density and force model to rely on contextual signal
- Expected: +0.5–1.0 F1 on PRON-only bucket

### Multi-Task Learning with Entity Typing
- Auxiliary task: predict coarse entity type (PERSON, ORG, GPE, etc.) from cluster representation
- Forces ClusterEncoder to extract type-discriminative features, helps pronoun resolution (he/she → PERSON, it → ORG/GPE)
- Expected: +0.3–0.5 F1 overall, concentrated in PRON+PROPN

### Learned Threshold per Cluster Type
- Current: single global null_bias threshold for all cluster pairs
- Proposal: learn per-type thresholds (PROPN-PROPN gets lower threshold = more merges, PRON-PRON gets higher = more conservative)
- Expected: +0.2–0.4 F1 by optimizing precision/recall trade-off per bucket

---

## Prioritization

**Immediate (Next Experiment):**
1. **Hybrid pooling** (§2) — minimal code change, targets max-score signal preservation
2. **Structured cluster features** (already in progress) — adds canonical/first/stats to pooled representation

**Medium-Term (Next 2–3 Experiments):**
3. **Iterative cascade** (§3) — highest expected ROI, combines lexical + neural strengths
4. **GNN message passing** (§1) — requires new architecture but targets core pronoun bottleneck

**Long-Term (Research Extensions):**
5. **Multi-task entity typing** — requires auxiliary labels, may need external NER model
6. **Data augmentation** — requires train-time synthesis pipeline

---

## Expected Final Performance

Combining all three structural extensions (§1–3) conservatively:
- **GNN:** +1.5 F1 (pronoun improvement)
- **Hybrid pooling:** +0.4 F1 (max-signal preservation)
- **Iterative cascade:** +0.6 F1 (precision on easy cases, better pronoun anchoring)
- **Total expected:** 85.56 + 2.5 = **88.0 F1** (±0.3)

This would close ~2/3 of the remaining gap to the frozen-encoder ceiling (~87–88 F1) without fine-tuning RoBERTa or increasing trained parameter count beyond 16.8M.

---

## Open Questions

1. **GNN depth:** 1 layer (direct neighbors) vs. 2 layers (2-hop)? Deeper may over-smooth pronoun signals.
2. **Hybrid pooling variants:** Max vs. top-K average vs. attention-weighted max?
3. **Cascade threshold tuning:** Fixed vs. learned vs. per-type? Needs ablation.
4. **Joint training:** Train all three extensions together or sequentially? Risk of instability.

---

## References to Current Codebase

- `ClusterEncoder` (stage2_context_encoder.py:161): Attention pooling implementation, target for hybrid pooling
- `MentionMatcher` (stage2_context_encoder.py:267): Log-mean-exp aggregation, target for max-score channel
- `_stage_b_merge_edges` (train_stage2.py:1453): Cross-window scoring loop, target for cascade logic
- `_lex_matrix` (train_stage2.py:1823): Lexical feature computation, reuse for Phase 1 of cascade
- `eval_stage_b` (train_stage2.py:1520): Evaluation harness, no changes needed

---

## Notes

- All three extensions preserve O(NK + N²/K²) complexity: GNN is O(C²), hybrid pooling is O(1) per pair, cascade is O(C²) lexical + reduced neural
- Trained parameter count remains ~16.8M (GNN adds ~0.5M, hybrid pooling adds ~0.1M)
- No changes to Stage A or frozen encoders
- Backwards compatible: can fall back to current MentionMatcher if extensions regress
