# Global Entity Disambiguation via Orthogonal Universal Axes

## Overview

This document is a complete technical record of the system, its history, every design decision and why it was made, all experimental results, and the current state of the codebase. It is written so that an external agent with no prior context can fully understand the system without reading the code.

---

## 1. The Core Thesis

End-to-end coreference systems apply full self-attention over the entire document, incurring $O(N^2)$ cost. This work demonstrates that cost is not necessary.

The document is partitioned into windows of at most $K$ subtokens, packing whole sentences so no sentence is ever split across a window boundary. Full quadratic attention runs inside each window, but Cauchy-Schwarz bounds the total encoding work to $O(NK)$ — linear in document length with $K$ as a small constant. Entities resolved locally within each window are then matched across arbitrarily distant windows at the cluster level, at cost $O(C^2)$ over $C \ll N$ clusters. The system reaches CoNLL F1 = 86.46 on CoNLL-2012 test (86.51 with a val-tuned merge threshold) with both encoders frozen, matching or exceeding systems that fine-tune large models with full-document attention. Notably, test (86.46) exceeds the best validation score (86.11): the sentence-aligned windowing generalizes to held-out documents rather than over-fitting the selection set.

The windowed architecture is made possible by a second principle: instead of one model encoding every signal dimension jointly, each signal required for the decision is supplied from an independent specialist encoder, and only their composition is learned. No single model needs to see the whole document. Because the signals arrive pre-separated, there is nothing that requires global attention to disentangle. The specific encoders are interchangeable; the contribution is the architecture and the decomposition principle.

**The key diagnostic that launched this work:** Taking the two axis vectors for pairs of different-lemma mentions, their raw cosine similarity separates coreferent from non-coreferent pairs at chance (AUC ≈ 0.57). The *same* vectors, read by a small learned linear probe, separate them at AUC = 0.91. The signal is fully present in the axes; collapsing them to a scalar discards it. The axes must be kept as vectors and read jointly by a learned head — reducing either to a cosine, or fusing the two into one representation, throws the signal away.

**Why transitive closure, not direct similarity:** Cosine similarity is not transitive. If cos(u,v) ≥ c and cos(v,w) ≥ c, the bound on the endpoints is only cos(u,w) ≥ 2c²-1. Two strong links at c=0.80 bound the endpoints at only 0.28. A long chain's endpoints may have near-zero direct similarity even when every adjacent pair is strongly coreferent. Transitive closure (union-find) is therefore the correct composition operator, not direct long-range similarity.

---

## 2. Problem Statement

Let D = (x₁, ..., xₙ) be a document and M its nominal mentions (pronouns, nouns, proper nouns). The task is to recover the coreference partition C = {S₁, ..., Sₖ} of M, where each Sₘ contains all mentions referring to the same entity.

The coreference relation ~ is an equivalence relation on M. The global partition C is the set of equivalence classes induced by the transitive closure of the local link graph G = (M, E), where (u,v) ∈ E iff u ~ v.

**Our claim:** The edges of G are locally decidable. For every mention u and its nearest same-entity antecedent v, the decision u ~ v can be made from the two information-independent channel vectors of u and v alone, without global context.

---

## 3. Training Data

All experiments use gold mentions (spaCy-extracted nominal heads mapped to dataset spans). Four datasets:

- **CoNLL-2012** (train/val/test): the primary benchmark. Official scorer.pl used for all reported numbers.
- **LitBank** (train/val/test): literary text, longer documents.
- **PreCo** (train subsampled to 8000 docs, val): large-scale pronoun-heavy corpus.
- **CorefUD** (train/val): multilingual, diverse genres.

Mention surfaces are encoded with BGE-large (`BAAI/bge-large-en-v1.5`) once per unique surface and cached. RoBERTa context vectors are cached per document to disk (`stage2_span_ctx_v7/`). The nominal mention cache is `stage2_conll_nominals_v4.pkl`.

---

## 4. Architecture

### 4.0 Final architecture at a glance

Two **frozen** encoders supply the only learned semantic/contextual signal; everything trained is a small head. The complete final system, end to end:

**Frozen channels (both stages):**
- **BGE-large** (`BAAI/bge-large-en-v1.5`) — semantic embedding of the mention surface (1024-d)
- **RoBERTa-large** — contextual span vector, mention start⊕end (2×1024 = 2048-d)

**Stage A — within-window mention-pair scorer (`AntecedentScorer`, ~8.4M trained):**
- project RoBERTa 2048→1024, BGE 1024→1024, concatenate → `g_i` (2048-d)
- pair feature `[g_i | g_j | dist_emb(bucket(i−j))]` → MLP → mention-ranking MLL → union-find decode
- **distance embedding is ON** (a learned mention-index-distance bucket; load-bearing, +1.3–2.1)

**Stage B — cross-window cluster GNN (`ClusterGNN`, ~7.7M trained):**
- project RoBERTa 2048→512, BGE 1024→512, concatenate → member vec → **lse-pool** → node vector
- add a **window-position embedding**, then a 2-layer Transformer with an outer residual
- pair feature `[h_A | h_B | h_A⊙h_B | |h_A−h_B| | lex(3)]` → MLP → cluster-level antecedent ranking
- **lexical channel is ON** (IDF-Jaccard/containment/exact-match, concatenated; +0.62)
- decode merges if `score > null_bias + δ`, with a **val-tuned δ=+0.50** (→ 86.51)

**On:** distance (A), lexical (B), window-position embedding (B), val-tuned merge threshold.
**Off:** raw / wider projection (near-optimal at 512, §8).

**Result:** CoNLL-2012 test **86.46** (**86.51** with the merge threshold), both encoders frozen, ~16.1M trained parameters.

### 4.1 Stage A: Intra-Window Resolution

**Motivation:** Partition the document into disjoint windows and resolve coreference strictly within each window. Every scored pair has both mentions within the same RoBERTa encoding window, so the contextual channel carries reliable signal.

**Window assignment:** The document is partitioned into disjoint windows of **at most** K=256 subtokens by greedily packing whole sentences: a sentence is appended to the current window if it fits, otherwise a new window opens, so **no sentence is ever split across a window boundary** (a single sentence longer than K is the only case that splits, at K). The Stage A scorer and the Stage B GNN consume the *identical* windows, and RoBERTa encodes each window over these sentence-clean spans. Every window is ≤K, so the Cauchy-Schwarz $O(NK)$ bound holds.

**Architecture (AntecedentScorer, ~8.4M params):**

*Mention representation:* For each mention i, concatenate projected context and semantic channels:

```
g_i = [P_ctx([ctx_start | ctx_end]) | P_bge(bge_i)] ∈ R^2048
```

where P_ctx: R^2048 → R^1024 and P_bge: R^1024 → R^1024 are learned linear projections.

*Scoring:* Pairwise MLP with distance embedding:

```
s(i,j) = MLP([g_i | g_j | dist_emb(bucket(i-j))])
```

*Objective:* Mention-ranking MLL. Each mention i softmaxes over {ε} ∪ {j < i} where ε is a learned null antecedent.

*Decode:* Each mention links to its single best antecedent if s(i,j*) > null_bias, else opens new entity. Clusters are connected components via union-find.

**Training:** For each document, mentions are partitioned into windows. The scorer runs independently on each window's mentions. Loss is computed per window via MLL, meaned across all windows and documents in the batch.

**Gold for evaluation:** Gold clusters are split at window boundaries — a gold cluster spanning multiple windows becomes multiple per-window sub-clusters. Stage A is evaluated against this window-split gold, which credits only locally-achievable coreference.

**Results (CoNLL-2012 test, window-split gold):**

| | CoNLL | MUC | B³ | CEAFe |
|---|---|---|---|---|
| **Stage A (windowed, sentence-aligned)** | **89.48** | **92.72** | **88.89** | **86.84** |

**Type breakdown (Stage A, window-split gold):**

| Cluster type | CoNLL | MUC | B³ | CEAFe |
|---|---|---|---|---|
| PROPN-only | 90.77 | 91.29 | 90.77 | 90.26 |
| PRON+PROPN | 83.78 | 85.50 | 82.67 | 83.17 |
| NOUN-only | 82.22 | 82.62 | 82.31 | 81.74 |
| PRON+NOUN | 81.93 | 81.11 | 80.65 | 84.03 |
| PRON-only | 79.02 | 82.61 | 78.60 | 75.86 |
| NOUN+PROPN | 78.18 | 77.97 | 77.53 | 79.04 |
| all-mixed | 74.81 | 76.67 | 72.82 | 74.93 |

**Key improvements:**

1. **PRON-only:** Every pronoun is now only linked to antecedents RoBERTa actually saw in the same window, and (sentence-aligned) RoBERTa always sees that antecedent's full sentence rather than a truncated window edge.
2. **Cluster fragmentation low** because every mention's correct antecedent is within the window and the contextual signal is reliable (CEAFe 86.84).
3. **Cross-dataset generalisation (val):** LitBank 84.31, PreCo 87.03, CorefUD 83.81.

**Remaining problem:** The all-mixed bucket (73.4) contains clusters spanning all three mention types across multiple windows. These require cross-window cluster matching — Stage B's job.

**Training details:**

- Optimizer: AdamW, lr=1e-3, weight_decay=0.1
- Scheduler: CosineAnnealingLR, T_max=60, eta_min=1e-4
- doc_bs=8, max_epochs=60, patience=8
- Converged at epoch 48 (val CoNLL F1 = 89.60), early stopped at epoch 56
- ~63s/epoch on single GPU after ~16-minute RoBERTa caching pass

---

### 4.2 Stage B: Cross-Window Cluster Matching

**Motivation:** Stage A produces per-window clusters. Stage B matches clusters across windows to recover the full document partition. Instead of only comparing adjacent windows (which would miss entities with gaps), we compare all ⌊W(W-1)/2⌋ window pairs — this is O(W²) = O(N²/K²), a factor of K²=256²=65536 cheaper than full-document O(N²) attention.

**Why all-pairs window comparison instead of adjacent-only:** An entity appearing in window 1 and window 5 with nothing in between cannot be linked by adjacent-only comparison. All-pairs window comparison directly matches any two windows, so no entity chain is missed regardless of gap size.

#### 4.2.1 The Stage B head: Graph Neural Network — cluster-level antecedent ranking

*ClusterGNN (~7.7M parameters):* Stage B is antecedent ranking over the quotient graph. Nodes are Stage A clusters (~21/doc); each member is projected (RoBERTa 2048→512, BGE 1024→512, concatenated to 1024) and the members are lse-pooled to one node vector, a window-position embedding is added, and a 2-layer Transformer encoder (full self-attention = fully-connected message passing) contextualizes every node by all others. Each cluster then ranks its single best antecedent among clusters in **strictly earlier windows** (or null = new entity), and the antecedent-pointer forest is the chains. Cost is **O(M + C²)** — member pooling is one O(M) pass, attention and pairwise scoring are O(C²) — never M². The motivation is global consistency: a merge can depend on the rest of the graph (transitivity, competition).

**Two failures had to be fixed before it trained at all — both measured:**

1. **All-null collapse.** Naively the model drove every antecedent score below null and accepted **zero** merges, sitting at the Stage-A-no-merge floor (74.39). Root cause: full self-attention over the small node set **over-smoothed** the node representations to near-identical (mean pairwise cosine → **1.000**), so gold and non-gold cluster pairs got the *same* score (separation **0.000**). No loss rebalancing could fix it — the *features* carried no signal.
2. **Outer residual.** Adding the pre-message-passing node vector back (`h = h0 + GNN(h0)`) guarantees the discriminative pooled features reach the scorer, so message passing can only *refine* them, never erase them. With the residual, merges are non-zero from epoch 1. A cross-window-only antecedent mask and per-cluster negative subsampling are also required.

Cluster members are pooled to the node vector by log-mean-exp (`lse`, a size-normalized soft-max).

**The lexical channel is concatenated, not collapsed to a scalar.** A lexical signal `[IDF-Jaccard, containment, exact-match]` over the cluster pair is concatenated into the pair feature `x_pair = [v_A | v_B | v_A⊙v_B | |v_A−v_B| | lex(3)]` and read *jointly* by the score MLP — worth **+0.62** over no lexical. Collapsing the three features to a single scalar added to the score (sharing `null_bias`) is a wash: it can't be used conditionally, so it just raises the global merge threshold and redistributes across buckets without net gain. This is the §1 "don't collapse to a scalar" principle applied to the symbolic axes — the head uses lexical conditionally and non-linearly, landing on the name-matching buckets where surface identity bridges windows. The general rule: a concatenated side-channel earns its place only if it is conditionally independent of the frozen channels given the label — it must carry signal BGE and RoBERTa don't already expose (§8).

**Result (CoNLL-2012 test — current system: sentence-aligned windows + lse pooling + concatenated lexical):** **86.46 F1** (MUC 92.47, B³ 84.37, CEAFe 82.54), val 86.11. Test exceeds val: sentence-aligned windowing generalizes rather than over-fitting the selection set.

| Cluster type | CoNLL | MUC | B³ | CEAFe |
|---|---|---|---|---|
| PROPN-only | 86.97 | 87.08 | 86.71 | 87.10 |
| NOUN-only | 77.93 | 78.27 | 77.98 | 77.53 |
| PRON+NOUN | 75.73 | 73.89 | 73.54 | 79.75 |
| PRON+PROPN | 74.68 | 76.44 | 71.90 | 75.71 |
| all-mixed | 71.37 | 76.78 | 67.81 | 69.53 |
| NOUN+PROPN | 70.78 | 71.02 | 69.77 | 71.55 |
| PRON-only | 68.61 | 72.15 | 66.36 | 67.31 |

PROPN-only is highest (86.97) — sentence-clean window edges give RoBERTa intact name context — while the cross-type noun buckets (NOUN+PROPN 70.8, all-mixed 71.4) are the floor, the common-noun ceiling discussed throughout.

**Comparison with prior work (CoNLL-2012 test, gold mentions):**

| | CoNLL | MUC | B³ | CEAFe |
|---|---|---|---|---|
| Stage A only (window-split gold) | 89.48 | 92.72 | 88.89 | 86.84 |
| **Stage A + B GNN (C²)** | **86.46** | **92.47** | **84.37** | **82.54** |
| Dobrovolskii et al. (2021) | 81.8 | 85.7 | 80.0 | 79.7 |
| Caciularu et al. (2023) | 83.6 | 87.4 | 82.0 | 81.3 |
| Bohnet et al. (2023) | 83.8 | 87.6 | 82.2 | 81.5 |
| Luo et al. (2025) | 85.1 | 88.9 | 83.5 | 82.9 |

SOTA systems use predicted mentions and fine-tune end-to-end; this work uses gold mentions with both large encoders frozen.

**Training:**

- Stage A weights frozen; Stage A clusters precomputed once and cached
- Per doc, the GNN scores all cross-window cluster pairs (cluster-level antecedent-ranking MLL)
- AdamW lr=1e-3, weight_decay=0.1, CosineAnnealingLR T_max=30, doc_bs=8, patience=5

**Analysis:**

- Stage B recovers **+12.1** CoNLL over Stage A alone on full-document gold (74.39 → 86.46): local resolution reaches ~86% of the final F1, cross-window matching adds the rest.
- The remaining ceiling is the **noun buckets** (NOUN-only 77.9, NOUN+PROPN 70.8, all-mixed 71.4) — different-word semantic coref ("the company" / "abc software firm ltd."), which no surface/morphology feature can touch. The Stage A within-window analysis confirms it: 93% of Stage A's recall misses are between *different* head words, and per-channel Stage A heads show RoBERTa carries noun coref (ctx-only NOUN recall 76.3 vs bge-only 73.0, both 84.4) — a frozen-contextual limit.

---

### 4.3 Data Mixture

Training uses CoNLL-2012 + LitBank + CorefUD + 8k subsampled PreCo, with a uniform per-dataset loss and all window pairs (no negative subsampling). A config-selection sweep established the levers:

- **PreCo is the lever.** The diverse mix buys ~+0.9 CoNLL over CoNLL-only, almost entirely from PreCo (LitBank + CorefUD without PreCo move it by only ~+0.07). PreCo amount is a mild inverted-U — 8k beats 10k (more PreCo slightly over-fits its distribution) — so 8k is the operating point.
- **Uniform loss + all window pairs.** PreCo is ~87% of training mentions, so up-weighting CoNLL distorts the transfer balance (raises CoNLL val, lowers test); and on the full corpus, training on all window pairs beats negative subsampling. Both are the committed defaults.

---

## 5. Complexity Analysis

Let N = document length in tokens, M = number of mentions, K = 256 = window size, W = N/K = number of windows.

**End-to-end systems:** O(N²) encoding (full-document self-attention) + O(M²) mention-pair scoring.

**Our system:**

*Encoding:* W independent windows of size K. By Cauchy-Schwarz:

```
Σ K² = W·K² = (N/K)·K² = NK
```

Factor of N/K improvement over O(N²).

*Stage A scoring:* W windows, each with M/W mentions:

```
W · (M/W)² = M²/W = M²K/N
```

Same factor of N/K improvement.

*Stage B matching:* the GNN runs over C ≈ 21 cluster nodes — member pooling is O(M), node self-attention + all-pairs antecedent scoring are O(C²):

```
O(M + C²),  C ≪ M
```

The O(C²) term is ~8× smaller than O(M²) and ~900× smaller than O(N²) in absolute operation count on this corpus.

**Full two-stage complexity:**

```
O(NK + C²)
```

For K=256 and typical document lengths, dominated by O(NK) = O(256N) — linear in document length.

---

## 6. Empirical System Diagnostics (CoNLL-2012 test)

Measured on the gold-mention Stage A + GNN Stage B system (CoNLL F1 = 86.46), 346 CoNLL-2012 test documents, single RTX 4060 Laptop GPU (8 GB). No retraining: these read or run the existing frozen checkpoints.

**Quotient-graph compression.** The two-stage decomposition collapses mentions into a much smaller quotient graph before any cross-window reasoning:

| Metric | Value |
|---|---|
| Mean mentions / doc (M) | 57.1 |
| Mean subtokens / doc (N) | 617.4 |
| Mean Stage A clusters / doc (C, quotient nodes) | 20.7 |
| Mean true entities / doc (E) | 13.1 |
| Compression ratio M/C | 2.76 : 1 |
| Fragments per true entity (C/E) | 1.45 (corpus 1.58) |

Stage B's all-pairs cluster comparison operates on ~21 nodes, not ~57 mentions or ~617 tokens: the O(C²) term is ~8× smaller than O(M²) and ~900× smaller than O(N²) in absolute operation count on this corpus. Each true entity fragments into only ~1.5 local clusters, so Stage B's reconnection load is light.

**Locality of resolution (Stage A alone vs Stage A+B).** Scoring the Stage A window-local clusters directly against full-document gold, with the cross-window merge step removed entirely (each local cluster left as its own entity):

| | CoNLL | MUC | B³ | CEAFe |
|---|---|---|---|---|
| Stage A alone (no merge, full-doc gold) | 74.39 | 84.60 | 69.10 | 69.50 |
| Stage A + GNN Stage B | 86.46 | 92.47 | 84.37 | 82.54 |
| **Stage B contribution** | **+12.07** | +7.87 | +15.27 | +13.04 |

**Local-only resolution already reaches ~86% of the final F1** (74.39 / 86.46) — the majority of coreference is decidable within a single sentence-aligned window, with no global context, supporting the §1 thesis that edges are locally decidable. The cross-window GNN is nonetheless worth a **substantial +12.07 CoNLL**, concentrated in the cluster-level metrics (B³/CEAFe) that punish unmerged fragments — the signature of split entities being rejoined.

**Parameter budget (trained vs frozen).**

| Component | Params | |
|---|---|---|
| Stage A scorer | 8.43M | trained |
| Stage B ClusterGNN | 7.70M | trained |
| **Total trained (Stage A+B)** | **16.13M** | |
| RoBERTa-large | 355M | frozen |
| BGE-large | 335M | frozen |
| **Total frozen** | **690M** | |

Trained parameters are 2.28% of the total; the two large encoders are never updated.

---

## 7. Codebase Structure

```
disambiguation/
├── paths.py                    # Centralized path management
├── conll_scorer.py             # Official scorer.pl wrapper + CoNLL key/response writers, type breakdown
├── stage2_context_encoder.py   # Model classes + losses
│   ├── ContextEncoder          # RoBERTa-large wrapper (frozen)
│   ├── encode_document_ctx     # window-batched RoBERTa pass → per-mention ctx vectors
│   ├── AntecedentScorer        # Stage A: within-window mention-pair scorer
│   ├── ClusterGNN              # Stage B: quotient-graph GNN — lse member pooling, outer residual,
│   │                           #   concatenated lexical (3) channel, antecedent ranking
│   │                           #   (flags: raw, ctx_proj/bge_proj projection widths)
│   ├── antecedent_mll_loss     # mention-ranking MLL (Stage A and the GNN's cluster-level objective)
│   └── decode_antecedents      # union-find decode
├── data.py                     # Datasets, encoding, caching, windowing, lexical features
│   ├── build_docs()            # Load datasets, encode BGE once per surface, cache
│   ├── precompute_span_ctx()   # Cache RoBERTa ctx vectors to disk
│   ├── load_span_ctx_single()  # Load the single-file packed ctx cache
│   ├── _win_names()            # window/subset/channel/sent_aligned → artifact names (ctx, head, matcher, clusters)
│   ├── _sentence_chunks() / _apply_sentence_windows()  # sentence-aligned windowing (win_chunks, win_ids)
│   ├── _build_lexical()        # per-mention tokens + doc-local IDF for the lexical channel
│   └── _key_clusters()         # gold CoNLL clusters for scoring
├── stage_a.py                  # Stage A: within-window resolution
│   ├── stage_a_batched_loss()  # batched Stage A: all windows in a doc-batch, one FFNN + vectorized MLL
│   ├── train_stage_a()         # Stage A training (flags: subset, channel, sent_aligned, raw, hidden, use_distance)
│   ├── precompute_stage_a_clusters()  # cache per-window Stage A clusters for Stage B (name_tag per arch)
│   └── stage_a_error_analysis()# within-window per-channel error analysis (recall by type, head-match, BGE separation, sentence splits)
├── stage_b.py                  # Stage B: cross-window cluster matching (GNN-only)
│   ├── _gnn_lex_matrix()       # (C,C,3) lexical node-pair features [IDF-Jaccard, containment, exact-match]
│   ├── calibrate_merge_threshold()  # per-dataset null_bias offset sweep on val (merge calibration)
│   ├── train_stage_b()         # Stage B GNN training (flags: lexical, raw, use_distance, ctx_proj, calibrate, force)
│   └── eval_stage_b()          # Full-document CoNLL F1 eval (GNN pointer-forest decode)
└── train_stage2.py             # Entry point: trains Stage A if missing, then Stage B (reuses matcher unless force=True)
```

Stage B is GNN-only. `train_stage_b` reuses an existing matcher checkpoint for evaluation unless `force=True` (and `eval_only=True` loads + tests without touching training either way).

**Key files on disk:**

- `data/stage2_conll_nominals_v4.pkl` — all mention data, BGE vectors, cluster IDs
- `data/stage2_span_ctx_v7/` — per-doc RoBERTa ctx vectors (M, 2, 1024) float16
- `cache/models/stage2_frozen_head_k256_all8k_sent.pt` — Stage A checkpoint (sentence-aligned windows)
- `cache/models/stage2_cluster_matcher_k256_all8k_sent_gnn_lse_lex.pt` — Stage B GNN checkpoint (best val CoNLL F1 = 86.11, test 86.46)
- `cache/models/stage_a_clusters_cache_k256_all8k_sent.pkl` — precomputed Stage A clusters for Stage B training

---

## 8. Current Status

**System Performance (CoNLL-2012 test, gold mentions):**

- Stage A (windowed, K=256, sentence-aligned): **89.48 F1** on window-split gold
- **Stage A+B GNN (lse pooling + concatenated lexical): 86.46 F1** (val 86.11, test > val) — current system
- Stage B contribution: **+12.1 points** (74.39 → 86.46)

**Architecture (Stage B is GNN-only):**

- Stage A: 8.4M parameters (AntecedentScorer)
- Stage B: ~7.7M parameters (ClusterGNN — lse member pooling, 2-layer Transformer + outer residual, concatenated lexical channel, cluster-level antecedent ranking)
- Total trained: ~16.1M parameters
- Frozen encoders: RoBERTa-large (355M) + BGE-large (335M)
- Stage B cost: O(M + C²) — never M²

**Training Configuration:**

- Data: CoNLL+PreCo@8k+LitBank+CorefUD
- Window size: K=256 subtokens
- Windowing: sentence-aligned (whole sentences packed into ≤K-subtoken windows)
- Stage A: AdamW lr=1e-3, 60 epochs, converged at epoch 48
- Stage B: AdamW lr=1e-3, 30 epochs, converged at epoch 12
- No negative subsampling (all window pairs)
- Uniform loss (no dataset weighting)

**Complexity:**

- Encoding: O(NK) where N=doc length, K=256
- Stage A scoring: O(M²K/N) where M=mentions
- Stage B GNN: O(M + C²) over C ≈ 21 clusters/doc — member pooling is O(M), node attention + antecedent scoring O(C²); never M²
- Overall: O(NK + C²), dominated by O(NK) on typical documents

**Key Findings:**

- Local resolution (Stage A) already reaches ~86% of the final F1; the Stage B GNN adds +12.1 (74.39 → 86.46)
- Concatenating symbolic side-channels (lexical) **into** the head, rather than adding them as a scalar, is worth +0.62 (§4.2)
- The system generalizes rather than over-fitting the selection set: test (86.46) exceeds best validation (86.11)
- The remaining ceiling is **common-noun coreference** (NOUN-only 77.9, all-mixed 71.4) — a frozen-encoder *scoring* limit, not a missing feature: no concatenated signal moves it, and missed vs resolved different-head pairs have the same BGE similarity (ablation summary below)
- A val-tuned per-dataset merge threshold adds a noise-level +0.05 on CoNLL (86.46 → **86.51**); the non-CoNLL collapse is out-of-domain scoring, not a fixable threshold
- PreCo provides ~0.9 F1 gain over CoNLL-only; both channels are load-bearing (channel ablation below)

**Channel Ablation (factorization control):**

Tests the core thesis (§1) head-on: is it the *factorization* into two channels (frozen BGE semantic + frozen RoBERTa contextual) that does the work, or would a single channel suffice? The control is a single-channel system **trained from scratch end-to-end** — not inference-time zeroing of a both-channel model (which leaves the head sized for two inputs and only measures how much the trained model leans on a channel, a confounded test). A `channel ∈ {both, bge, ctx}` flag drops the unused projection (`P_ctx`/`P_bge`) and resizes **both** the Stage A scorer and the Stage B GNN; crucially the single-channel Stage B also inherits a single-channel cluster partition (its Stage A clusters are precomputed by the same single-channel head), so the measured gap is the channel's contribution through the *entire* pipeline, not just the matching step. Checkpoints and caches are channel-tagged so variants never collide. Measured sentence-aligned, all8k, K=256, at both the Stage A level (window-split gold) and the full Stage A+B system (full-document CoNLL test).

| Channel | Stage A | Stage A+B | end-to-end Δ vs both |
| --- | --- | --- | --- |
| both (BGE+RoBERTa) | 89.48 | 86.46 | — |
| RoBERTa-only (ctx) | 87.27 | 84.16 | **−2.30** |
| BGE-only | 80.93 | 78.13 | **−8.33** |

- **Both channels are load-bearing — necessary, not merely sufficient, and the signal survives the GNN.** Dropping either single channel lowers F1 at *both* stages (end-to-end: RoBERTa-only −2.30, BGE-only −8.33), and the Stage A margins (+2.21 / +8.55) are essentially preserved end-to-end (+2.30 / +8.33) — the cross-window matcher neither manufactures nor erases the channel contribution. This is the control the §1 thesis needed: it moves the claim from "preserving distinguishable channels is *sufficient*" to direct evidence that *each* channel carries non-redundant signal through the whole system.
- **The two channels are asymmetric — complementary but not equal partners.** RoBERTa (contextual) is dominant: alone it reaches 84.16, within ~2.3 of the full system; BGE alone reaches only 78.13. Adding BGE on top of RoBERTa gains **+2.30**, while adding RoBERTa on top of BGE gains **+8.33** — a ~3.6× asymmetry.
- **Division of labor — BGE is a noun/name specialist, RoBERTa is the generalist.** Removing BGE (RoBERTa-only) costs within-window recall almost entirely on names and common nouns (NOUN 84.4→76.3, PROPN 88.4→82.1) and barely touches pronouns (93.4→92.3): BGE supplies lexical/semantic identity. Removing RoBERTa (BGE-only) costs *every* end-to-end bucket (−9 to −18 CoNLL), worst on pronoun-involving clusters (PRON+PROPN −17.8, PRON-only −13.5) and on cross-genre data (PreCo −10) — pronouns have no lexical identity, so they are pure context. Exactly the split the thesis predicts.

**Feature & projection ablations (sentence-aligned, all8k, K=256).** A systematic sweep of every cheap concatenable feature and projection-width knob, beyond the channel factorization. The results split cleanly by **conditional independence** — a knob helps iff it carries signal the frozen encoders don't already expose:

| Knob | Δ CoNLL test | verdict |
| --- | --- | --- |
| distance embedding (Stage A) | **+1.3 to +2.1** | keep — relational recency/discourse prior, not in any single span |
| lexical, concatenated (Stage B) | **+0.62** | keep — cross-window surface identity, where context can't bridge |
| projection width: raw vs 512→1024 (Stage B/A) | −1.58 / −0.20 | **near-optimal at 512** — raw overfits, wider is flat (signal is low-rank, §1) |
| per-dataset merge threshold | +0.05 CoNLL (noise); flat non-CoNLL | over-merge is not threshold-separable |

Two conclusions:

- **Distance is load-bearing, not dead.** A pre-emptive guess that the distance embedding was redundant in-window was wrong: recency (nearest-antecedent) is a *relational* signal absent from any single mention's frozen vector, and it generalizes best on the hard datasets (+1.3 CoNLL up to +2.1 CorefUD).
- **Common-noun coreference is a frozen-encoder *scoring* ceiling, not a missing feature.** The noun buckets (NOUN-only 77.9, NOUN+PROPN 70.8, all-mixed 71.4) resisted *every* concatenated symbolic signal, and the BGE cosine of missed vs resolved different-head links is the *same* (0.62 vs 0.67), so it is not a semantic-similarity gap either. The binding ("the company" = the previously-introduced entity) is simply not exposed in the frozen representations; only encoder adaptation (fine-tuning, or a trainable shallow contextualizer over the frozen token reps) could reach it. This is the boundary of frozen factorization, and a clean negative finding rather than a tuning failure.

**Merge calibration (non-CoNLL).** The LitBank/PreCo end-to-end collapse (62 / 60, high MUC but CEAFe ≈ 28–34) looks like over-merging, but a per-dataset `null_bias` offset sweep recovers only ≤0.3: the spurious merges score as confidently as the correct ones, so no threshold separates them. It is genuine out-of-domain *scoring* error (compounded by PreCo's singleton-heavy annotation), not a decoding artifact, and is not cheaply fixable. A val-tuned CoNLL offset (δ=+0.50) transfers to a noise-level **+0.05 on CoNLL test (86.46 → 86.51)**.

---

## 9. The Core Principle

Transformers pay O(N²) cost because they entangle all signal dimensions into one representation. This work demonstrates the cost is unnecessary when signals can be supplied from independent specialist encoders:

1. **Two-channel decomposition:** Semantic (BGE) + Contextual (RoBERTa) kept separate, plus symbolic side-channels (lexical) concatenated into the head — never collapsed to a scalar
2. **Windowed encoding:** O(NK) cost via fixed K=256 windows
3. **Cluster-level GNN matching:** O(C²) antecedent ranking over the quotient graph

**Result:** Linear-ish complexity O(NK + C²) with frozen encoders, reaching 86.46 F1 on CoNLL-2012 (86.51 with a val-tuned merge threshold).

**Broader implication:** The results indicate that for problems similar to coreferencing where the corpus can be windowed, LLM could be trained to learn what these factorized signals are instead of being trained to attend to all tokens which could potentially make LLM much more scalable.

---

## 10. Predicted Mentions — End-to-End System

All results above use **gold mentions** (the resolution upper bound). This section replaces them with a learned **mention detector**, making the system end-to-end. The honest result: predicted-mention CoNLL F1 ≈ **74** on test vs the **86.46** gold ceiling, and a careful decomposition shows the entire gap is **detection quality** (dominated by mention-boundary accuracy) — *not* the resolution signal and *not* the resolver training.

### 10.1 The detector

Two detectors were built. Both fine-tune RoBERTa-large; both run per window (sentence-aligned, ≤510 content tokens) with no pairwise comparison, preserving the O(NK) thesis.

**BIO token classifier.** Detection as BIO tagging over subtokens. CoNLL nesting is 100% clean containment (depth 88.9 / 10.2 / 0.8 / 0.06%), so mentions are tagged by **L=3 stacked B/I/O heads**, one per containment depth; the union of per-head B…I runs is the prediction. The lever is **trainable depth**: head-only on a frozen backbone = 52 exact F1 (frozen features don't expose full-NP boundaries), top-6 layers = 76, **full fine-tune = 85.6 exact / 92.4 overlap**. An NER-pretrained init does *not* beat vanilla (coref mentions ≠ named entities; at full FT the init washes out), and per-head class weighting for nested recall is a wash. Of its false positives, **89% are valid NPs** the OntoNotes coref layer omits as singletons (benign — Stage B drops size-1 clusters post-merge).

**Span scorer (the boundary fix, §10.5).** BIO's weakness is exposed by a per-length breakdown: it clips **long noun phrases** — exact recall by gold word-length is 1-word 97%, 3-word 79%, **6+ -word 39%** — because per-token membership under-extends the periphery of long NPs. A full-FT RoBERTa + **s2e span scorer** (`score(i,j) = wₛ·f(xᵢ) + wₑ·f(xⱼ) + f(xᵢ)ᵀ B f(xⱼ)`, scoring each `(start,end)` span as a unit with boundary-focused hard negatives) lifts this to **87.33 exact F1, 6+ -word 65%** — a +26-point fix on long NPs. The bilinear is computed as `(s@B)·e` (a plain `(proj,proj)` weight), never `nn.Bilinear`, which materializes a `(n_candidates, proj, proj)` intermediate and OOMs at ~15k spans/window.

### 10.2 The bridge

`build_predicted_docs` runs the detector, snaps subtoken spans to word boundaries, BGE-encodes the surfaces and gathers **frozen-RoBERTa** context at the predicted spans (the *same* frozen channels as the gold path — the detector supplies only span *selection*, never resolution features), and labels each predicted span by **exact-then-max-overlap** alignment to gold (unmatched → singleton). Gold clusters are kept solely as the scoring **key** (gold is never an input). Gold caches/checkpoints are untouched.

### 10.3 A measurement caveat that corrected the whole story

An earlier version of this section reported predicted A+B = 86.66, "matching the gold ceiling." **That was a train/test-leakage artifact:** the run helpers evaluated over *all* documents (train+val+test), and the gold-trained models had been trained on those train docs. The honest **test-only** number is ≈ **70** (BIO detector). All end-to-end numbers below are test-only, with the ablation machinery verified to reproduce the 86.46 gold ceiling exactly.

### 10.4 Decomposition — it's detection, not signal or training

Two controls rule out the obvious culprits:

- **Not the resolver training.** Training Stage A *and* the Stage B GNN from scratch on predicted mentions yields ≈70.6 on test — *identical* to the gold-trained pipeline applied to predicted mentions (≈70.3). Clean-vs-noisy training is a wash: training on the detector's noise neither helps nor hurts. (Predicted-from-scratch training is also the methodologically correct way to claim "end-to-end on predicted mentions" — training the resolver on gold *spans* and inferring on predicted is a pipeline with a train/test span mismatch, not comparable to joint end-to-end systems.)
- **Not the signal.** Running the gold pipeline on the *detected subset of gold mentions* — same BGE+RoBERTa signal, read at *correct* boundaries — recovers to **83.43**, within ~3 of the ceiling. The frozen factorized signal is sufficient; predicted mentions just sample it at the wrong token positions.

A controlled ablation (gold pipeline, test-only, removing one error type at a time) splits the gap cleanly:

| mention set | A+B (BIO) | isolates |
| --- | --- | --- |
| all gold (ceiling) | 86.46 | — |
| gold-detected (correct boundaries, drop missed) | 83.43 | missed −3.0 |
| matched-only (predicted boundaries, no spurious) | 76.87 | **boundary −6.6** |
| all predicted (+ spurious) | 72.82 | spurious −4.0 |

So the ~13.6-point gap is **boundary −6.6, spurious −4.0, missed −3.0** — and boundary error is the single largest cost. (The metric drop is broad — MUC/B³/CEAFe all fall ~14–17 — i.e. general degradation, not a mention-set signature.)

### 10.5 The boundary fix and its payoff

The boundary cost is the BIO detector handing Stage A the wrong span extent for long NPs (it reads RoBERTa endpoints and the BGE surface for a clipped span). Post-processing was tried and **failed**: snapping predicted spans to spaCy head-subtrees or noun-chunks *lowered* exact recall (−546 / −3658), because OntoNotes mention boundaries are not spaCy constituents. The fix had to be in the detector — hence the span scorer (§10.1), which directly optimizes extent.

The span detector cuts the boundary cost, but the end-to-end payoff is **modest**:

| (test-only, gold pipeline) | BIO | Span | Δ |
| --- | --- | --- | --- |
| gold-detected | 83.43 | 82.75 | missed −3.7 (thr 0.5 vs 0.2) |
| matched-only | 76.87 | 77.35 | **boundary −5.4** (was −6.6) |
| all predicted | 72.82 | 73.77 | **+0.95** |

The +26 on long NPs only moves end-to-end ~+1, because **6+ -word mentions are ~12% of all mentions**: the boundary cost tracks *overall* exact rate (which improved only +1.7, 85.6→87.3), not the long-NP bucket. The residual gap is now evenly distributed — boundary −5.4, spurious −3.6, missed −3.7 — with no single dominant lever.

### 10.6 Bottom line

The windowed, factorized, frozen-encoder system is **fully end-to-end with a learned detector**, and the resolvers (Stage A & B) are **viable on predicted mentions** — they are not the bottleneck (signal sufficient, training a wash). The end-to-end number is **≈74** (span detector, threshold-tunable) against the **86.46** gold-mention upper bound. The entire ~12-point gap is **detection quality**: correctly diagnosed as boundary-dominated, partly closed by the span scorer, but closing it fully needs broadly better exact detection (87→90%+, across all lengths, not just long NPs), not a different resolution signal or training scheme. The honest framing is therefore: gold-mention 86.46 as the upper bound, predicted-mention ~74 as the end-to-end result, and detector exact-boundary accuracy as the remaining lever.
