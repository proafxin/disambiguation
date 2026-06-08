import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

CKPT_NAME = "stage2_global_coref.pt"
CKPT_B_NAME = "stage2_cluster_matcher.pt"
BACKBONE = "roberta-large"
BGE_DIM = 1024
CTX_DIM = 1024
CONTENT = 512  # tokens per fixed window
WINDOW = CONTENT + 2  # + <s>/</s>


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(BACKBONE)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(out.last_hidden_state, p=2, dim=-1)


def encode_document_ctx(
    content_ids: np.ndarray, encoder: "ContextEncoder", cls_id: int, sep_id: int, device: str,
    window: int = CONTENT,
) -> torch.Tensor:
    n = len(content_ids)
    spans = [(s, min(s + window, n)) for s in range(0, n, window)]
    width = max(e - s for s, e in spans) + 2
    ids = np.ones((len(spans), width), dtype=np.int64)
    mask = np.zeros((len(spans), width), dtype=np.int64)
    for k, (s, e) in enumerate(spans):
        w = content_ids[s:e]
        ids[k, 0] = cls_id
        ids[k, 1 : 1 + len(w)] = w
        ids[k, 1 + len(w)] = sep_id
        mask[k, : 2 + len(w)] = 1
    out = encoder(torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device))
    pos_list, val_list = [], []
    for k, (s, e) in enumerate(spans):
        pos_list.append(torch.arange(s, e, device=device))
        val_list.append(out[k, 1 : 1 + (e - s)])
    pos = torch.cat(pos_list)
    vals = torch.cat(val_list, dim=0)
    ctx = torch.zeros(n, CTX_DIM, device=device, dtype=vals.dtype).index_add(0, pos, vals)
    counts = torch.zeros(n, device=device, dtype=vals.dtype).index_add(
        0, pos, torch.ones(len(pos), device=device, dtype=vals.dtype)
    )
    ctx /= counts.unsqueeze(1)
    return F.normalize(ctx, p=2, dim=-1)


def compute_mention_ctx_vecs(ctx: torch.Tensor, span_sub: np.ndarray) -> torch.Tensor:
    # Returns (M, 2, CTX_DIM): (ctx_start, ctx_end) per mention.
    M = len(span_sub)
    out = torch.zeros(M, 2, ctx.shape[1], dtype=ctx.dtype, device=ctx.device)
    for k, (s, e) in enumerate(span_sub):
        out[k, 0] = ctx[int(s)]
        out[k, 1] = ctx[int(e)]
    return out


# ── Stage A ───────────────────────────────────────────────────────────────────


class AntecedentScorer(nn.Module):
    def __init__(
        self,
        proj_dim: int = 1024,
        hidden: int = 1024,
        dropout: float = 0.3,
        chunk: int = 8192,
    ):
        super().__init__()
        self.chunk = chunk
        self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)
        self.P_bge = nn.Linear(BGE_DIM, proj_dim)
        self.register_buffer("dist_bounds", torch.tensor([2, 3, 4, 5, 8, 16, 32, 64]))
        self.dist_emb = nn.Embedding(len(self.dist_bounds) + 1, 32)
        g = 2 * proj_dim
        self.ffnn = nn.Sequential(
            nn.Linear(2 * g + 32, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.drop = nn.Dropout(dropout)
        self.null_bias = nn.Parameter(torch.zeros(1))

    def mention_rep(self, ctx: torch.Tensor, bge: torch.Tensor) -> torch.Tensor:
        # (M, 2*proj_dim)
        c = self.P_ctx(self.drop(ctx.flatten(1)))
        s = self.P_bge(self.drop(bge))
        return torch.cat([c, s], dim=-1)

    def forward(
        self, ctx: torch.Tensor, bge: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M = ctx.shape[0]
        device = ctx.device
        idx = torch.arange(M, device=device)
        ante_mask = idx.unsqueeze(0) < idx.unsqueeze(1)

        g = self.mention_rep(ctx, bge)
        scores = torch.zeros(M, M, device=device, dtype=g.dtype)
        i_idx, j_idx = ante_mask.nonzero(as_tuple=True)
        for s0 in range(0, i_idx.shape[0], self.chunk):
            ii, jj = i_idx[s0 : s0 + self.chunk], j_idx[s0 : s0 + self.chunk]
            gi, gj = g[ii], g[jj]
            bucket = torch.bucketize((ii - jj).clamp(min=0), self.dist_bounds, right=True)
            parts = [gi, gj, self.dist_emb(bucket)]
            feat = torch.cat(parts, dim=-1)
            scores[ii, jj] = self.ffnn(feat).squeeze(-1).to(scores.dtype)
        return scores, ante_mask


def mll_loss(
    scores: torch.Tensor,
    ante_mask: torch.Tensor,
    cluster_id: torch.Tensor,
    null_bias: torch.Tensor,
) -> torch.Tensor:
    M = scores.shape[0]
    idx = torch.arange(M, device=scores.device)
    neg = torch.finfo(scores.dtype).min
    null_col = null_bias.expand(M, 1)
    denom = torch.logsumexp(torch.cat([null_col, scores.masked_fill(~ante_mask, neg)], dim=1), dim=1)
    gold = (cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1)) & ante_mask
    has_gold = gold.any(dim=1)
    gold_num = torch.logsumexp(scores.masked_fill(~gold, neg), dim=1)
    num = torch.where(has_gold, gold_num, null_bias.squeeze().expand(M))
    return (denom - num)[idx >= 1].mean()


# ── Stage B ───────────────────────────────────────────────────────────────────


class ClusterEncoder(nn.Module):
    def __init__(self, proj_dim: int = 1024, dropout: float = 0.3, use_structured: bool = True):
        super().__init__()
        self.proj_dim = proj_dim
        self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)
        self.P_bge = nn.Linear(BGE_DIM, proj_dim)
        self.query = nn.Parameter(torch.randn(2 * proj_dim))
        self.drop = nn.Dropout(dropout)
        self.use_structured = use_structured

    def forward(self, ctx: torch.Tensor, bge: torch.Tensor) -> torch.Tensor:
        # ctx: (M, 2, CTX_DIM), bge: (M, BGE_DIM) -> output dim varies based on use_structured
        c = self.P_ctx(self.drop(ctx.flatten(1)))  # (M, proj_dim)
        s = self.P_bge(self.drop(bge))  # (M, proj_dim)
        m = torch.cat([c, s], dim=-1)  # (M, 2*proj_dim)
        attn = torch.softmax(m @ self.query / (m.shape[-1] ** 0.5), dim=0)
        pooled = attn @ m  # (2*proj_dim,)
        
        if not self.use_structured:
            return pooled
        
        # Structured cluster features: canonical, first, stats
        # Split pooled back into ctx and bge parts for consistent dimensions
        pooled_ctx, pooled_bge = pooled[:self.proj_dim], pooled[self.proj_dim:]  # each (proj_dim,)
        
        # Canonical mention: highest BGE norm = most specific
        bge_norms = torch.norm(bge, dim=-1)  # (M,)
        canonical_idx = torch.argmax(bge_norms)
        canonical_ctx, canonical_bge = c[canonical_idx], s[canonical_idx]  # each (proj_dim,)
        
        # First mention: discourse position
        first_ctx, first_bge = c[0], s[0]  # each (proj_dim,)
        
        # Cluster statistics
        stats = torch.cat([
            bge_norms.mean().unsqueeze(0),  # avg specificity
            bge_norms.std().unsqueeze(0) if len(bge) > 1 else torch.zeros(1, device=bge.device, dtype=bge.dtype),   # specificity variance (0 for singleton)
            torch.tensor([len(bge)], device=bge.device, dtype=bge.dtype),  # cluster size
        ])  # (3,)
        
        # Concatenate: pooled_ctx + pooled_bge + canonical_ctx + canonical_bge + first_ctx + first_bge + stats
        # = proj_dim + proj_dim + proj_dim + proj_dim + proj_dim + proj_dim + 3 = 6*proj_dim + 3
        # Wait, that's still wrong! We want 4*proj_dim + 3.
        # Let's use: [pooled | canonical | first | stats] where each mention part is the full 2*proj_dim concat
        # Actually, let's recompute: we want 4*proj_dim + 3 total.
        # Option: [pooled(2*p) | canonical(proj_dim) | first(proj_dim) | stats(3)] = 4*proj_dim + 3
        # So canonical and first should each be proj_dim, not 2*proj_dim
        # Let's use just the BGE projection for canonical/first to avoid doubling
        
        return torch.cat([pooled, canonical_bge, first_bge, stats], dim=-1)  # (2*proj_dim + proj_dim + proj_dim + 3) = (4*proj_dim + 3)

    def forward_batched(self, clusters: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        # clusters: list of (ctx (M,2,CTX_DIM), bge (M,BGE_DIM)) -> (C, output_dim)
        if not self.use_structured:
            # Fast path: just pooling, no structured features
            max_m = max(ctx.shape[0] for ctx, _ in clusters)
            device = self.query.device
            dtype = clusters[0][0].dtype
            ctx_pad = torch.zeros(len(clusters), max_m, 2 * CTX_DIM, device=device, dtype=dtype)
            bge_pad = torch.zeros(len(clusters), max_m, BGE_DIM, device=device, dtype=dtype)
            lengths = []
            for k, (ctx, bge) in enumerate(clusters):
                m = ctx.shape[0]
                ctx_pad[k, :m] = ctx.flatten(1)
                bge_pad[k, :m] = bge
                lengths.append(m)
            c = self.P_ctx(self.drop(ctx_pad))  # (C, max_m, proj_dim)
            s = self.P_bge(self.drop(bge_pad))  # (C, max_m, proj_dim)
            m_all = torch.cat([c, s], dim=-1)  # (C, max_m, 2*proj_dim)
            mask = torch.zeros(len(clusters), max_m, device=device)
            for k, l in enumerate(lengths):
                mask[k, :l] = 1.0
            attn = (m_all @ self.query) / (m_all.shape[-1] ** 0.5)
            attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = torch.softmax(attn, dim=1).unsqueeze(-1)
            pooled = (attn * m_all).sum(dim=1)  # (C, 2*proj_dim)
            return pooled
        
        # Structured path: compute per cluster individually to get correct canonical/first
        results = []
        for ctx, bge in clusters:
            results.append(self.forward(ctx, bge))
        return torch.stack(results)  # (C, 4*proj_dim + 3)


class ClusterMatcher(nn.Module):
    def __init__(self, proj_dim: int = 1024, hidden: int = 1024, dropout: float = 0.3, use_structured: bool = True):
        super().__init__()
        self.cluster_enc = ClusterEncoder(proj_dim, dropout, use_structured)
        # Input dimension: if structured, each cluster is (4*proj_dim + 3), pair is 2x that
        # if not structured, each cluster is (2*proj_dim), pair is 2x that
        if use_structured:
            g = 4 * proj_dim + 3
        else:
            g = 2 * proj_dim
        self.ffnn = nn.Sequential(
            nn.Linear(2 * g, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.null_bias = nn.Parameter(torch.zeros(1))

    def encode_clusters(self, clusters: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if self.cluster_enc.use_structured:
            # Structured mode: process each cluster individually to get correct dimensions
            results = []
            for ctx, bge in clusters:
                results.append(self.cluster_enc.forward(ctx, bge))
            return torch.stack(results)  # (C, 4*proj_dim + 3)
        else:
            # Fast batched mode for non-structured
            return self.cluster_enc.forward_batched(clusters)

    def forward(
        self,
        left: list[tuple[torch.Tensor, torch.Tensor]],
        right: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # returns (L, R) score matrix
        gl = self.encode_clusters(left)  # (L, 2*proj_dim)
        gr = self.encode_clusters(right)  # (R, 2*proj_dim)
        L, R = gl.shape[0], gr.shape[0]
        gi = gl.unsqueeze(1).expand(L, R, -1)
        gj = gr.unsqueeze(0).expand(L, R, -1)
        return self.ffnn(torch.cat([gi, gj], dim=-1)).squeeze(-1)  # (L, R)


class MentionMatcher(nn.Module):
    # Mention-level Stage B: instead of collapsing each cluster to one attention-pooled
    # vector, score a cluster pair by aggregating over the full mention-pair interaction
    # matrix (logsumexp ~ soft-max). Preserves per-mention evidence — the strongest single
    # mention pair (e.g. a shared proper noun) can drive the merge, which the pooled head
    # smears away. Drop-in for ClusterMatcher: same forward(left, right) -> (L, R) + null_bias.
    def __init__(self, proj_dim: int = 512, hidden: int = 1024, dropout: float = 0.3, chunk: int = 4096, iterative: bool = False):
        super().__init__()
        self.chunk = chunk
        self.iterative = iterative
        self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)
        self.P_bge = nn.Linear(BGE_DIM, proj_dim)
        self.drop = nn.Dropout(dropout)
        g = 2 * proj_dim
        self.pair = nn.Sequential(
            nn.Linear(4 * g, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # independent lexical-identity channel: additive term over [IDF-Jaccard, containment, exact],
        # kept separate from the neural score (not concatenated onto mention vectors). Zero-init so it
        # starts neutral and learns its weights. Active only when lexical features are supplied.
        self.lex = nn.Linear(3, 1, bias=False)
        nn.init.zeros_(self.lex.weight)
        self.null_bias = nn.Parameter(torch.zeros(1))

    def _vecs(self, ctx: torch.Tensor, bge: torch.Tensor) -> torch.Tensor:
        # (M, 2, CTX_DIM), (M, BGE_DIM) -> (M, 2*proj_dim)
        return torch.cat([self.P_ctx(self.drop(ctx.flatten(1))), self.P_bge(self.drop(bge))], dim=-1)

    def forward_many(self, pairs: list[tuple[list, list]], lex_list: list | None = None) -> list[torch.Tensor]:
        # Batch every window-pair's mention-pair scoring into ONE chunked FFNN + ONE segmented
        # reduction. For each window-pair p we enumerate its within-pair (Lm_p x Rm_p) mention
        # pairs and tag each with a flat cluster-pair bucket id; all pairs' features are scored
        # together, then reduced per bucket via LOG-MEAN-EXP (logsumexp - log N) and split back
        # into per-pair (L_p, R_p) matrices. log-mean-exp keeps "strongest pair dominates" but
        # removes the cluster-size bias of plain logsumexp (which over-merges large clusters).
        feats, buckets, shapes = [], [], []
        off = 0
        for left, right in pairs:
            lv = torch.cat([self._vecs(c, b) for c, b in left], 0)  # (Lm_p, g)
            rv = torch.cat([self._vecs(c, b) for c, b in right], 0)  # (Rm_p, g)
            dev = lv.device
            li = torch.cat([torch.full((c.shape[0],), i, device=dev, dtype=torch.long) for i, (c, _) in enumerate(left)])
            rj = torch.cat([torch.full((c.shape[0],), j, device=dev, dtype=torch.long) for j, (c, _) in enumerate(right)])
            lp, rp, lm, rm = len(left), len(right), lv.shape[0], rv.shape[0]
            ai = lv.unsqueeze(1).expand(lm, rm, -1)
            bj = rv.unsqueeze(0).expand(lm, rm, -1)
            feats.append(torch.cat([ai, bj, ai * bj, (ai - bj).abs()], dim=-1).reshape(lm * rm, -1))
            buckets.append((li.unsqueeze(1) * rp + rj.unsqueeze(0)).reshape(-1) + off)
            shapes.append((lp, rp))
            off += lp * rp
        feat = torch.cat(feats, 0)  # (total_mention_pairs, 4g)
        bucket = torch.cat(buckets, 0)
        score = torch.empty(feat.shape[0], device=feat.device, dtype=feat.dtype)
        for s0 in range(0, feat.shape[0], self.chunk):
            score[s0 : s0 + self.chunk] = self.pair(feat[s0 : s0 + self.chunk]).squeeze(-1)
        gmax = score.max()
        e = torch.exp((score - gmax).float())
        sums = torch.zeros(off, device=feat.device, dtype=torch.float32).index_add(0, bucket, e)
        counts = torch.zeros(off, device=feat.device, dtype=torch.float32).index_add(0, bucket, torch.ones_like(e))
        flat = gmax.float() + torch.log(sums) - torch.log(counts)  # log-mean-exp (size-bias removed)
        out, o = [], 0
        for p, (lp, rp) in enumerate(shapes):
            s = flat[o : o + lp * rp].reshape(lp, rp)
            if lex_list is not None:
                s = s + self.lex(lex_list[p].to(s.dtype)).squeeze(-1)  # additive lexical channel
            out.append(s)
            o += lp * rp
        return out

    def forward(
        self,
        left: list[tuple[torch.Tensor, torch.Tensor]],
        right: list[tuple[torch.Tensor, torch.Tensor]],
        lex: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_many([(left, right)], None if lex is None else [lex])[0]  # (L, R)


def cluster_match_loss(
    scores: torch.Tensor,
    left_cids: list[int],
    right_cids: list[int],
    null_bias: torch.Tensor,
) -> torch.Tensor:
    # MLL over right clusters + null for each left cluster.
    neg = torch.finfo(scores.dtype).min
    losses = []
    for i, lcid in enumerate(left_cids):
        gold_mask = torch.tensor([rcid == lcid for rcid in right_cids], device=scores.device)
        denom = torch.logsumexp(torch.cat([null_bias, scores[i]]), dim=0)
        if gold_mask.any():
            num = torch.logsumexp(scores[i].masked_fill(~gold_mask, neg), dim=0)
        else:
            num = null_bias.squeeze()
        losses.append(denom - num)
    return torch.stack(losses).mean()


def decode_cluster_matches(scores: np.ndarray, null_bias: float) -> list[tuple[int, int]]:
    # For each left cluster, link to best right cluster if score > null_bias.
    pairs = []
    for i in range(scores.shape[0]):
        j = int(np.argmax(scores[i]))
        if scores[i, j] > null_bias:
            pairs.append((i, j))
    return pairs


# ── Shared utilities ──────────────────────────────────────────────────────────


def _uf_find(parent: list, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def decode_antecedents(scores: np.ndarray, ante_mask: np.ndarray, null_bias: float = 0.0) -> list[list[int]]:
    M = scores.shape[0]
    parent = list(range(M))
    for i in range(1, M):
        cand = np.where(ante_mask[i])[0]
        if len(cand) == 0:
            continue
        j = int(cand[np.argmax(scores[i, cand])])
        if scores[i, j] > null_bias:
            parent[_uf_find(parent, i)] = _uf_find(parent, j)
    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(_uf_find(parent, i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(BACKBONE)
