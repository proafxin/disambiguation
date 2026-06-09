import math

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


# ── Stage A ───────────────────────────────────────────────────────────────────


class AntecedentScorer(nn.Module):
    def __init__(
        self,
        proj_dim: int = 1024,
        hidden: int = 1024,
        dropout: float = 0.3,
        chunk: int = 8192,
        channel: str = "both",
    ):
        super().__init__()
        self.chunk = chunk
        self.channel = channel
        if channel in ("both", "ctx"):
            self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)
        if channel in ("both", "bge"):
            self.P_bge = nn.Linear(BGE_DIM, proj_dim)
        self.register_buffer("dist_bounds", torch.tensor([2, 3, 4, 5, 8, 16, 32, 64]))
        self.dist_emb = nn.Embedding(len(self.dist_bounds) + 1, 32)
        g = (2 if channel == "both" else 1) * proj_dim
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
        # (M, g) where g = 2*proj_dim (both) or proj_dim (single channel)
        parts = []
        if self.channel in ("both", "ctx"):
            parts.append(self.P_ctx(self.drop(ctx.flatten(1))))
        if self.channel in ("both", "bge"):
            parts.append(self.P_bge(self.drop(bge)))
        return torch.cat(parts, dim=-1)

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


# ── Stage B ───────────────────────────────────────────────────────────────────


class ClusterEncoder(nn.Module):
    def __init__(self, proj_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.proj_dim = proj_dim
        self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)
        self.P_bge = nn.Linear(BGE_DIM, proj_dim)
        self.query = nn.Parameter(torch.randn(2 * proj_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, ctx: torch.Tensor, bge: torch.Tensor) -> torch.Tensor:
        # ctx: (M, 2, CTX_DIM), bge: (M, BGE_DIM) -> (2*proj_dim,) attention-pooled cluster vector
        c = self.P_ctx(self.drop(ctx.flatten(1)))  # (M, proj_dim)
        s = self.P_bge(self.drop(bge))  # (M, proj_dim)
        m = torch.cat([c, s], dim=-1)  # (M, 2*proj_dim)
        attn = torch.softmax(m @ self.query / (m.shape[-1] ** 0.5), dim=0)
        return attn @ m  # (2*proj_dim,)

    def forward_batched(self, clusters: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        # clusters: list of (ctx (M,2,CTX_DIM), bge (M,BGE_DIM)) -> (C, 2*proj_dim)
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
        return (attn * m_all).sum(dim=1)  # (C, 2*proj_dim)


class ClusterMatcher(nn.Module):
    def __init__(self, proj_dim: int = 1024, hidden: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.cluster_enc = ClusterEncoder(proj_dim, dropout)
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
    def __init__(self, proj_dim: int = 512, hidden: int = 1024, dropout: float = 0.3, chunk: int = 4096, channel: str = "both"):
        super().__init__()
        self.chunk = chunk
        self.channel = channel
        if channel in ("both", "ctx"):
            self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)
        if channel in ("both", "bge"):
            self.P_bge = nn.Linear(BGE_DIM, proj_dim)
        self.drop = nn.Dropout(dropout)
        g = (2 if channel == "both" else 1) * proj_dim
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
        # (M, 2, CTX_DIM), (M, BGE_DIM) -> (M, g); g = 2*proj_dim (both) or proj_dim (single channel)
        parts = []
        if self.channel in ("both", "ctx"):
            parts.append(self.P_ctx(self.drop(ctx.flatten(1))))
        if self.channel in ("both", "bge"):
            parts.append(self.P_bge(self.drop(bge)))
        return torch.cat(parts, dim=-1)

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


class ClusterGNN(nn.Module):
    # Stage B as cluster-level antecedent ranking over the quotient graph. Nodes = Stage A
    # clusters (~21/doc); a small Transformer encoder (full self-attention = fully-connected
    # GNN) contextualizes every cluster by all others, so each cluster's antecedent choice is
    # globally informed (transitivity, competition). Each cluster then ranks its single best
    # antecedent among clusters in strictly earlier windows, or null (= new entity) — Stage A is
    # authoritative within a window. The antecedent-pointer forest IS the coreference chains.
    # O(C²): attention and ranking are both over C nodes.
    def __init__(
        self,
        proj_dim: int = 512,
        hidden: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.3,
        channel: str = "both",
        max_windows: int = 64,
        member_pool: str = "attn",
    ):
        super().__init__()
        self.channel = channel
        self.max_windows = max_windows
        self.member_pool = member_pool  # how a cluster's members collapse to one node vector
        if channel in ("both", "ctx"):
            self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)
        if channel in ("both", "bge"):
            self.P_bge = nn.Linear(BGE_DIM, proj_dim)
        self.drop = nn.Dropout(dropout)
        g = (2 if channel == "both" else 1) * proj_dim
        self.member_q = nn.Parameter(torch.randn(g))  # learned query for attn pooling (unused otherwise)
        self.node_in = nn.Linear(g, hidden)
        self.win_emb = nn.Embedding(max_windows, hidden)  # window-position signal per cluster
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.gnn = nn.TransformerEncoder(layer, n_layers)
        self.score = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.null_bias = nn.Parameter(torch.zeros(1))

    def _member_vecs(self, ctx: torch.Tensor, bge: torch.Tensor) -> torch.Tensor:
        # (m, 2, CTX_DIM), (m, BGE_DIM) -> (m, g)
        parts = []
        if self.channel in ("both", "ctx"):
            parts.append(self.P_ctx(self.drop(ctx.flatten(1))))
        if self.channel in ("both", "bge"):
            parts.append(self.P_bge(self.drop(bge)))
        return torch.cat(parts, dim=-1)

    def _pool_members(self, v: torch.Tensor) -> torch.Tensor:
        # (m, g) -> (g,) cluster node vector. attn = learned-query attention (default, smears);
        # max/lse preserve the single strongest member (the discriminative proper noun a merge
        # should ride on); mean averages. lse is a size-normalized soft-max (log-mean-exp).
        if self.member_pool == "mean":
            return v.mean(0)
        if self.member_pool == "max":
            return v.max(0).values
        if self.member_pool == "lse":
            return torch.logsumexp(v, dim=0) - math.log(v.shape[0])
        a = torch.softmax(v @ self.member_q / (v.shape[-1] ** 0.5), dim=0)  # (m,)
        return a @ v

    def _node_feats(self, clusters: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        # pool each cluster's members -> (C, g)
        return torch.stack([self._pool_members(self._member_vecs(ctx, bge)) for ctx, bge in clusters])

    def forward(
        self, clusters: list[tuple[torch.Tensor, torch.Tensor]], win_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # clusters ordered by first-mention position; win_ids: (C,) long window indices.
        # Returns (C, C) antecedent scores (row i over antecedents j<i) and the j<i mask.
        C = len(clusters)
        h0 = self.node_in(self._node_feats(clusters))  # (C, hidden)
        h0 = h0 + self.win_emb(win_ids.clamp(max=self.max_windows - 1))
        # Outer residual around message passing: full self-attention over a small node set
        # over-smooths node reps to near-identical (measured cos≈1.0, zero gold/neg separation
        # -> all-null collapse). Adding h0 back guarantees node-specific signal reaches the
        # scorer, so the GNN can only *refine* the (working) pooled features, never erase them.
        h = h0 + self.gnn(h0.unsqueeze(0)).squeeze(0)  # (C, hidden)
        idx = torch.arange(C, device=h.device)
        # Cross-window only: Stage A is authoritative within a window, so a cluster may only
        # take an antecedent from a strictly earlier window (never re-merge same-window clusters).
        ante = (idx.unsqueeze(1) > idx.unsqueeze(0)) & (win_ids.unsqueeze(1) != win_ids.unsqueeze(0))
        hi = h.unsqueeze(1).expand(C, C, -1)
        hj = h.unsqueeze(0).expand(C, C, -1)
        feat = torch.cat([hi, hj, hi * hj, (hi - hj).abs()], dim=-1)
        scores = self.score(feat).squeeze(-1)  # (C, C)
        return scores, ante


def antecedent_mll_loss(
    scores: torch.Tensor,
    ante_mask: torch.Tensor,
    gold_id: torch.Tensor,
    null_bias: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    # Mention-ranking MLL over an ordered node set: each node i (i>=1) softmaxes over
    # {ε} ∪ {j < i}; the gold set is earlier nodes sharing node i's gold entity id, and ε
    # (null) is gold when no antecedent shares the entity.
    # pos_weight up-weights nodes that DO have a true antecedent. In the cross-window cluster
    # regime most nodes are null (entities rarely fragment), so an unweighted MLL collapses to
    # "always null" — the rare merge-positive nodes must be up-weighted to keep gradient.
    M = scores.shape[0]
    idx = torch.arange(M, device=scores.device)
    neg = torch.finfo(scores.dtype).min
    null_col = null_bias.expand(M, 1)
    denom = torch.logsumexp(torch.cat([null_col, scores.masked_fill(~ante_mask, neg)], dim=1), dim=1)
    gold = (gold_id.unsqueeze(0) == gold_id.unsqueeze(1)) & ante_mask
    has_gold = gold.any(dim=1)
    gold_num = torch.logsumexp(scores.masked_fill(~gold, neg), dim=1)
    num = torch.where(has_gold, gold_num, null_bias.squeeze().expand(M))
    per = denom - num
    if pos_weight == 1.0:
        return per[idx >= 1].mean()
    w = torch.where(has_gold, per.new_tensor(pos_weight), per.new_tensor(1.0))
    sel = idx >= 1
    return (per * w)[sel].sum() / w[sel].sum()


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
