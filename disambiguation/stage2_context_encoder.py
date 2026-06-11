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
    window: int = CONTENT, spans: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    n = len(content_ids)
    # spans = encoding chunks. Default is fixed-K blocks; sentence-aligned windowing passes
    # sentence-packed chunks instead so a sentence's tokens are never split across a chunk.
    if spans is None:
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
        use_lexical: bool = False,
    ):
        super().__init__()
        self.channel = channel
        self.max_windows = max_windows
        self.member_pool = member_pool  # how a cluster's members collapse to one node vector
        self.use_lexical = use_lexical
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
        # The lexical channel (3-d) is CONCATENATED into the pair feature vector and read jointly
        # by the score MLP — not added as a separate scalar — so the head can use it conditionally
        # and non-linearly and it never couples through the shared null_bias. (An agreement channel
        # was tried the same way and removed: it was functionally redundant with the RoBERTa
        # contextual channel — see RESEARCH.md §4.2 — and added nothing.)
        sym = 3 if use_lexical else 0
        self.score = nn.Sequential(
            nn.Linear(4 * hidden + sym, hidden),
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
        self, clusters: list[tuple[torch.Tensor, torch.Tensor]], win_ids: torch.Tensor,
        lex: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # clusters ordered by first-mention position; win_ids: (C,) long window indices;
        # lex: optional (C, C, 3) lexical features concatenated into the pair score.
        # Returns (C, C) scores (row i over j<i) and the mask.
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
        parts = [hi, hj, hi * hj, (hi - hj).abs()]
        if lex is not None:
            parts.append(lex)  # (C, C, 3) lexical features, read jointly by the MLP
        feat = torch.cat(parts, dim=-1)
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
