import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from transformers import AutoModel, AutoTokenizer

from disambiguation.paths import MODELS_DIR

CKPT_NAME = "stage2_global_coref.pt"
CKPT_B_NAME = "stage2_cluster_matcher.pt"
BACKBONE = "roberta-large"
BGE_DIM = 1024
CTX_DIM = 1024
CONTENT = 128  # tokens per fixed window
WINDOW = CONTENT + 2  # + <s>/</s>


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(BACKBONE)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(out.last_hidden_state, p=2, dim=-1)


def encode_document_ctx(
    content_ids: np.ndarray, encoder: "ContextEncoder", cls_id: int, sep_id: int, device: str
) -> torch.Tensor:
    n = len(content_ids)
    spans = [(s, min(s + CONTENT, n)) for s in range(0, n, CONTENT)]
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


def compute_mention_ctx_vecs(
    ctx: torch.Tensor, span_sub: np.ndarray
) -> torch.Tensor:
    # Returns (M, 2, CTX_DIM): (ctx_start, ctx_end) per mention.
    M = len(span_sub)
    out = torch.zeros(M, 2, ctx.shape[1], dtype=ctx.dtype, device=ctx.device)
    for k, (s, e) in enumerate(span_sub):
        out[k, 0] = ctx[int(s)]
        out[k, 1] = ctx[int(e)]
    return out


# ── Stage A ───────────────────────────────────────────────────────────────────

class AntecedentScorer(nn.Module):
    def __init__(self, proj_dim: int = 1024, hidden: int = 1024, dropout: float = 0.3, chunk: int = 8192):
        super().__init__()
        self.chunk = chunk
        self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)
        self.P_bge = nn.Linear(BGE_DIM, proj_dim)
        self.register_buffer("dist_bounds", torch.tensor([2, 3, 4, 5, 8, 16, 32, 64]))
        self.dist_emb = nn.Embedding(len(self.dist_bounds) + 1, 32)
        g = 2 * proj_dim
        self.ffnn = nn.Sequential(
            nn.Linear(2 * g + 32, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
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
            feat = torch.cat([gi, gj, self.dist_emb(bucket)], dim=-1)
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
    def __init__(self, proj_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)
        self.P_bge = nn.Linear(BGE_DIM, proj_dim)
        self.query = nn.Parameter(torch.randn(2 * proj_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, ctx: torch.Tensor, bge: torch.Tensor) -> torch.Tensor:
        # ctx: (M, 2, CTX_DIM), bge: (M, BGE_DIM) -> (2*proj_dim,)
        m = torch.cat([self.P_ctx(self.drop(ctx.flatten(1))), self.P_bge(self.drop(bge))], dim=-1)  # (M, 2*proj_dim)
        attn = torch.softmax(m @ self.query / (m.shape[-1] ** 0.5), dim=0)  # (M,)
        return attn @ m  # (2*proj_dim,)


class ClusterMatcher(nn.Module):
    def __init__(self, proj_dim: int = 1024, hidden: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.cluster_enc = ClusterEncoder(proj_dim, dropout)
        g = 2 * proj_dim
        self.ffnn = nn.Sequential(
            nn.Linear(2 * g, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.null_bias = nn.Parameter(torch.zeros(1))

    def encode_clusters(
        self, clusters: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        # clusters: list of (ctx, bge) -> (C, 2*proj_dim)
        return torch.stack([self.cluster_enc(ctx, bge) for ctx, bge in clusters])

    def forward(
        self,
        left: list[tuple[torch.Tensor, torch.Tensor]],
        right: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # returns (L, R) score matrix
        gl = self.encode_clusters(left)   # (L, 2*proj_dim)
        gr = self.encode_clusters(right)  # (R, 2*proj_dim)
        L, R = gl.shape[0], gr.shape[0]
        gi = gl.unsqueeze(1).expand(L, R, -1)
        gj = gr.unsqueeze(0).expand(L, R, -1)
        return self.ffnn(torch.cat([gi, gj], dim=-1)).squeeze(-1)  # (L, R)


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


def decode_cluster_matches(
    scores: np.ndarray, null_bias: float
) -> list[tuple[int, int]]:
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


def decode_antecedents(
    scores: np.ndarray, ante_mask: np.ndarray, null_bias: float = 0.0
) -> list[list[int]]:
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
