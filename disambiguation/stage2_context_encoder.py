import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from transformers import AutoModel, AutoTokenizer

from disambiguation.paths import MODELS_DIR

CKPT_NAME = "stage2_global_coref.pt"
BACKBONE = "roberta-large"
BGE_DIM = 1024
CTX_DIM = 1024  # roberta-large hidden size
CONTENT = 128  # context subtokens per sliding window
STRIDE = 64  # window step; overlap = CONTENT - STRIDE = 64
WINDOW = CONTENT + 2  # + <s>/</s>


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(BACKBONE)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(out.last_hidden_state, p=2, dim=-1)  # (B, L, CTX_DIM) unit vectors


def encode_document_ctx(
    content_ids: np.ndarray, encoder: "ContextEncoder", cls_id: int, sep_id: int, device: str
) -> torch.Tensor:
    n = len(content_ids)
    spans, start = [], 0
    while True:
        end = min(start + CONTENT, n)
        spans.append((start, end))
        if end >= n:
            break
        start += STRIDE
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


class AntecedentScorer(nn.Module):
    def __init__(self, proj_dim: int = 1024, hidden: int = 1024, dropout: float = 0.3, chunk: int = 8192):
        super().__init__()
        self.chunk = chunk  # candidate pairs scored per FFNN batch
        self.P_ctx = nn.Linear(2 * CTX_DIM, proj_dim)  # contextual channel: [start; end] -> learned space
        self.P_bge = nn.Linear(BGE_DIM, proj_dim)  # semantic channel: bge -> learned space
        self.register_buffer("dist_bounds", torch.tensor([2, 3, 4, 5, 8, 16, 32, 64]))
        self.dist_emb = nn.Embedding(len(self.dist_bounds) + 1, 32)
        g = 2 * proj_dim  # per-mention representation: [ctx_proj ; bge_proj]
        self.ffnn = nn.Sequential(  # full-signal pairwise classifier: [g_i, g_j, g_i*g_j, dist]
            nn.Linear(3 * g + 32, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.drop = nn.Dropout(dropout)
        self.null_bias = nn.Parameter(torch.zeros(1))

    def mention_rep(self, ctx: torch.Tensor, bge: torch.Tensor) -> torch.Tensor:
        # (M, 2*proj_dim): projected [start;end] contextual channel concatenated with projected bge channel
        c = self.P_ctx(self.drop(ctx.flatten(1)))
        s = self.P_bge(self.drop(bge))
        return torch.cat([c, s], dim=-1)

    def forward(
        self, ctx: torch.Tensor, bge: torch.Tensor, tok_pos: torch.Tensor, window: int = CONTENT
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M = ctx.shape[0]
        device = ctx.device
        idx = torch.arange(M, device=device)
        ante = idx.unsqueeze(0) < idx.unsqueeze(1)  # (M, M) j < i
        tok = tok_pos.unsqueeze(1) - tok_pos.unsqueeze(0)  # token dist; >= 0 for j < i (mentions sorted)
        ante_mask = ante & (tok <= window)  # local: candidate only if within the context window

        g = self.mention_rep(ctx, bge)  # (M, 2*proj_dim) per-mention rep
        scores = torch.zeros(M, M, device=device, dtype=g.dtype)
        i_idx, j_idx = ante_mask.nonzero(as_tuple=True)  # in-window candidate pairs
        for s0 in range(0, i_idx.shape[0], self.chunk):
            ii, jj = i_idx[s0 : s0 + self.chunk], j_idx[s0 : s0 + self.chunk]
            gi, gj = g[ii], g[jj]
            bucket = torch.bucketize((ii - jj).clamp(min=0), self.dist_bounds, right=True)
            feat = torch.cat([gi, gj, gi * gj, self.dist_emb(bucket)], dim=-1)
            scores[ii, jj] = self.ffnn(feat).squeeze(-1).to(scores.dtype)
        return scores, ante_mask


def mll_loss(
    scores: torch.Tensor,
    ante_mask: torch.Tensor,
    cluster_id: torch.Tensor,
    null_bias: torch.Tensor,
    sent_id: torch.Tensor | None = None,
) -> torch.Tensor:
    M = scores.shape[0]
    idx = torch.arange(M, device=scores.device)
    if sent_id is not None:
        ante_mask = ante_mask & (sent_id.unsqueeze(0) == sent_id.unsqueeze(1))
    neg = torch.finfo(scores.dtype).min
    null_col = null_bias.expand(M, 1)
    denom = torch.logsumexp(torch.cat([null_col, scores.masked_fill(~ante_mask, neg)], dim=1), dim=1)
    gold = (cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1)) & ante_mask
    has_gold = gold.any(dim=1)
    # marginalize over ALL gold antecedents: any correct link yields the right cluster, so no coreferent is penalized
    gold_num = torch.logsumexp(scores.masked_fill(~gold, neg), dim=1)
    num = torch.where(has_gold, gold_num, null_bias.squeeze().expand(M))
    return (denom - num)[idx >= 1].mean()


def _uf_find(parent: list, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def decode_antecedents(
    scores: np.ndarray, ante_mask: np.ndarray, null_bias: float = 0.0, sent_id: np.ndarray | None = None
) -> list[list[int]]:
    M = scores.shape[0]
    parent = list(range(M))
    for i in range(1, M):
        mask = ante_mask[i]
        if sent_id is not None:
            mask = mask & (sent_id[:M] == sent_id[i])
        cand = np.where(mask)[0]
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
