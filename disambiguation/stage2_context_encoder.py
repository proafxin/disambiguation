import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

CKPT_NAME = "stage2_global_coref.pt"
BACKBONE = "roberta-large"
BGE_DIM = 1024
CTX_DIM = 1024  # roberta-large hidden size
D_MODEL = 512  # mention representation dim
CONTENT = 128  # context subtokens per sliding window
STRIDE = 64  # window step; overlap = CONTENT - STRIDE = 64
WINDOW = CONTENT + 2  # + <s>/</s>
TOP_K = 50  # max antecedent candidates per mention
SENT_DIST_BINS = 11  # buckets: 0,1,2,3,4,5,6,7,8-11,12-19,20+
SENT_DIST_DIM = 16


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
    # Tile the document with CONTENT-token windows at STRIDE, encode all windows in one forward,
    # average overlapping positions, L2-normalize. Returns a differentiable (n_tokens, CTX_DIM) tensor.
    n = len(content_ids)
    spans, start = [], 0
    while True:
        end = min(start + CONTENT, n)
        spans.append((start, end))
        if end >= n:
            break
        start += STRIDE
    width = max(e - s for s, e in spans) + 2
    ids = np.ones((len(spans), width), dtype=np.int64)  # roberta pad id = 1
    mask = np.zeros((len(spans), width), dtype=np.int64)
    for k, (s, e) in enumerate(spans):
        w = content_ids[s:e]
        ids[k, 0] = cls_id
        ids[k, 1 : 1 + len(w)] = w
        ids[k, 1 + len(w)] = sep_id
        mask[k, : 2 + len(w)] = 1
    out = encoder(torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device))  # (W, L, CTX_DIM)
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
    ctx: torch.Tensor, span_sub: np.ndarray, sent_offsets: np.ndarray, sent_lengths: np.ndarray
) -> torch.Tensor:
    # Returns (M, 4, CTX_DIM): ctx_start, ctx_end, ctx_mean, ctx_sent_mean.
    M = len(span_sub)
    out = torch.zeros(M, 4, ctx.shape[1], dtype=ctx.dtype, device=ctx.device)
    for k, (s, e) in enumerate(span_sub):
        out[k, 0] = ctx[int(s)]
        out[k, 1] = ctx[int(e)]
        out[k, 2] = ctx[int(s) : int(e) + 1].mean(0)
        so, sl = int(sent_offsets[k]), int(sent_lengths[k])
        out[k, 3] = ctx[so : so + sl].mean(0)
    return out


def gather_spans_tensor(ctx: torch.Tensor, span_sub: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Legacy: kept for finetune path. Slices span subtoken windows for live encoding.
    _MAX_SPAN_SUB = 30
    lens = [min(int(e) - int(s) + 1, _MAX_SPAN_SUB) for s, e in span_sub]
    S = max(lens)
    pieces = [F.pad(ctx[int(s) : int(s) + L], (0, 0, 0, S - L)) for (s, _), L in zip(span_sub, lens, strict=False)]
    return torch.stack(pieces), torch.tensor(lens, dtype=torch.long, device=device)


class MentionEncoder(nn.Module):
    def __init__(self, ctx_dim: int = CTX_DIM, bge_dim: int = BGE_DIM, d_model: int = D_MODEL, dropout: float = 0.3):
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim * 2, d_model)
        self.bge_proj = nn.Linear(bge_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, ctx_start: torch.Tensor, ctx_sent_mean: torch.Tensor, mention_bge: torch.Tensor
    ) -> torch.Tensor:
        ctx = self.drop(F.relu(self.ctx_proj(torch.cat([ctx_start, ctx_sent_mean], dim=-1))))
        bge = self.drop(F.relu(self.bge_proj(mention_bge)))
        return ctx + bge  # (M, d_model)


class AntecedentScorer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, hidden: int = 512, dropout: float = 0.3, chunk: int = 128):
        super().__init__()
        self.chunk = chunk
        self.mention_score = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.sent_dist_emb = nn.Embedding(SENT_DIST_BINS, SENT_DIST_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3 + SENT_DIST_DIM, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        # Learned null bias: shifts the link/no-link threshold away from the fixed 0.
        # Initialised to 0 so training starts from the same point as before.
        self.null_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, reps: torch.Tensor, sent_ids: torch.Tensor, top_k: int = TOP_K
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M = reps.shape[0]
        unary = self.mention_score(reps).squeeze(-1)  # (M,)

        idx = torch.arange(M, device=reps.device)
        ante_mask = idx.unsqueeze(0) < idx.unsqueeze(1)  # (M, M) j < i

        # Distance-aware top-K pruning: union of top-K/2 nearest + top-K/2 by unary score.
        # Guarantees nearby mentions are always candidates even if their unary score is low.
        if top_k < M:
            k_near = max(1, top_k // 2)
            k_unary = max(1, top_k - k_near)
            # Nearest candidates: j closest to i by position (j < i, so i-j is the distance)
            pos = torch.arange(M, device=reps.device).float()
            dist_pos = pos.unsqueeze(1) - pos.unsqueeze(0)  # (M, M) i - j, positive for j < i
            near_scores = dist_pos.masked_fill(~ante_mask, float("inf"))  # smaller = nearer
            near_topk = near_scores.topk(min(k_near, M - 1), dim=1, largest=False).values[:, -1].unsqueeze(1)
            near_mask = ante_mask & (near_scores <= near_topk)
            # Top unary candidates
            cand_unary = unary.unsqueeze(0).expand(M, -1).masked_fill(~ante_mask, float("-inf"))
            k_u = min(k_unary, M - 1)
            unary_topk = cand_unary.topk(k_u, dim=1).values[:, -1].unsqueeze(1)
            unary_mask = ante_mask & (cand_unary >= unary_topk)
            ante_mask = near_mask | unary_mask

        dist_raw = (sent_ids.unsqueeze(1) - sent_ids.unsqueeze(0)).abs()
        dist_buckets = dist_raw.clamp(max=7)
        dist_buckets = torch.where(dist_raw > 7, torch.full_like(dist_raw, 8), dist_buckets)
        dist_buckets = torch.where(dist_raw > 11, torch.full_like(dist_raw, 9), dist_buckets)
        dist_buckets = torch.where(dist_raw > 19, torch.full_like(dist_raw, 10), dist_buckets)
        dist_emb = self.sent_dist_emb(dist_buckets)

        rows = []
        for s in range(0, M, self.chunk):
            r_i = reps[s : s + self.chunk].unsqueeze(1).expand(-1, M, -1)
            r_j = reps.unsqueeze(0).expand(r_i.shape[0], -1, -1)
            d_emb = dist_emb[s : s + self.chunk]
            feats = torch.cat([r_i, r_j, r_i - r_j, d_emb], dim=-1)
            pair = self.mlp(feats).squeeze(-1)
            u_i = unary[s : s + self.chunk].unsqueeze(1)
            u_j = unary.unsqueeze(0)
            rows.append(pair + u_i + u_j)
        return torch.cat(rows, dim=0), ante_mask


def mll_loss(
    scores: torch.Tensor,
    ante_mask: torch.Tensor,
    cluster_id: torch.Tensor,
    null_bias: torch.Tensor,
    sent_id: torch.Tensor | None = None,
) -> torch.Tensor:
    # Mention-ranking marginal log-likelihood with:
    # 1. Learned null bias: replaces fixed 0 threshold with a trainable scalar.
    # 2. Nearest-antecedent weighting: gold antecedents closer to i get higher weight,
    #    so the model learns to build tight chains rather than relying on distant links.
    M = scores.shape[0]
    idx = torch.arange(M, device=scores.device)
    if sent_id is not None:
        ante_mask &= (sent_id.unsqueeze(0) == sent_id.unsqueeze(1))
    neg = torch.finfo(scores.dtype).min
    null_col = null_bias.expand(M, 1)  # learned threshold, not fixed 0
    denom = torch.logsumexp(torch.cat([null_col, scores.masked_fill(~ante_mask, neg)], dim=1), dim=1)
    gold = (cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1)) & ante_mask
    has_gold = gold.any(dim=1)
    # Distance weight: w(i,j) = 1 / (i - j), normalised over gold antecedents.
    # Nearest gold antecedent gets the highest weight.
    dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).float().clamp(min=1)  # (M, M) i - j
    inv_dist = (1.0 / dist).masked_fill(~gold, 0.0)
    inv_dist_sum = inv_dist.sum(dim=1, keepdim=True).clamp(min=1e-9)
    weights = inv_dist / inv_dist_sum  # (M, M) normalised, sums to 1 over gold antecedents
    # Weighted log-sum: sum_j w(i,j) * s(i,j) for gold j
    weighted_num = (weights * scores.masked_fill(~gold, 0.0)).sum(dim=1)
    num = torch.where(has_gold, weighted_num, null_bias.squeeze().expand(M))
    return (denom - num)[idx >= 1].mean()


def _uf_find(parent: list, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def decode_antecedents(
    scores: np.ndarray, ante_mask: np.ndarray, null_bias: float = 0.0, sent_id: np.ndarray | None = None
) -> list[list[int]]:
    # Links each mention to its best antecedent if score > null_bias, else opens new entity.
    M = scores.shape[0]
    parent = list(range(M))
    for i in range(1, M):
        mask = ante_mask[i]
        if sent_id is not None:
            mask &= (sent_id[:M] == sent_id[i])
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
