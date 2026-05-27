import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

MODELS_DIR = Path(__file__).parent.parent.parent / "cache" / "models"
CKPT_NAME = "stage2_global_coref.pt"
BACKBONE = "roberta-large"
BGE_DIM = 1024
CTX_DIM = 1024        # roberta-large hidden size
D_MODEL = 512         # mention representation dim
WIDTH_DIM = 32        # span-width embedding dim
MAX_WIDTH = 30        # span widths >= this share the last bucket
CONTENT = 128         # context subtokens per sliding window
STRIDE = 64           # window step; overlap = CONTENT - STRIDE = 64
WINDOW = CONTENT + 2  # + <s>/</s>
TOP_K = 50            # max antecedent candidates per mention
SENT_DIST_BINS = 11   # buckets: 0,1,2,3,4,5,6,7,8-11,12-19,20+
SENT_DIST_DIM = 16



class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(BACKBONE)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(out.last_hidden_state, p=2, dim=-1)  # (B, L, CTX_DIM) unit vectors


def encode_document_ctx(content_ids: np.ndarray, encoder: "ContextEncoder", cls_id: int, sep_id: int, device: str) -> torch.Tensor:
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
        ids[k, 1:1 + len(w)] = w
        ids[k, 1 + len(w)] = sep_id
        mask[k, :2 + len(w)] = 1
    out = encoder(torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device))  # (W, L, CTX_DIM)
    pos_list, val_list = [], []
    for k, (s, e) in enumerate(spans):
        pos_list.append(torch.arange(s, e, device=device))
        val_list.append(out[k, 1:1 + (e - s)])
    pos = torch.cat(pos_list)
    vals = torch.cat(val_list, dim=0)
    ctx = torch.zeros(n, CTX_DIM, device=device, dtype=vals.dtype).index_add(0, pos, vals)
    counts = torch.zeros(n, device=device, dtype=vals.dtype).index_add(0, pos, torch.ones(len(pos), device=device, dtype=vals.dtype))
    ctx = ctx / counts.unsqueeze(1)
    return F.normalize(ctx, p=2, dim=-1)


def compute_mention_ctx_vecs(ctx: torch.Tensor, span_sub: np.ndarray) -> torch.Tensor:
    # Extract (ctx_start, ctx_end, ctx_mean) for each mention from the document context tensor.
    # Returns (M, 3, CTX_DIM) float32. ctx_mean is the uniform mean over the span's subtokens.
    M = len(span_sub)
    out = torch.zeros(M, 3, ctx.shape[1], dtype=ctx.dtype, device=ctx.device)
    for k, (s, e) in enumerate(span_sub):
        s, e = int(s), min(int(e) + 1, ctx.shape[0])  # e is inclusive in span_sub
        out[k, 0] = ctx[s]           # start
        out[k, 1] = ctx[e - 1]       # end
        out[k, 2] = ctx[s:e].mean(0) # mean
    return out


def gather_spans_tensor(ctx: torch.Tensor, span_sub: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Legacy: kept for finetune path. Slices span subtoken windows for live encoding.
    _MAX_SPAN_SUB = 30
    lens = [min(int(e) - int(s) + 1, _MAX_SPAN_SUB) for s, e in span_sub]
    S = max(lens)
    pieces = [F.pad(ctx[int(s):int(s) + L], (0, 0, 0, S - L)) for (s, _), L in zip(span_sub, lens)]
    return torch.stack(pieces), torch.tensor(lens, dtype=torch.long, device=device)


class MentionEncoder(nn.Module):
    def __init__(self, ctx_dim: int = CTX_DIM, bge_dim: int = BGE_DIM, d_model: int = D_MODEL, dropout: float = 0.3):
        super().__init__()
        # Span rep: [ctx_start; ctx_end; ctx_mean; width_emb] ++ mention-surface BGE.
        # ctx_start/end/mean are precomputed from frozen RoBERTa (mean pooling over span subtokens).
        # Mean pooling replaces learned attention: median span is 1 subtoken so learned attn is degenerate,
        # and precomputing 3 fixed vectors per mention lets everything fit in RAM with no disk I/O.
        self.width_emb = nn.Embedding(MAX_WIDTH, WIDTH_DIM)
        self.proj = nn.Linear(ctx_dim * 3 + WIDTH_DIM + bge_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, ctx_start: torch.Tensor, ctx_end: torch.Tensor, ctx_mean: torch.Tensor,
                mention_bge: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        w = self.width_emb(width.clamp(max=MAX_WIDTH - 1))
        g = torch.cat([ctx_start, ctx_end, ctx_mean, w, mention_bge], dim=-1)
        return self.drop(F.relu(self.proj(g)))  # (M, d_model)


class AntecedentScorer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, hidden: int = 512, dropout: float = 0.3, chunk: int = 128):
        super().__init__()
        self.chunk = chunk
        # Unary mention score: how salient is this mention as an antecedent candidate?
        self.mention_score = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.sent_dist_emb = nn.Embedding(SENT_DIST_BINS, SENT_DIST_DIM)
        # Pairwise: [r_i, r_j, r_i - r_j] — difference is the discriminatory signal; product is redundant given individual reps
        self.mlp = nn.Sequential(nn.Linear(d_model * 3 + SENT_DIST_DIM, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, reps: torch.Tensor, sent_ids: torch.Tensor, top_k: int = TOP_K) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (M, M) score matrix and (M, M) bool candidate mask (True = valid antecedent slot).
        M = reps.shape[0]
        unary = self.mention_score(reps).squeeze(-1)  # (M,)

        idx = torch.arange(M, device=reps.device)
        ante_mask = idx.unsqueeze(0) < idx.unsqueeze(1)  # (M, M) j < i

        # Top-K pruning: for each mention i, keep only the top_k antecedents by unary score
        if top_k < M:
            # unary scores for candidates; mask non-antecedents with -inf before topk
            cand_scores = unary.unsqueeze(0).expand(M, -1).masked_fill(~ante_mask, float("-inf"))
            k = min(top_k, M - 1)
            topk_vals, _ = cand_scores.topk(k, dim=1)
            threshold = topk_vals[:, -1].unsqueeze(1)  # (M, 1) minimum score in top-k
            ante_mask = ante_mask & (cand_scores >= threshold)

        # Sentence distance embedding — vectorised bucket mapping
        dist_raw = (sent_ids.unsqueeze(1) - sent_ids.unsqueeze(0)).abs()  # (M, M) int
        # buckets: 0-7 identity, 8-11->8, 12-19->9, 20+->10
        dist_buckets = dist_raw.clamp(max=7)
        dist_buckets = torch.where(dist_raw > 7,  torch.full_like(dist_raw, 8),  dist_buckets)
        dist_buckets = torch.where(dist_raw > 11, torch.full_like(dist_raw, 9),  dist_buckets)
        dist_buckets = torch.where(dist_raw > 19, torch.full_like(dist_raw, 10), dist_buckets)
        dist_emb = self.sent_dist_emb(dist_buckets)  # (M, M, SENT_DIST_DIM)

        rows = []
        for s in range(0, M, self.chunk):
            r_i = reps[s:s + self.chunk].unsqueeze(1).expand(-1, M, -1)   # (c, M, d)
            r_j = reps.unsqueeze(0).expand(r_i.shape[0], -1, -1)          # (c, M, d)
            d_emb = dist_emb[s:s + self.chunk]                             # (c, M, SENT_DIST_DIM)
            feats = torch.cat([r_i, r_j, r_i - r_j, d_emb], dim=-1)
            pair = self.mlp(feats).squeeze(-1)                             # (c, M)
            u_i = unary[s:s + self.chunk].unsqueeze(1)                     # (c, 1)
            u_j = unary.unsqueeze(0)                                       # (1, M)
            rows.append(pair + u_i + u_j)
        return torch.cat(rows, dim=0), ante_mask  # (M, M), (M, M)


def mll_loss(scores: torch.Tensor, ante_mask: torch.Tensor, cluster_id: torch.Tensor, sent_id: torch.Tensor | None = None) -> torch.Tensor:
    # Mention-ranking marginal log-likelihood over the pruned candidate set.
    M = scores.shape[0]
    idx = torch.arange(M, device=scores.device)
    if sent_id is not None:
        ante_mask = ante_mask & (sent_id.unsqueeze(0) == sent_id.unsqueeze(1))
    neg = torch.finfo(scores.dtype).min
    null_col = torch.zeros(M, 1, device=scores.device, dtype=scores.dtype)
    denom = torch.logsumexp(torch.cat([null_col, scores.masked_fill(~ante_mask, neg)], dim=1), dim=1)  # (M,)
    gold = (cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1)) & ante_mask                            # (M, M)
    has_gold = gold.any(dim=1)
    num_gold = torch.logsumexp(scores.masked_fill(~gold, neg), dim=1)
    num = torch.where(has_gold, num_gold, torch.zeros_like(num_gold))
    return (denom - num)[idx >= 1].mean()


def decode_antecedents(scores: np.ndarray, ante_mask: np.ndarray, sent_id: np.ndarray | None = None) -> list[list[int]]:
    # Each mention links to its single best antecedent in the pruned candidate set if score > null (0).
    M = scores.shape[0]
    parent = list(range(M))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(1, M):
        mask = ante_mask[i]
        if sent_id is not None:
            mask = mask & (sent_id[:M] == sent_id[i])
        cand = np.where(mask)[0]
        if len(cand) == 0:
            continue
        j = int(cand[np.argmax(scores[i, cand])])
        if scores[i, j] > 0.0:
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(BACKBONE)
