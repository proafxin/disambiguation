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
MAX_SPAN_SUB = 30     # subtokens kept per span for attention pooling
CONTENT = 128         # context subtokens per sliding window
STRIDE = 64           # window step; overlap = CONTENT - STRIDE = 64
WINDOW = CONTENT + 2  # + <s>/</s>


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(BACKBONE)  # fine-tuned end-to-end when finetune=True

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


def gather_spans_tensor(ctx: torch.Tensor, span_sub: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Slice each mention's span subtokens out of the (n_tokens, CTX_DIM) document context and
    # right-pad to the batch's longest span. Returns ((M, S, CTX_DIM), (M,) lengths).
    lens = [min(int(e) - int(s) + 1, MAX_SPAN_SUB) for s, e in span_sub]
    S = max(lens)
    pieces = [F.pad(ctx[int(s):int(s) + L], (0, 0, 0, S - L)) for (s, _), L in zip(span_sub, lens)]
    return torch.stack(pieces), torch.tensor(lens, dtype=torch.long, device=device)


class MentionEncoder(nn.Module):
    def __init__(self, ctx_dim: int = CTX_DIM, bge_dim: int = BGE_DIM, d_model: int = D_MODEL, dropout: float = 0.1):
        super().__init__()
        # c2f span rep: [ctx_start; ctx_end; attention-pooled span ctx; width emb] ++ head-word BGE anchor.
        self.attn = nn.Linear(ctx_dim, 1)
        self.width_emb = nn.Embedding(MAX_WIDTH, WIDTH_DIM)
        self.proj = nn.Linear(ctx_dim * 3 + WIDTH_DIM + bge_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, span_ctx: torch.Tensor, span_len: torch.Tensor, head_bge: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        M, S = span_ctx.shape[0], span_ctx.shape[1]
        valid = torch.arange(S, device=span_ctx.device).unsqueeze(0) < span_len.unsqueeze(1)  # (M, S)
        scores = self.attn(span_ctx).squeeze(-1).masked_fill(~valid, float("-inf"))           # (M, S)
        pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * span_ctx).sum(1)                # (M, ctx_dim)
        start = span_ctx[:, 0]
        end = span_ctx[torch.arange(M, device=span_ctx.device), span_len - 1]
        w = self.width_emb(width.clamp(max=MAX_WIDTH - 1))
        g = torch.cat([start, end, pooled, w, head_bge], dim=-1)
        return self.drop(F.relu(self.proj(g)))  # (M, d_model)


class AntecedentScorer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, hidden: int = 512, dropout: float = 0.1, chunk: int = 128):
        super().__init__()
        self.chunk = chunk
        self.mlp = nn.Sequential(nn.Linear(d_model * 4, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        # Pairwise antecedent logits s(i, j) for the full (M, M) grid; the loss/decode mask j >= i.
        M = reps.shape[0]
        rows = []
        for s in range(0, M, self.chunk):
            r_i = reps[s:s + self.chunk].unsqueeze(1).expand(-1, M, -1)  # (c, M, d)
            r_j = reps.unsqueeze(0).expand(r_i.shape[0], -1, -1)         # (c, M, d)
            feats = torch.cat([r_i, r_j, (r_i - r_j).abs(), r_i * r_j], dim=-1)
            rows.append(self.mlp(feats).squeeze(-1))
        return torch.cat(rows, dim=0)  # (M, M)


def mll_loss(scores: torch.Tensor, cluster_id: torch.Tensor, sent_id: torch.Tensor | None = None) -> torch.Tensor:
    # Mention-ranking marginal log-likelihood. For each mention i (document order), candidates are
    # the dummy null antecedent (score 0) plus every earlier mention j < i. Maximize the probability
    # mass on correct antecedents (earlier same-cluster mentions), or on the null if i opens a cluster.
    # If sent_id is given, candidates are restricted to earlier mentions in the same sentence
    # (intra-sentence resolution): a mention whose only coreferents are in other sentences attaches to null.
    M = scores.shape[0]
    idx = torch.arange(M, device=scores.device)
    ante = idx.unsqueeze(0) < idx.unsqueeze(1)            # (M, M) True where j < i
    if sent_id is not None:
        ante = ante & (sent_id.unsqueeze(0) == sent_id.unsqueeze(1))
    neg = torch.finfo(scores.dtype).min
    null_col = torch.zeros(M, 1, device=scores.device, dtype=scores.dtype)
    denom = torch.logsumexp(torch.cat([null_col, scores.masked_fill(~ante, neg)], dim=1), dim=1)  # (M,)
    gold = (cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1)) & ante                            # (M, M)
    has_gold = gold.any(dim=1)
    num_gold = torch.logsumexp(scores.masked_fill(~gold, neg), dim=1)
    num = torch.where(has_gold, num_gold, torch.zeros_like(num_gold))  # null (score 0) is correct when no antecedent
    return (denom - num)[idx >= 1].mean()


def decode_antecedents(scores: np.ndarray, sent_id: np.ndarray | None = None) -> list[list[int]]:
    # Each mention links to its single best earlier antecedent if that score beats the null (0),
    # else opens a new entity. Clusters are the connected components of the chosen links.
    # If sent_id is given, candidates are restricted to earlier mentions in the same sentence.
    M = scores.shape[0]
    parent = list(range(M))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(1, M):
        cand = np.where(sent_id[:i] == sent_id[i])[0] if sent_id is not None else np.arange(i)
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
