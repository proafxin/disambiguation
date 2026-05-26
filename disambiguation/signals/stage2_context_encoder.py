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
CTX_DIM = 1024  # roberta-large hidden size
D_MODEL = 384
WINDOW = 512          # total positions per forward (incl. <s> and </s>)
STRIDE = 256          # overlap = WINDOW - 2 - STRIDE content tokens (~p95 of conll adjacent gaps)
CONTENT = WINDOW - 2  # content subtokens per window


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(BACKBONE)
        for p in self.roberta.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(out.last_hidden_state, p=2, dim=-1)  # (B, L, CTX_DIM) unit vectors


def encode_document(
    content_ids: np.ndarray,  # (n,) roberta subtoken ids, no special tokens
    model: "ContextEncoder",
    cls_id: int,
    sep_id: int,
    device: str,
) -> np.ndarray:
    n = len(content_ids)
    ctx = np.zeros((n, CTX_DIM), dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        start = 0
        while start < n:
            end = min(start + CONTENT, n)
            window = content_ids[start:end]
            ids = np.concatenate(([cls_id], window, [sep_id])).astype(np.int64)
            t = torch.from_numpy(ids).unsqueeze(0).to(device)
            mask = torch.ones_like(t)
            out = model(t, mask).squeeze(0).float().cpu().numpy()  # (len+2, CTX_DIM)
            ctx[start:end] += out[1:-1]
            counts[start:end] += 1.0
            if end == n:
                break
            start += STRIDE
    ctx /= counts[:, None]
    return (ctx / np.linalg.norm(ctx, axis=1, keepdims=True).clip(min=1e-8)).astype(np.float32)


class GlobalCorefHead(nn.Module):
    def __init__(self, bge_dim: int = BGE_DIM, ctx_dim: int = CTX_DIM, d_model: int = D_MODEL, dropout: float = 0.1):
        super().__init__()
        # Mention head rep = proj([context vector, BGE vector]); pair = [r_i, r_j, |r_i-r_j|, r_i*r_j].
        # Same recipe proven in Stage 1, with context now from frozen roberta-large instead of a small transformer.
        self.r_proj = nn.Linear(ctx_dim + bge_dim, d_model)
        self.pair_head = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, bge: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        r = self.r_proj(torch.cat([ctx, bge], dim=-1))  # (M, d_model)
        M = r.shape[0]
        r_i = r.unsqueeze(1).expand(M, M, -1)
        r_j = r.unsqueeze(0).expand(M, M, -1)
        pair = torch.cat([r_i, r_j, (r_i - r_j).abs(), r_i * r_j], dim=-1)
        return self.pair_head(pair).squeeze(-1)  # (M, M) pairwise logits


class LogisticCorefHead(nn.Module):
    def __init__(self, bge_dim: int = BGE_DIM, ctx_dim: int = CTX_DIM, d_align: int = 256):
        super().__init__()
        # Explicit 2x2 similarity S -> logistic (single linear layer). s_sem and s_ctx are raw
        # cosines (pure, no params, each within one space). s_cross compares BGE-space to
        # roberta-space, which is meaningless raw, so it gets a learned low-rank alignment.
        self.p_sem = nn.Linear(bge_dim, d_align, bias=False)
        self.p_ctx = nn.Linear(ctx_dim, d_align, bias=False)
        self.cls = nn.Linear(3, 1)  # weights over [s_sem, s_ctx, s_cross] + bias = logistic regression

    def forward(self, bge: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        s_sem = bge @ bge.t()  # bge already unit-norm
        s_ctx = ctx @ ctx.t()  # ctx already unit-norm
        u = F.normalize(self.p_sem(bge), dim=-1)
        v = F.normalize(self.p_ctx(ctx), dim=-1)
        cross = u @ v.t()
        s_cross = 0.5 * (cross + cross.t())  # symmetric: (u_i.v_j + u_j.v_i)/2
        feats = torch.stack([s_sem, s_ctx, s_cross], dim=-1)  # (M, M, 3)
        return self.cls(feats).squeeze(-1)  # (M, M) pairwise logits


def pair_bce_loss(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    M = logits.shape[0]
    tril = torch.ones(M, M, dtype=torch.bool, device=logits.device).tril(-1)  # i > j
    if tril.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[tril], gold[tril])


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(BACKBONE)
