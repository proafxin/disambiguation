from pathlib import Path

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).parent.parent.parent / "cache" / "models"
NOMINAL_POS = {"PRON", "NOUN", "PROPN"}

BGE_DIM = 384
D_MODEL = 384
MAX_LEN = 64
CKPT_NAME = "stage1_nominal_coref.pt"


class NominalCorefScorer(nn.Module):
    def __init__(
        self,
        bge_dim: int = BGE_DIM,
        d_model: int = D_MODEL,
        n_heads: int = 6,
        n_layers: int = 3,
        max_len: int = MAX_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.bge_dim = bge_dim
        self.input_proj = nn.Linear(bge_dim, d_model) if bge_dim != d_model else nn.Identity()
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Nominal rep = proj([contextual hidden, raw BGE]); pair = [r_i, r_j, |r_i-r_j|, r_i*r_j].
        self.r_proj = nn.Linear(d_model + bge_dim, d_model)
        pair_dim = d_model * 4
        self.pair_head = nn.Sequential(
            nn.Linear(pair_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        bge: torch.Tensor,        # (B, L, bge_dim) float, per-token BGE static embedding
        pad_mask: torch.Tensor,   # (B, L) bool, True = padding token
        nom_idx: torch.Tensor,    # (B, M) long, token positions of nominal heads (pad with 0)
        nom_mask: torch.Tensor,   # (B, M) bool, True = valid nominal
    ) -> torch.Tensor:
        B, L, _ = bge.shape
        pos = torch.arange(L, device=bge.device).clamp(max=self.max_len - 1)
        x = self.input_proj(bge) + self.pos_emb(pos).unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L, d_model)

        d = h.shape[-1]
        h_nom = torch.gather(h, 1, nom_idx.unsqueeze(-1).expand(-1, -1, d))          # (B, M, d_model)
        b_nom = torch.gather(bge, 1, nom_idx.unsqueeze(-1).expand(-1, -1, self.bge_dim))  # (B, M, bge_dim)
        r = self.r_proj(torch.cat([h_nom, b_nom], dim=-1))  # (B, M, d_model)

        M = r.shape[1]
        r_i = r.unsqueeze(2).expand(-1, -1, M, -1)
        r_j = r.unsqueeze(1).expand(-1, M, -1, -1)
        pair = torch.cat([r_i, r_j, (r_i - r_j).abs(), r_i * r_j], dim=-1)
        return self.pair_head(pair).squeeze(-1)  # (B, M, M) pairwise logits


def pair_bce_loss(
    logits: torch.Tensor,   # (B, M, M) logits
    gold: torch.Tensor,     # (B, M, M) float in {0, 1}, 1 where pair corefers
    nom_mask: torch.Tensor, # (B, M) bool
) -> torch.Tensor:
    B, M, _ = logits.shape
    tril = torch.ones(M, M, dtype=torch.bool, device=logits.device).tril(-1)  # i > j only
    valid = nom_mask.unsqueeze(2) & nom_mask.unsqueeze(1) & tril.unsqueeze(0)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid], gold[valid])


def _uf_find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def decode_clusters(nom_idx: np.ndarray, logits: np.ndarray, threshold: float) -> list[list[int]]:
    # Link nominal pairs whose P(coref) >= threshold, then take connected components.
    M = len(nom_idx)
    prob = 1.0 / (1.0 + np.exp(-logits))
    parent = list(range(M))
    for i in range(M):
        for j in range(i):
            if prob[i, j] >= threshold:
                parent[_uf_find(parent, i)] = _uf_find(parent, j)
    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(_uf_find(parent, i), []).append(int(nom_idx[i]))
    return [frozenset(g) for g in groups.values() if len(g) >= 2]


def load_stage1(device: str = "cpu") -> tuple["NominalCorefScorer", spacy.Language, object]:
    from sentence_transformers import SentenceTransformer

    model = NominalCorefScorer().to(device)
    ckpt = torch.load(MODELS_DIR / CKPT_NAME, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()
    nlp = spacy.load("en_core_web_lg")
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    return model, nlp, bge


def resolve_stage1(
    sent: str,
    model: "NominalCorefScorer",
    nlp: spacy.Language,
    bge: object,
    device: str = "cpu",
    threshold: float = 0.5,
) -> list[list[tuple[int, str]]]:
    doc = nlp(sent)
    tokens = [t.text for t in doc][:MAX_LEN]
    L = len(tokens)
    nominal_positions = [t.i for t in doc if t.pos_ in NOMINAL_POS and t.i < L]
    if len(nominal_positions) < 2:
        return []

    vecs = np.asarray(bge.encode(tokens, normalize_embeddings=True), dtype=np.float32)
    bge_t = torch.from_numpy(vecs).unsqueeze(0).to(device)
    pad = torch.zeros(1, L, dtype=torch.bool, device=device)
    nom_t = torch.tensor(nominal_positions, dtype=torch.long, device=device).unsqueeze(0)
    nom_mask = torch.ones(1, len(nominal_positions), dtype=torch.bool, device=device)
    with torch.inference_mode():
        logits = model(bge_t, pad, nom_t, nom_mask).squeeze(0).float().cpu().numpy()

    clusters = decode_clusters(np.array(nominal_positions), logits, threshold)
    return [[(idx, doc[idx].text) for idx in cluster] for cluster in clusters]


if __name__ == "__main__":
    torch.manual_seed(0)
    m = NominalCorefScorer()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Parameters: {n_params:,}")
    B, L, Mn = 4, 30, 8
    bge = torch.randn(B, L, BGE_DIM)
    pad = torch.zeros(B, L, dtype=torch.bool)
    pad[0, 25:] = True
    nom_idx = torch.randint(0, L, (B, Mn))
    nom_mask = torch.ones(B, Mn, dtype=torch.bool)
    nom_mask[0, 5:] = False
    out = m(bge, pad, nom_idx, nom_mask)
    print(f"logits shape: {tuple(out.shape)}  (expect ({B}, {Mn}, {Mn}))")
    gold = (torch.rand(B, Mn, Mn) > 0.7).float()
    loss = pair_bce_loss(out, gold, nom_mask)
    print(f"loss: {loss.item():.4f}")
    logits_np = out[0].detach().numpy()
    clusters = decode_clusters(np.arange(Mn), logits_np, 0.5)
    print(f"decoded clusters (sent 0): {clusters}")
