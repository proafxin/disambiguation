from pathlib import Path

import numpy as np
import spacy
import torch
import torch.nn as nn

from disambiguation.signals.abstract_features import (
    DEP_IDS, ENT_TYPE_IDS, GENDER_IDS, NUMBER_IDS, POS_IDS, PRONTYPE_IDS,
)
from disambiguation.signals.train_full import _compute_depth

MODELS_DIR = Path(__file__).parent.parent.parent / "cache" / "models"
NOMINAL_POS = {"PRON", "NOUN", "PROPN"}

# Vocabulary sizes (+1 for unknown)
N_POS = len(POS_IDS) + 1
N_DEP = len(DEP_IDS) + 1
N_GENDER = len(GENDER_IDS)
N_NUMBER = len(NUMBER_IDS)
N_PRONTYPE = len(PRONTYPE_IDS)
N_ENT = len(ENT_TYPE_IDS) + 1

# Continuous features per token (non-categorical)
N_CONT = 12  # person, case/mood/tense/verbform/aspect/voice/animacy/numtype flags, ent_iob, depth, n_children, pos_in_sent


def build_sentence_graph(
    sent: spacy.tokens.Span | spacy.tokens.Doc,
    sent_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    cat = np.zeros((sent_len, 6), dtype=np.int64)   # [pos, dep, gender, number, prontype, ent_type]
    cont = np.zeros((sent_len, N_CONT), dtype=np.float32)
    edge_src = []
    edge_dst = []
    edge_type = []

    offset = sent[0].i if hasattr(sent, '__iter__') else 0

    for ti in range(sent_len):
        tok = sent[ti]
        morph = tok.morph.to_dict()

        cat[ti, 0] = POS_IDS.get(tok.pos_, len(POS_IDS))
        cat[ti, 1] = DEP_IDS.get(tok.dep_, len(DEP_IDS))
        cat[ti, 2] = GENDER_IDS.get(morph.get("Gender", "unknown"), 3)
        cat[ti, 3] = NUMBER_IDS.get(morph.get("Number", "unknown"), 2)
        cat[ti, 4] = PRONTYPE_IDS.get(morph.get("PronType", "unknown"), 5)
        cat[ti, 5] = ENT_TYPE_IDS.get(tok.ent_type_, len(ENT_TYPE_IDS))

        cont[ti, 0] = int(morph.get("Person", "0")) if morph.get("Person") else 0
        cont[ti, 1] = float(bool(morph.get("Case")))
        cont[ti, 2] = float(bool(morph.get("Mood")))
        cont[ti, 3] = float(bool(morph.get("Tense")))
        cont[ti, 4] = float(bool(morph.get("VerbForm")))
        cont[ti, 5] = float(bool(morph.get("Aspect")))
        cont[ti, 6] = float(bool(morph.get("Voice")))
        cont[ti, 7] = float(bool(morph.get("Animacy")))
        cont[ti, 8] = float(bool(morph.get("NumType")))
        cont[ti, 9] = int(tok.ent_iob_ != "O")
        cont[ti, 10] = _compute_depth(tok) / 20.0
        cont[ti, 11] = (tok.n_lefts + tok.n_rights) / max(sent_len - 1, 1)

        head_ti = tok.head.i - offset
        if tok.head != tok and 0 <= head_ti < sent_len:
            dep_id = DEP_IDS.get(tok.dep_, len(DEP_IDS))
            # child → head (dep arc direction)
            edge_src.append(ti)
            edge_dst.append(head_ti)
            edge_type.append(dep_id)
            # head → child (reverse arc, offset by N_DEP)
            edge_src.append(head_ti)
            edge_dst.append(ti)
            edge_type.append(dep_id + N_DEP)

    nominal_positions = [ti for ti in range(sent_len) if sent[ti].pos_ in NOMINAL_POS]

    # Per-nominal local lexical ids (dense within sentence): [lemma_id, surface_id].
    # Used by the pair head for same_lemma / same_surface match features.
    lemma_ids: dict[int, int] = {}
    surf_ids: dict[int, int] = {}
    nom_lex = np.empty((len(nominal_positions), 2), dtype=np.int64)
    for k, ti in enumerate(nominal_positions):
        tok = sent[ti]
        nom_lex[k, 0] = lemma_ids.setdefault(tok.lemma, len(lemma_ids))
        nom_lex[k, 1] = surf_ids.setdefault(tok.lower, len(surf_ids))

    edges = np.array([edge_src, edge_dst], dtype=np.int64) if edge_src else np.zeros((2, 0), dtype=np.int64)
    etypes = np.array(edge_type, dtype=np.int64) if edge_type else np.zeros(0, dtype=np.int64)
    return cat, cont, edges, etypes, nominal_positions, nom_lex


class DepGraphTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Categorical embeddings
        emb_dim = d_model // 8
        self.emb_pos = nn.Embedding(N_POS, emb_dim)
        self.emb_dep = nn.Embedding(N_DEP, emb_dim)
        self.emb_gender = nn.Embedding(N_GENDER, emb_dim)
        self.emb_number = nn.Embedding(N_NUMBER, emb_dim)
        self.emb_prontype = nn.Embedding(N_PRONTYPE, emb_dim)
        self.emb_ent = nn.Embedding(N_ENT, emb_dim)

        cat_dim = emb_dim * 6
        self.input_proj = nn.Linear(cat_dim + N_CONT, d_model)

        # Edge type embeddings for graph bias (dep arcs both directions)
        self.emb_edge = nn.Embedding(N_DEP * 2 + 1, n_heads)  # +1 for self-loop

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Mention-ranking head: score(mention_i, antecedent_j) → scalar.
        # Pair input = [repr_i, repr_j, signed_dist, abs_dist, same_lemma, same_surface]
        self.n_pair_feats = 4
        self.rank_head = nn.Sequential(
            nn.Linear(d_model * 2 + self.n_pair_feats, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        # Null antecedent bias (learned scalar per mention)
        self.null_score = nn.Linear(d_model, 1)

    def _build_batch_attn_bias(
        self,
        edge_list: list[tuple[np.ndarray, np.ndarray]],
        lengths: list[int],
        l_max: int,
        device: torch.device,
    ) -> torch.Tensor:
        B = len(edge_list)
        H = self.emb_edge.embedding_dim
        bias = torch.zeros(B, H, l_max, l_max, device=device, dtype=torch.float32)
        self_type = torch.tensor([N_DEP * 2], dtype=torch.long, device=device)
        self_emb = self.emb_edge(self_type).squeeze(0)  # (H,)
        for b, (edges, etypes) in enumerate(edge_list):
            L = lengths[b]
            idx = torch.arange(L, device=device)
            bias[b, :, idx, idx] = self_emb.unsqueeze(1).expand(-1, L)
            if edges.shape[1] > 0:
                src = torch.from_numpy(edges[0]).to(device)
                dst = torch.from_numpy(edges[1]).to(device)
                et = torch.from_numpy(etypes).to(device)
                bias[b, :, dst, src] = self.emb_edge(et).T
        return bias.reshape(B * H, l_max, l_max)

    def encode(
        self,
        cat: torch.Tensor,       # (B, L, 6) int64
        cont: torch.Tensor,      # (B, L, N_CONT) float
        pad_mask: torch.Tensor,  # (B, L) bool, True=pad
        attn_bias: torch.Tensor | None = None,  # (B*H, L, L) float
    ) -> torch.Tensor:
        B, L, _ = cat.shape
        e_pos = self.emb_pos(cat[..., 0])
        e_dep = self.emb_dep(cat[..., 1])
        e_gen = self.emb_gender(cat[..., 2])
        e_num = self.emb_number(cat[..., 3])
        e_pron = self.emb_prontype(cat[..., 4])
        e_ent = self.emb_ent(cat[..., 5])
        x = torch.cat([e_pos, e_dep, e_gen, e_num, e_pron, e_ent, cont], dim=-1)
        x = self.input_proj(x)

        if attn_bias is not None:
            out = x
            for layer in self.encoder.layers:
                out = layer(out, src_mask=attn_bias.to(x.dtype), src_key_padding_mask=pad_mask, is_causal=False)
        else:
            out = self.encoder(x, src_key_padding_mask=pad_mask)

        return out

    def forward(
        self,
        cat: torch.Tensor,
        cont: torch.Tensor,
        pad_mask: torch.Tensor,
        nominal_idx: torch.Tensor,   # (B, M) int64, padded with 0
        nom_mask: torch.Tensor,      # (B, M) bool, True=valid
        nom_lex: torch.Tensor,       # (B, M, 2) int64, [lemma_id, surface_id], padded with -1
        attn_bias: torch.Tensor | None = None,  # (B*H, L, L)
    ) -> torch.Tensor:
        B, M = nominal_idx.shape
        L = cat.shape[1]
        out = self.encode(cat, cont, pad_mask, attn_bias)

        gather_idx = nominal_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        x_nom = torch.gather(out, 1, gather_idx)  # (B, M, d_model)

        # Mention-ranking scores: for each mention i, score against all j < i (causal) + null
        # Returns (B, M, M+1): scores[b, i, j] = score of antecedent j for mention i; j=M is null
        x_i = x_nom.unsqueeze(2).expand(-1, -1, M, -1)   # (B, M, M, d)
        x_j = x_nom.unsqueeze(1).expand(-1, M, -1, -1)   # (B, M, M, d)

        # Pairwise features: recency (signed/abs token gap) + lexical match (lemma/surface)
        pos = nominal_idx.float()
        gap = (pos.unsqueeze(2) - pos.unsqueeze(1)) / max(L - 1, 1)  # (B, M, M), i - j
        lemma = nom_lex[..., 0]
        surf = nom_lex[..., 1]
        same_lemma = (lemma.unsqueeze(2) == lemma.unsqueeze(1)).to(gap.dtype)
        same_surface = (surf.unsqueeze(2) == surf.unsqueeze(1)).to(gap.dtype)
        pair_feats = torch.stack([gap, gap.abs(), same_lemma, same_surface], dim=-1)  # (B, M, M, 4)

        pair_repr = torch.cat([x_i, x_j, pair_feats], dim=-1)
        pair_scores = self.rank_head(pair_repr).squeeze(-1)  # (B, M, M)
        null_scores = self.null_score(x_nom)  # (B, M, 1)
        return torch.cat([pair_scores, null_scores], dim=-1)  # (B, M, M+1)


def mention_ranking_loss(
    scores: torch.Tensor,   # (B, M, M+1)
    gold_ante: torch.Tensor,  # (B, M, M+1) float, 1 where antecedent is gold (incl. null col)
    nom_mask: torch.Tensor,   # (B, M) bool
) -> torch.Tensor:
    B, M, _ = scores.shape
    # Causal mask: mention i can only attend to j < i or null (col M).
    # tril(-1) excludes the diagonal so a mention is never its own antecedent.
    causal = torch.ones(M, M + 1, dtype=torch.bool, device=scores.device).tril(-1)
    causal[:, M] = True  # null always allowed
    causal_mask = causal.unsqueeze(0)  # (1, M, M+1)

    # Mask out future antecedents and padding
    nom_valid = nom_mask.unsqueeze(2) & nom_mask.unsqueeze(1)  # (B, M, M)
    nom_valid_ext = torch.cat([nom_valid, nom_mask.unsqueeze(2)], dim=-1)  # (B, M, M+1)
    allowed = causal_mask & nom_valid_ext

    masked_scores = scores.masked_fill(~allowed, float("-inf"))
    log_probs = masked_scores - torch.logsumexp(masked_scores, dim=-1, keepdim=True)

    # Gold: at least null is always gold (no antecedent case)
    has_gold = (gold_ante * allowed.float()).sum(-1) > 0
    gold_ante_safe = gold_ante.clone()
    gold_ante_safe[~has_gold, M] = 1.0  # fallback to null

    # Marginal log-likelihood: log sum_gold exp(log_prob)
    gold_log_probs = log_probs + torch.log(gold_ante_safe.clamp(min=1e-9))
    mll = -torch.logsumexp(gold_log_probs.masked_fill(gold_ante_safe == 0, float("-inf")), dim=-1)

    valid_mentions = nom_mask & (masked_scores.max(-1).values > float("-inf"))
    if valid_mentions.sum() == 0:
        return scores.sum() * 0.0
    return mll[valid_mentions].mean()


def load_stage1(device: str = "cpu") -> tuple["DepGraphTransformer", spacy.Language]:
    model = DepGraphTransformer().to(device)
    ckpt = torch.load(MODELS_DIR / "stage1_graph_transformer.pt", map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
    model.eval()
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["senter"])
    return model, nlp


def _decode_clusters(
    nominal_positions: list[int],
    scores: np.ndarray,  # (M, M+1)
    null_margin: float = 0.0,
) -> list[list[int]]:
    M = len(nominal_positions)
    parent = list(range(M))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(M):
        masked = scores[i].copy()
        masked[i:M] = float("-inf")
        masked[M] -= null_margin
        ante = int(np.argmax(masked))
        if ante < M:
            parent[find(i)] = find(ante)

    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(find(i), []).append(nominal_positions[i])
    return [g for g in groups.values() if len(g) >= 2]


def resolve_stage1(
    sent: str | spacy.tokens.Span | spacy.tokens.Doc,
    model: "DepGraphTransformer",
    nlp: spacy.Language,
    device: str = "cpu",
    null_margin: float = 0.0,
) -> list[list[tuple[int, str]]]:
    # Stage 1 is intra-sentence: the caller supplies the sentence unit (matching
    # training, which uses dataset-provided boundaries). A raw string is parsed
    # and treated as a single sentence — no internal segmentation.
    if isinstance(sent, str):
        sent = nlp(sent)

    sent_len = len(sent)
    cat, cont, edges, etypes, nominal_positions, nom_lex = build_sentence_graph(sent, sent_len)
    if len(nominal_positions) < 2:
        return []

    cat_t = torch.from_numpy(cat).unsqueeze(0).to(device)
    cont_t = torch.from_numpy(cont).unsqueeze(0).to(device)
    pad_mask = torch.zeros(1, sent_len, dtype=torch.bool, device=device)
    nom_t = torch.tensor(nominal_positions, dtype=torch.long, device=device).unsqueeze(0)
    nom_mask = torch.ones(1, len(nominal_positions), dtype=torch.bool, device=device)
    nom_lex_t = torch.from_numpy(nom_lex).unsqueeze(0).to(device)
    with torch.no_grad():
        attn_bias = model._build_batch_attn_bias([(edges, etypes)], [sent_len], sent_len, device)
        scores = model(cat_t, cont_t, pad_mask, nom_t, nom_mask, nom_lex_t, attn_bias)
        scores = scores.squeeze(0).cpu().numpy()

    clusters = _decode_clusters(nominal_positions, scores, null_margin)
    return [[(idx, sent[idx].text) for idx in cluster] for cluster in clusters]


if __name__ == "__main__":
    import sys
    raw = sys.stdin.read().strip() if not sys.stdin.isatty() else (
        "Mary said she would call her sister when the meeting with the directors ended ."
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, n = load_stage1(dev)
    for cluster in resolve_stage1(raw, m, n, device=dev):
        print(f"  {[w for _, w in cluster]}")
