from pathlib import Path

import numpy as np
import spacy
import torch
import torch.nn as nn

from disambiguation.signals.abstract_features import (
    CAT_SPEC, CAT_CARDINALITIES, DEP_IDS, N_CAT, N_CONT_FULL, N_PAIR_SYNT,
    IDX_GENDER, IDX_NUMBER, IDX_PERSON, IDX_ENT_TYPE,
    MORPH_IDS,
)
from disambiguation.signals.train_full import _compute_depth

MODELS_DIR = Path(__file__).parent.parent.parent / "cache" / "models"
NOMINAL_POS = {"PRON", "NOUN", "PROPN"}

N_DEP = len(DEP_IDS) + 1  # dep arc types for the graph attention bias
EMB_DIM = 16              # per-categorical embedding width

# Clause-boundary dependency labels: walking up to one of these (or root)
# identifies the local clause a token belongs to.
CLAUSE_DEPS = {
    "ROOT", "ccomp", "xcomp", "advcl", "acl", "acl:relcl", "relcl",
    "csubj", "csubj:pass", "parataxis", "conj",
}


def build_sentence_graph(
    sent: spacy.tokens.Span | spacy.tokens.Doc,
    sent_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray]:
    cat = np.zeros((sent_len, N_CAT), dtype=np.int64)
    cont = np.zeros((sent_len, N_CONT_FULL), dtype=np.float32)
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_type: list[int] = []
    parent = np.arange(sent_len, dtype=np.int64)  # sentence-local head (self if root/external)
    deps: list[str] = []

    offset = sent[0].i

    for ti in range(sent_len):
        tok = sent[ti]
        morph = tok.morph.to_dict()
        deps.append(tok.dep_)

        # Every spaCy categorical attribute → its own column (fallback len(map)).
        for col, (name, m) in enumerate(CAT_SPEC):
            if name == "pos":
                v = tok.pos_
            elif name == "tag":
                v = tok.tag_
            elif name == "dep":
                v = tok.dep_
            elif name == "ent_type":
                v = tok.ent_type_
            elif name == "ent_iob":
                v = tok.ent_iob_
            elif name == "shape":
                v = tok.shape_
            else:
                v = morph.get(name)
            cat[ti, col] = m.get(v, len(m))

        cont[ti, 0] = float(tok.is_alpha)
        cont[ti, 1] = float(tok.is_digit)
        cont[ti, 2] = float(tok.is_title)
        cont[ti, 3] = float(tok.is_upper)
        cont[ti, 4] = float(tok.is_lower)
        cont[ti, 5] = float(tok.is_stop)
        cont[ti, 6] = float(tok.like_num)
        cont[ti, 7] = _compute_depth(tok) / 20.0
        cont[ti, 8] = tok.n_lefts / max(sent_len - 1, 1)
        cont[ti, 9] = tok.n_rights / max(sent_len - 1, 1)
        cont[ti, 10] = ti / max(sent_len - 1, 1)
        cont[ti, 11] = float(tok.is_sent_start or False)
        cont[ti, 12] = float(tok.is_bracket)
        cont[ti, 13] = float(tok.is_quote)

        head_ti = tok.head.i - offset
        if tok.head != tok and 0 <= head_ti < sent_len:
            parent[ti] = head_ti
            dep_id = DEP_IDS.get(tok.dep_, len(DEP_IDS))
            edge_src.append(ti)
            edge_dst.append(head_ti)
            edge_type.append(dep_id)
            edge_src.append(head_ti)
            edge_dst.append(ti)
            edge_type.append(dep_id + N_DEP)

    nominal_positions = [ti for ti in range(sent_len) if sent[ti].pos_ in NOMINAL_POS]

    # Per-nominal lexical fingerprints: spaCy integer hashes (tok.lemma, tok.lower).
    # Used only for equality (same_lemma / same_surface) in the pair head — never as
    # a magnitude or embedding index — so the hash value itself carries no signal.
    nom_lex = np.empty((len(nominal_positions), 2), dtype=np.int64)
    for k, ti in enumerate(nominal_positions):
        tok = sent[ti]
        # spaCy hashes are uint64; mask to int64 range for numpy.
        nom_lex[k, 0] = tok.lemma & 0x7FFFFFFFFFFFFFFF
        nom_lex[k, 1] = tok.lower & 0x7FFFFFFFFFFFFFFF

    pair_synt = _pair_syntactic_features(nominal_positions, parent, deps)
    edges = np.array([edge_src, edge_dst], dtype=np.int64) if edge_src else np.zeros((2, 0), dtype=np.int64)
    etypes = np.array(edge_type, dtype=np.int64) if edge_type else np.zeros(0, dtype=np.int64)
    return cat, cont, edges, etypes, nominal_positions, nom_lex, pair_synt


def _ancestors(ti: int, parent: np.ndarray) -> tuple[set[int], int]:
    chain: set[int] = set()
    cur = ti
    depth = 0
    while parent[cur] != cur and cur not in chain and depth <= len(parent):
        chain.add(cur)
        cur = int(parent[cur])
        depth += 1
    return chain | {cur}, depth


def _clause_head(ti: int, parent: np.ndarray, deps: list[str]) -> int:
    cur = ti
    seen: set[int] = set()
    while cur not in seen:
        if deps[cur] in CLAUSE_DEPS or parent[cur] == cur:
            return cur
        seen.add(cur)
        cur = int(parent[cur])
    return cur


def _pair_syntactic_features(positions: list[int], parent: np.ndarray, deps: list[str]) -> np.ndarray:
    # Per nominal pair (i, j): [same_clause, dominates, path_len_norm,
    #   arc_appos, arc_conj, arc_poss, arc_relcl, same_head].
    M = len(positions)
    feats = np.zeros((M, M, N_PAIR_SYNT), dtype=np.float32)
    if M == 0:
        return feats

    anc = [_ancestors(t, parent) for t in positions]
    clause = [_clause_head(t, parent, deps) for t in positions]
    norm = max(int(parent.shape[0]), 1)

    for a in range(M):
        ta = positions[a]
        anc_a, dep_a = anc[a]
        for b in range(M):
            tb = positions[b]
            anc_b, dep_b = anc[b]
            feats[a, b, 0] = float(clause[a] == clause[b])
            feats[a, b, 1] = float(tb in anc_a or ta in anc_b)
            common = anc_a & anc_b
            lca_depth = max((_ancestors(c, parent)[1] for c in common), default=0)
            feats[a, b, 2] = (dep_a + dep_b - 2 * lca_depth) / norm
            direct = parent[ta] == tb or parent[tb] == ta
            if direct:
                child_dep = deps[ta] if parent[ta] == tb else deps[tb]
                feats[a, b, 3] = float(child_dep == "appos")
                feats[a, b, 4] = float(child_dep == "conj")
                feats[a, b, 5] = float(child_dep in {"nmod:poss", "poss"})
                feats[a, b, 6] = float(child_dep in {"acl:relcl", "relcl"})
            feats[a, b, 7] = float(parent[ta] == parent[tb] and parent[ta] != ta)  # same syntactic head
    return feats


def _uf_find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _agreement_match(cat: torch.Tensor, nominal_idx: torch.Tensor, col: int, unknown_val: int) -> torch.Tensor:
    g = torch.gather(cat[..., col], 1, nominal_idx)  # (B, M)
    either_unknown = (g == unknown_val).unsqueeze(2) | (g == unknown_val).unsqueeze(1)
    return (either_unknown | (g.unsqueeze(2) == g.unsqueeze(1))).float()


class DepGraphTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # One embedding per spaCy categorical attribute (full attribute set).
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, EMB_DIM) for card in CAT_CARDINALITIES]
        )
        cat_dim = EMB_DIM * N_CAT
        self.input_proj = nn.Linear(cat_dim + N_CONT_FULL, d_model)

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

        # Mention-ranking head. Pair input = [repr_i, repr_j, signed_dist, abs_dist,
        #   same_lemma, same_surface, gender/number/person/ent match, pair_synt].
        self.n_pair_feats = 2 + 2 + 4 + N_PAIR_SYNT
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
        cat: torch.Tensor,       # (B, L, N_CAT) int64
        cont: torch.Tensor,      # (B, L, N_CONT_FULL) float
        pad_mask: torch.Tensor,  # (B, L) bool, True=pad
        attn_bias: torch.Tensor | None = None,  # (B*H, L, L) float
    ) -> torch.Tensor:
        embs = [emb(cat[..., i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(embs + [cont], dim=-1)
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
        pair_synt: torch.Tensor,     # (B, M, M, N_PAIR_SYNT) float, syntactic pair relations
        attn_bias: torch.Tensor | None = None,  # (B*H, L, L)
    ) -> torch.Tensor:
        B, M = nominal_idx.shape
        L = cat.shape[1]
        out = self.encode(cat, cont, pad_mask, attn_bias)

        gather_idx = nominal_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        x_nom = torch.gather(out, 1, gather_idx)  # (B, M, d_model)
        dtype = x_nom.dtype

        # Mention-ranking scores: for each mention i, score against all j < i (causal) + null
        # Returns (B, M, M+1): scores[b, i, j] = score of antecedent j for mention i; j=M is null
        x_i = x_nom.unsqueeze(2).expand(-1, -1, M, -1)   # (B, M, M, d)
        x_j = x_nom.unsqueeze(1).expand(-1, M, -1, -1)   # (B, M, M, d)

        # Recency (signed/abs token gap) + lexical match (lemma/surface)
        pos = nominal_idx.float()
        gap = (pos.unsqueeze(2) - pos.unsqueeze(1)) / max(L - 1, 1)  # (B, M, M), i - j
        same_lemma = (nom_lex[..., 0].unsqueeze(2) == nom_lex[..., 0].unsqueeze(1)).float()
        same_surface = (nom_lex[..., 1].unsqueeze(2) == nom_lex[..., 1].unsqueeze(1)).float()

        # Agreement matches: unknown value = len(map) → treat as wildcard (always matches).
        gender_unknown = len(MORPH_IDS["Gender"])      # 3
        number_unknown = len(MORPH_IDS["Number"])      # 2
        person_unknown = len(MORPH_IDS["Person"])      # 3

        pair_feats = torch.stack(
            [gap, gap.abs(), same_lemma, same_surface,
             _agreement_match(cat, nominal_idx, IDX_GENDER, gender_unknown),
             _agreement_match(cat, nominal_idx, IDX_NUMBER, number_unknown),
             _agreement_match(cat, nominal_idx, IDX_PERSON, person_unknown),
             _agreement_match(cat, nominal_idx, IDX_ENT_TYPE, len(CAT_SPEC[IDX_ENT_TYPE][1]))],
            dim=-1,
        )  # (B, M, M, 8)

        pair_repr = torch.cat([x_i, x_j, pair_feats.to(dtype), pair_synt.to(dtype)], dim=-1)
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
    for i in range(M):
        masked = scores[i].copy()
        masked[i:M] = float("-inf")
        masked[M] -= null_margin
        ante = int(np.argmax(masked))
        if ante < M:
            parent[_uf_find(parent, i)] = _uf_find(parent, ante)
    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(_uf_find(parent, i), []).append(nominal_positions[i])
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
    cat, cont, edges, etypes, nominal_positions, nom_lex, pair_synt = build_sentence_graph(sent, sent_len)
    if len(nominal_positions) < 2:
        return []

    cat_t = torch.from_numpy(cat).unsqueeze(0).to(device)
    cont_t = torch.from_numpy(cont).unsqueeze(0).to(device)
    pad_mask = torch.zeros(1, sent_len, dtype=torch.bool, device=device)
    nom_t = torch.tensor(nominal_positions, dtype=torch.long, device=device).unsqueeze(0)
    nom_mask = torch.ones(1, len(nominal_positions), dtype=torch.bool, device=device)
    nom_lex_t = torch.from_numpy(nom_lex).unsqueeze(0).to(device)
    pair_synt_t = torch.from_numpy(pair_synt).unsqueeze(0).to(device)
    with torch.no_grad():
        attn_bias = model._build_batch_attn_bias([(edges, etypes)], [sent_len], sent_len, device)
        scores = model(cat_t, cont_t, pad_mask, nom_t, nom_mask, nom_lex_t, pair_synt_t, attn_bias)
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
