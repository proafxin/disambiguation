import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

CKPT_NAME = "stage2_global_coref.pt"
CKPT_B_NAME = "stage2_cluster_matcher.pt"
BACKBONE = "roberta-large"  # contextual encoder (1024-d); SpanBERT/spanbert-large-cased tested worse (frozen)
# tag appended to all on-disk artifacts so a different encoder never shares RoBERTa's cache;
# empty for the roberta-large default (backward-compatible with existing caches).
ENCODER_TAG = "" if BACKBONE == "roberta-large" else "_" + BACKBONE.split("/")[-1].split("-")[0]
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
        return F.normalize(out.last_hidden_state, p=2, dim=-1, eps=1e-4)  # eps>0 in fp16


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
    ctx /= counts.clamp(min=1).unsqueeze(1)  # uncovered positions stay 0 (avoid 0/0 -> NaN)
    return F.normalize(ctx, p=2, dim=-1, eps=1e-4)  # eps>0 in fp16: a zero (uncovered) row -> 0, not NaN


# ── Stage A ───────────────────────────────────────────────────────────────────


class AntecedentScorer(nn.Module):
    def __init__(
        self,
        proj_dim: int = 1024,
        hidden: int = 1024,
        dropout: float = 0.3,
        chunk: int = 8192,
        channel: str = "both",
        raw: bool = False,
        use_distance: bool = True,
    ):
        super().__init__()
        self.chunk = chunk
        self.channel = channel
        self.raw = raw  # raw=True skips P_ctx/P_bge and feeds the unprojected vectors to the ffnn
        self.use_distance = use_distance
        ctx_in, bge_in = 2 * CTX_DIM, BGE_DIM  # raw per-mention widths (ctx is start⊕end = 2048, bge 1024)
        if not raw:
            if channel in ("both", "ctx"):
                self.P_ctx = nn.Linear(ctx_in, proj_dim)
            if channel in ("both", "bge"):
                self.P_bge = nn.Linear(bge_in, proj_dim)
        if use_distance:
            self.register_buffer("dist_bounds", torch.tensor([2, 3, 4, 5, 8, 16, 32, 64]))
            self.dist_emb = nn.Embedding(len(self.dist_bounds) + 1, 32)
        ctx_g, bge_g = (ctx_in if raw else proj_dim), (bge_in if raw else proj_dim)  # per-mention rep width
        g = (ctx_g if channel in ("both", "ctx") else 0) + (bge_g if channel in ("both", "bge") else 0)
        self.ffnn = nn.Sequential(
            nn.Linear(2 * g + (32 if use_distance else 0), hidden),
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
        # (M, g); raw=True returns the unprojected [ctx|bge], else the learned projections
        parts = []
        if self.channel in ("both", "ctx"):
            c = self.drop(ctx.flatten(1))
            parts.append(c if self.raw else self.P_ctx(c))
        if self.channel in ("both", "bge"):
            b = self.drop(bge)
            parts.append(b if self.raw else self.P_bge(b))
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
            parts = [g[ii], g[jj]]
            if self.use_distance:
                bucket = torch.bucketize((ii - jj).clamp(min=0), self.dist_bounds, right=True)
                parts.append(self.dist_emb(bucket))
            feat = torch.cat(parts, dim=-1)
            scores[ii, jj] = self.ffnn(feat).squeeze(-1).to(scores.dtype)
        return scores, ante_mask


class MentionTransformer(nn.Module):
    # Stage A RELATIONAL head: a per-window Transformer over the window's mentions (full
    # self-attention = fully-connected GNN over mentions), then intra-window antecedent ranking.
    # AntecedentScorer scores each mention pair from FIXED per-mention reps (independent FFNN);
    # here every mention is contextualized by all other mentions in the window BEFORE scoring, so
    # the referential binding ("the company" <- "Microsoft") can surface relationally from the SAME
    # frozen reps the FFNN sees. Same (ctx, bge) -> (scores, ante) interface as AntecedentScorer, so
    # eval/decode are unchanged; training uses a padded per-window MLL (mention_transformer_loss).
    # This is the controlled head A/B that isolates head-vs-representation on the noun ceiling.
    # O(W^2) per window — the budget windowing already pays.
    def __init__(
        self,
        proj_dim: int = 512,
        hidden: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.3,
        channel: str = "both",
        use_distance: bool = True,
        max_positions: int = 256,
    ):
        super().__init__()
        self.channel = channel
        self.use_distance = use_distance
        ctx_in, bge_in = 2 * CTX_DIM, BGE_DIM
        if channel in ("both", "ctx"):
            self.P_ctx = nn.Linear(ctx_in, proj_dim)
        if channel in ("both", "bge"):
            self.P_bge = nn.Linear(bge_in, proj_dim)
        self.drop = nn.Dropout(dropout)
        g = (proj_dim if channel in ("both", "ctx") else 0) + (proj_dim if channel in ("both", "bge") else 0)
        self.node_in = nn.Linear(g, hidden)
        # Within-window mention-ORDER embedding: the self-attention is otherwise a permutation-
        # invariant bag (order only re-enters at the pair distance). Added to nodes before the
        # transformer so the relational mixing is order-aware (cf. ClusterGNN.win_emb).
        self.pos_emb = nn.Embedding(max_positions, hidden)
        if use_distance:
            self.register_buffer("dist_bounds", torch.tensor([2, 3, 4, 5, 8, 16, 32, 64]))
            self.dist_emb = nn.Embedding(len(self.dist_bounds) + 1, 32)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, n_layers)
        self.score = nn.Sequential(
            nn.Linear(4 * hidden + (32 if use_distance else 0), hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.null_bias = nn.Parameter(torch.zeros(1))

    def _mention_vecs(self, ctx: torch.Tensor, bge: torch.Tensor) -> torch.Tensor:
        # (..., 2, CTX_DIM), (..., BGE_DIM) -> (..., g); projects each channel and concatenates
        parts = []
        if self.channel in ("both", "ctx"):
            parts.append(self.P_ctx(self.drop(ctx.flatten(-2))))
        if self.channel in ("both", "bge"):
            parts.append(self.P_bge(self.drop(bge)))
        return torch.cat(parts, dim=-1)

    def _pair_score(self, hi: torch.Tensor, hj: torch.Tensor, bucket: torch.Tensor | None) -> torch.Tensor:
        # symmetric+order pair feature [hi | hj | hi*hj | |hi-hj| (| dist)] -> scalar score
        parts = [hi, hj, hi * hj, (hi - hj).abs()]
        if self.use_distance:
            parts.append(self.dist_emb(bucket))
        return self.score(torch.cat(parts, dim=-1)).squeeze(-1)

    def forward(self, ctx: torch.Tensor, bge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # single window: ctx (M,2,CTX), bge (M,BGE) -> scores (M,M), ante (M,M) [j<i]. Used at eval.
        M = ctx.shape[0]
        device = ctx.device
        idx = torch.arange(M, device=device)
        h0 = self.node_in(self._mention_vecs(ctx, bge))  # (M, hidden)
        h0 = h0 + self.pos_emb(idx.clamp(max=self.pos_emb.num_embeddings - 1))
        h = h0 + self.tf(h0.unsqueeze(0)).squeeze(0)  # outer residual (see ClusterGNN note)
        ante = idx.unsqueeze(0) < idx.unsqueeze(1)  # ante[i,j] = j<i
        hi = h.unsqueeze(1).expand(M, M, h.shape[-1])
        hj = h.unsqueeze(0).expand(M, M, h.shape[-1])
        bucket = (
            torch.bucketize((idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0), self.dist_bounds, right=True)
            if self.use_distance else None
        )
        return self._pair_score(hi, hj, bucket), ante

    def batched_scores(self, ctx: torch.Tensor, bge: torch.Tensor, pad: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        # padded training path: ctx (B,M,2,CTX), bge (B,M,BGE), pad (B,M) True=padding -> dense
        # scores (B,M,M). The transformer runs over the whole padded batch (cheap), but the pair
        # feature [hi|hj|hi*hj||hi-hj|(|dist)] is materialized only for the j<i pairs and scored in
        # chunks — the full dense (B,M,M,4H) tensor OOMs on a high-mention window. Padded positions
        # are zeroed post-attention (no NaN) and every pair touching one is masked by the MLL anyway.
        B, M = ctx.shape[0], ctx.shape[1]
        device = ctx.device
        idx = torch.arange(M, device=device)
        h0 = self.node_in(self._mention_vecs(ctx, bge))  # (B, M, hidden)
        h0 = h0 + self.pos_emb(idx.clamp(max=self.pos_emb.num_embeddings - 1)).unsqueeze(0)
        h = h0 + self.tf(h0, src_key_padding_mask=pad)  # (B, M, hidden)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        ai, aj = (idx.unsqueeze(1) > idx.unsqueeze(0)).nonzero(as_tuple=True)  # (P,) i>j within a window
        bb = torch.arange(B, device=device).repeat_interleave(ai.shape[0])
        ii, jj = ai.repeat(B), aj.repeat(B)  # (B*P,)
        scores = torch.zeros(B, M, M, device=device, dtype=h.dtype)
        for s0 in range(0, bb.shape[0], chunk):
            sl = slice(s0, s0 + chunk)
            b_, i_, j_ = bb[sl], ii[sl], jj[sl]
            bucket = torch.bucketize((i_ - j_).clamp(min=0), self.dist_bounds, right=True) if self.use_distance else None
            scores[b_, i_, j_] = self._pair_score(h[b_, i_], h[b_, j_], bucket).to(scores.dtype)
        return scores


class MentionDetector(nn.Module):
    # s2e/Maverick-style span detector over FROZEN window token reps (no gold mentions). Every
    # candidate span (i,j) with 0<=j-i<max_span gets one score:
    #   score(i,j) = w_s·f_s(x_i) + w_e·f_e(x_j) + f_s(x_i)ᵀ B f_e(x_j)
    # ~O(T·max_span), represents nested/overlapping mentions natively. The heavy negative imbalance
    # (~thousands of candidates per ~M gold mentions) is handled at the LOSS by sampling negatives
    # 1:1 with positives per window (see detector_loss) — no pos_weight.
    def __init__(self, proj: int = 512, dropout: float = 0.2, max_span: int = 30):
        super().__init__()
        self.max_span = max_span
        self.drop = nn.Dropout(dropout)
        self.f_start = nn.Sequential(nn.Linear(CTX_DIM, proj), nn.GELU(), nn.Dropout(dropout))
        self.f_end = nn.Sequential(nn.Linear(CTX_DIM, proj), nn.GELU(), nn.Dropout(dropout))
        self.start_score = nn.Linear(proj, 1)
        self.end_score = nn.Linear(proj, 1)
        self.bil = nn.Bilinear(proj, proj, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x (T, CTX) window token reps -> (start_idx, end_idx, logit) over valid candidate spans
        T = x.shape[0]
        s, e = self.f_start(self.drop(x)), self.f_end(self.drop(x))
        ss, es = self.start_score(s).squeeze(-1), self.end_score(e).squeeze(-1)
        i = torch.arange(T, device=x.device).unsqueeze(1)
        j = i + torch.arange(self.max_span, device=x.device).unsqueeze(0)  # (T, max_span)
        valid = j < T
        ii, jj = i.expand_as(j)[valid], j[valid]
        logit = ss[ii] + es[jj] + self.bil(s[ii], e[jj]).squeeze(-1)
        return ii, jj, logit


BIO_BACKBONE = "Jean-Baptiste/roberta-large-ner-english"  # NER-pretrained init for the mention detector


class BIOTagger(nn.Module):
    # RoBERTa token classifier for mention detection. L stacked B/I/O heads, one per containment depth:
    # head 0 tags flat/outermost mentions, head 1 the depth-1 nested ones, etc. CoNLL nesting is 100%
    # clean containment (0% crossing), so per-depth labels never conflict and each head is an
    # independent 3-way {O,B,I} classifier reading the raw last_hidden_state (not normalized).
    #
    # Default init is the NER-pretrained backbone (its features already localize entity spans), frozen,
    # with only the heads trained — the GDELT setup. n_trainable_layers>0 unfreezes the top N encoder
    # layers (needed if pronoun/common-noun recall lags, since the NER backbone was tuned for named
    # entities). A frozen backbone runs under no_grad (no activation/grad/optimizer-state cost — fits 8 GB).
    def __init__(
        self, n_layers: int = 3, dropout: float = 0.2, model_name: str = BIO_BACKBONE, n_trainable_layers: int = 0
    ):
        super().__init__()
        self.n_layers = n_layers
        self.roberta = AutoModel.from_pretrained(model_name)
        self.backbone_trainable = n_trainable_layers > 0
        for p in self.roberta.parameters():
            p.requires_grad_(False)
        if n_trainable_layers > 0:
            for layer in self.roberta.encoder.layer[-n_trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            self.roberta.gradient_checkpointing_enable()  # 8 GB: trade compute for activation memory
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(CTX_DIM, 3) for _ in range(n_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # (B, W) ids -> (L, B, W, 3) per-token class logits {O=0, B=1, I=2} for each depth head.
        # A fully frozen backbone runs under no_grad so its activations are freed immediately.
        if self.backbone_trainable:
            h = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            with torch.no_grad():
                h = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.drop(h)
        return torch.stack([head(h) for head in self.heads], dim=0)


class SpanDetector(nn.Module):
    # Full-FT RoBERTa + s2e/Maverick span scorer. Scores each candidate (start,end) span (length<max_span)
    # as a UNIT: score(i,j) = wₛ·fₛ(xᵢ) + wₑ·fₑ(xⱼ) + fₛ(xᵢ)ᵀ B fₑ(xⱼ). Predicting the exact extent IS the
    # objective (vs BIO's per-token membership, which clips long NPs). O(T·max_span) per window, no
    # span-pair comparison. Backbone fine-tuned (frozen features cap boundaries); n_trainable_layers as BIO.
    def __init__(self, model_name: str = BACKBONE, n_trainable_layers: int = 24, dropout: float = 0.2,
                 proj: int = 512, max_span: int = 30):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.backbone_trainable = n_trainable_layers > 0
        for p in self.roberta.parameters():
            p.requires_grad_(False)
        if n_trainable_layers > 0:
            for layer in self.roberta.encoder.layer[-n_trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            self.roberta.gradient_checkpointing_enable()
        self.max_span = max_span
        self.drop = nn.Dropout(dropout)
        self.f_start = nn.Sequential(nn.Linear(CTX_DIM, proj), nn.GELU(), nn.Dropout(dropout))
        self.f_end = nn.Sequential(nn.Linear(CTX_DIM, proj), nn.GELU(), nn.Dropout(dropout))
        self.start_score = nn.Linear(proj, 1)
        self.end_score = nn.Linear(proj, 1)
        # bilinear as a plain (proj,proj) weight scored manually as (s@B · e). nn.Bilinear materializes a
        # (n_candidates, proj, proj) intermediate (OOM at ~15k spans); this is just (n, proj).
        self.B = nn.Parameter(torch.empty(proj, proj))
        nn.init.xavier_uniform_(self.B)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.backbone_trainable:
            return self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        with torch.no_grad():
            return self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def score(self, h: torch.Tensor) -> tuple:
        # h (L, CTX) content-token reps of one window -> (start_idx, end_idx, logit) over candidate spans
        s, e = self.f_start(self.drop(h)), self.f_end(self.drop(h))
        ss, es = self.start_score(s).squeeze(-1), self.end_score(e).squeeze(-1)
        L = h.shape[0]
        i = torch.arange(L, device=h.device).unsqueeze(1)
        j = i + torch.arange(self.max_span, device=h.device).unsqueeze(0)
        valid = j < L
        ii, jj = i.expand_as(j)[valid], j[valid]
        logit = ss[ii] + es[jj] + ((s[ii] @ self.B) * e[jj]).sum(-1)
        return ii, jj, logit


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
        raw: bool = False,
        ctx_proj: int | None = None,
        bge_proj: int | None = None,
    ):
        super().__init__()
        self.channel = channel
        self.max_windows = max_windows
        self.member_pool = member_pool  # how a cluster's members collapse to one node vector
        self.use_lexical = use_lexical
        self.raw = raw  # raw=True skips P_ctx/P_bge; node_in projects the unprojected member vecs
        ctx_in, bge_in = 2 * CTX_DIM, BGE_DIM
        # per-channel projection widths; default to proj_dim. ctx_proj=1024 gives RoBERTa (the dominant,
        # most-compressed channel) more room without going fully raw.
        ctx_proj, bge_proj = ctx_proj or proj_dim, bge_proj or proj_dim
        if not raw:
            if channel in ("both", "ctx"):
                self.P_ctx = nn.Linear(ctx_in, ctx_proj)
            if channel in ("both", "bge"):
                self.P_bge = nn.Linear(bge_in, bge_proj)
        self.drop = nn.Dropout(dropout)
        ctx_g, bge_g = (ctx_in if raw else ctx_proj), (bge_in if raw else bge_proj)
        g = (ctx_g if channel in ("both", "ctx") else 0) + (bge_g if channel in ("both", "bge") else 0)
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
        # and non-linearly and it never couples through the shared null_bias.
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
        # (m, 2, CTX_DIM), (m, BGE_DIM) -> (m, g); raw=True keeps the unprojected vectors
        parts = []
        if self.channel in ("both", "ctx"):
            c = self.drop(ctx.flatten(1))
            parts.append(c if self.raw else self.P_ctx(c))
        if self.channel in ("both", "bge"):
            b = self.drop(bge)
            parts.append(b if self.raw else self.P_bge(b))
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
    # use_fast required: the word→subtoken mapping uses word_ids(), a fast-tokenizer API
    return AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
