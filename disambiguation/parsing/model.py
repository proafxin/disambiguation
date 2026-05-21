import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.linear(x)))


class Biaffine(nn.Module):
    def __init__(self, n_in: int, n_out: int, bias_x: bool = True, bias_y: bool = False) -> None:
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        n_x = n_in + int(bias_x)
        n_y = n_in + int(bias_y)
        self.U = nn.Parameter(torch.zeros(n_out, n_x, n_y))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # x: [B, W, n_in], y: [B, W, n_in]
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[..., :1])], dim=-1)
        # Equivalent to einsum("bxi,oij,byj->bxyo") using matmul
        # U: [n_out, n_x, n_y], x: [B, W, n_x], y: [B, W, n_y]
        n_out = self.U.shape[0]
        # x @ U -> [n_out, B, W, n_y]
        # Reshape U to [n_out * n_x, n_y], x to [B * W, n_x]
        b, w, _ = x.shape
        # For each output class, compute x @ U[o] @ y^T
        # U: [n_out, n_x, n_y] -> reshape to [n_out * n_x, n_y]
        # x: [B, W, n_x] @ U[o]: [n_x, n_y] -> [B, W, n_y] for each o
        # Then [B, W, n_y] @ y^T: [B, n_y, W] -> [B, W, W] for each o
        xu = torch.matmul(x, self.U.view(n_out, x.shape[-1], -1).permute(1, 0, 2).reshape(x.shape[-1], -1))
        # xu: [B, W, n_out * n_y]
        xu = xu.view(b, w, n_out, y.shape[-1])
        # xu: [B, W, n_out, n_y], y: [B, W, n_y] -> y: [B, 1, n_y, W]
        s = torch.matmul(xu, y.unsqueeze(1).transpose(-1, -2))
        # s: [B, W, n_out, W] -> [B, W, W, n_out]
        s = s.permute(0, 1, 3, 2)
        if n_out == 1:
            s = s.squeeze(-1)
        return s


class BiaffineParser(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_encoder_hidden: int = 768,
        n_arc_mlp: int = 500,
        n_rel_mlp: int = 100,
        n_pos_mlp: int = 256,
        mlp_dropout: float = 0.33,
        n_rels: int = 53,
        n_upos: int = 19,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.word_proj = nn.Linear(n_encoder_hidden, n_encoder_hidden, bias=False)
        self.arc_mlp_d = MLP(n_encoder_hidden, n_arc_mlp, mlp_dropout)
        self.arc_mlp_h = MLP(n_encoder_hidden, n_arc_mlp, mlp_dropout)
        self.rel_mlp_d = MLP(n_encoder_hidden, n_rel_mlp, mlp_dropout)
        self.rel_mlp_h = MLP(n_encoder_hidden, n_rel_mlp, mlp_dropout)
        self.pos_mlp = MLP(n_encoder_hidden, n_pos_mlp, mlp_dropout)
        self.pos_out = nn.Linear(n_pos_mlp, n_upos)
        self.arc_attn = Biaffine(n_arc_mlp, 1, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_rel_mlp, n_rels, bias_x=True, bias_y=True)

    def forward(self, subwords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # subwords: [B, W, fix_len] — word-level subword grid
        batch_size, n_words, fix_len = subwords.shape
        device = subwords.device

        word_reprs = []
        for b in range(batch_size):
            # Flatten grid to compact sequence, track word membership
            flat_ids = []
            word_indices = []
            subword_counts = []
            for w in range(n_words):
                count = 0
                for t in range(fix_len):
                    tok = subwords[b, w, t].item()
                    if tok != 0:
                        flat_ids.append(tok)
                        word_indices.append(w)
                        count += 1
                subword_counts.append(count)

            input_ids = torch.tensor([flat_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            enc_out = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state[0]

            # Average pool subword representations per word
            word_repr = torch.zeros(n_words, enc_out.shape[-1], dtype=enc_out.dtype, device=device)
            word_idx_tensor = torch.tensor(word_indices, dtype=torch.long, device=device)
            word_repr.scatter_add_(0, word_idx_tensor.unsqueeze(-1).expand_as(enc_out), enc_out)
            counts = torch.tensor(subword_counts, dtype=enc_out.dtype, device=device).unsqueeze(-1).clamp(min=1)
            word_repr = word_repr / counts
            word_reprs.append(word_repr)

        word_repr = torch.stack(word_reprs)
        word_repr = self.word_proj(word_repr)

        # Arc scores
        arc_d = self.arc_mlp_d(word_repr)
        arc_h = self.arc_mlp_h(word_repr)
        s_arc = self.arc_attn(arc_d, arc_h)

        # Rel scores
        rel_d = self.rel_mlp_d(word_repr)
        rel_h = self.rel_mlp_h(word_repr)
        s_rel = self.rel_attn(rel_d, rel_h)

        # POS scores
        s_pos = self.pos_out(self.pos_mlp(word_repr))

        return s_arc, s_rel, s_pos
