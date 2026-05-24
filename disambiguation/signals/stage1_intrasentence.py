import numpy as np
import spacy
import torch
import torch.nn as nn

from disambiguation.signals.abstract_features import (
    DEP_IDS, ENT_TYPE_IDS, GENDER_IDS, NUMBER_IDS, POS_IDS, PRONTYPE_IDS,
)
from disambiguation.signals.train_full import _compute_depth

N_RAW_ATTRIBUTES = 32


def extract_raw_attributes(doc: spacy.tokens.Doc, sent_lens: list[int]) -> np.ndarray:
    n_tokens = sum(sent_lens)
    buf = np.zeros((n_tokens, N_RAW_ATTRIBUTES), dtype=np.float32)

    pos = 0
    abs_p = 0
    for sl in sent_lens:
        sent_start = abs_p
        for i in range(abs_p, abs_p + sl):
            tok = doc[i]
            morph = tok.morph.to_dict()
            ti = i - sent_start
            head_rel = tok.head.i - sent_start if tok.head != tok else -1

            # Column 0-2: Syntactic (POS, TAG, DEP)
            buf[pos, 0] = POS_IDS.get(tok.pos_, len(POS_IDS))
            buf[pos, 1] = DEP_IDS.get(tok.dep_, len(DEP_IDS))

            # Column 3-14: Morphological (12 attributes)
            buf[pos, 2] = GENDER_IDS.get(morph.get("Gender", "unknown"), 3)
            buf[pos, 3] = NUMBER_IDS.get(morph.get("Number", "unknown"), 2)
            buf[pos, 4] = int(morph.get("Person", "0")) if morph.get("Person") else 0
            buf[pos, 5] = PRONTYPE_IDS.get(morph.get("PronType", "unknown"), 5)
            # For Case/Mood/Tense/VerbForm/Aspect/Voice/Animacy/NumType: just binary presence
            buf[pos, 6] = float(bool(morph.get("Case")))
            buf[pos, 7] = float(bool(morph.get("Mood")))
            buf[pos, 8] = float(bool(morph.get("Tense")))
            buf[pos, 9] = float(bool(morph.get("VerbForm")))
            buf[pos, 10] = float(bool(morph.get("Aspect")))
            buf[pos, 11] = float(bool(morph.get("Voice")))
            buf[pos, 12] = float(bool(morph.get("Animacy")))
            buf[pos, 13] = float(bool(morph.get("NumType")))

            # Column 15-16: Entity (ent_type, ent_iob)
            buf[pos, 14] = ENT_TYPE_IDS.get(tok.ent_type_, len(ENT_TYPE_IDS))
            buf[pos, 15] = int(tok.ent_iob_ != "O")

            # Column 17-19: Head info (head.pos, head.dep, head.idx_relative)
            buf[pos, 16] = POS_IDS.get(tok.head.pos_, len(POS_IDS))
            buf[pos, 17] = DEP_IDS.get(tok.head.dep_, len(DEP_IDS))
            buf[pos, 18] = head_rel / max(sl - 1, 1) if sl > 1 else 0

            # Column 20-22: Tree (depth, n_lefts, n_rights)
            buf[pos, 19] = _compute_depth(tok) / max(20, 1)
            buf[pos, 20] = tok.n_lefts
            buf[pos, 21] = tok.n_rights

            # Column 23-24: Position (pos_in_sent, pos_in_doc)
            buf[pos, 22] = ti / max(sl - 1, 1) if sl > 1 else 0
            buf[pos, 23] = i / max(len(doc) - 1, 1) if len(doc) > 1 else 0

            # Column 25-28: Syntactic flags (is_subject, is_object, is_poss, is_stop)
            buf[pos, 24] = int(tok.dep_ in {"nsubj", "nsubj:pass", "nsubj:outer", "csubj"})
            buf[pos, 25] = int(tok.dep_ in {"obj", "iobj"})
            buf[pos, 26] = int(tok.dep_ == "nmod:poss")
            buf[pos, 27] = int(tok.is_stop)

            # Column 28: is_punct
            buf[pos, 28] = int(tok.pos_ == "PUNCT")

            # Column 29: is_title (title case)
            buf[pos, 29] = float(tok.is_title)

            # Column 30: is_digit (token is a numeral)
            buf[pos, 30] = float(tok.is_digit)

            # Column 31: is_alpha (purely alphabetic)
            buf[pos, 31] = float(tok.is_alpha)

            pos += 1
        abs_p += sl
    return buf




class MiniTransformer(nn.Module):
    def __init__(self, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.input_dim = 32
        self.n_heads = n_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=n_heads,
            dim_feedforward=64,
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pair_head = nn.Sequential(
            nn.Linear(32 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, combined_attrs: torch.Tensor) -> torch.Tensor:
        if combined_attrs.dim() == 2:
            combined_attrs = combined_attrs.unsqueeze(0)

        batch_size, seq_len, _ = combined_attrs.shape
        n_tokens = seq_len - 12

        x = self.attention(combined_attrs)

        x_real = x[:, :n_tokens, :]

        x_i = x_real.unsqueeze(2)
        x_j = x_real.unsqueeze(1)

        x_i_expanded = x_i.expand(-1, -1, n_tokens, -1)
        x_j_expanded = x_j.expand(-1, n_tokens, -1, -1)
        pair_repr = torch.cat([x_i_expanded, x_j_expanded], dim=-1)

        pair_repr_flat = pair_repr.reshape(-1, 32 * 2)
        pair_flat = self.pair_head(pair_repr_flat).squeeze(-1)

        pair_scores = pair_flat.reshape(batch_size, n_tokens, n_tokens)

        return pair_scores
