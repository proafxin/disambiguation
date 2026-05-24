from pathlib import Path

import numpy as np
import spacy
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from disambiguation.signals.abstract_features import (
    DEP_IDS, ENT_TYPE_IDS, GENDER_IDS, NUMBER_IDS, POS_IDS, PRONTYPE_IDS,
)
from disambiguation.signals.train_full import _compute_depth

N_RAW_ATTRIBUTES = 32
MODELS_DIR = Path(__file__).parent.parent.parent / "cache" / "models"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
NOMINAL_POS = {"PRON", "NOUN", "PROPN"}


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

    def forward(
        self,
        combined_attrs: torch.Tensor,
        nominal_idx: torch.Tensor,
        attn_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if combined_attrs.dim() == 2:
            combined_attrs = combined_attrs.unsqueeze(0)
        if nominal_idx.dim() == 1:
            nominal_idx = nominal_idx.unsqueeze(0)

        batch_size, seq_len, _ = combined_attrs.shape
        n_tokens = seq_len - 12
        n_nom = nominal_idx.shape[1]

        x = self.attention(combined_attrs, src_key_padding_mask=attn_pad_mask)
        x_real = x[:, :n_tokens, :]

        gather_idx = nominal_idx.unsqueeze(-1).expand(-1, -1, x_real.shape[-1])
        x_nom = torch.gather(x_real, 1, gather_idx)

        x_i = x_nom.unsqueeze(2).expand(-1, -1, n_nom, -1)
        x_j = x_nom.unsqueeze(1).expand(-1, n_nom, -1, -1)
        pair_repr = torch.cat([x_i, x_j], dim=-1)

        pair_flat = self.pair_head(pair_repr.reshape(-1, 32 * 2)).squeeze(-1)
        return pair_flat.reshape(batch_size, n_nom, n_nom)


def load_stage1(device: str = "cpu") -> tuple[MiniTransformer, spacy.Language, SentenceTransformer]:
    model = MiniTransformer(n_heads=4, n_layers=1).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "stage1_mini_transformer.pt", map_location=device))
    model.eval()
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["senter", "lemmatizer"])
    bge = SentenceTransformer(BGE_MODEL)
    return model, nlp, bge


def _cluster_pairs(positions: list[int], scores: np.ndarray, threshold: float) -> list[list[int]]:
    n = len(positions)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if (scores[i, j] + scores[j, i]) / 2 >= threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(positions[i])
    return [g for g in groups.values() if len(g) >= 2]


def resolve_stage1(
    text: str,
    model: MiniTransformer,
    nlp: spacy.Language,
    bge: SentenceTransformer,
    threshold: float = 0.5,
    device: str = "cpu",
) -> list[list[list[tuple[int, str]]]]:
    doc = nlp(text)
    results = []

    for sent in doc.sents:
        sent_len = len(sent)
        nominal_positions = [i for i in range(sent_len) if sent[i].pos_ in NOMINAL_POS]
        if len(nominal_positions) < 2:
            results.append([])
            continue

        token_attrs = extract_raw_attributes(sent, [sent_len])
        emb = bge.encode([sent.text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        feature_matrix = np.vstack([token_attrs, emb.reshape(12, 32)])

        feature_t = torch.from_numpy(feature_matrix).float().unsqueeze(0).to(device)
        nom_t = torch.tensor(nominal_positions, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = torch.sigmoid(model(feature_t, nom_t).squeeze(0)).cpu().numpy()

        clusters = _cluster_pairs(nominal_positions, scores, threshold)
        results.append([[(idx, sent[idx].text) for idx in cluster] for cluster in clusters])

    return results


if __name__ == "__main__":
    import sys
    raw = sys.stdin.read().strip() if not sys.stdin.isatty() else (
        "Mary said she would call her sister when the meeting with the directors ended ."
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, n, b = load_stage1(dev)
    for sent_idx, clusters in enumerate(resolve_stage1(raw, m, n, b, device=dev)):
        if clusters:
            print(f"sentence {sent_idx}:")
            for cluster in clusters:
                print(f"  {[w for _, w in cluster]}")
