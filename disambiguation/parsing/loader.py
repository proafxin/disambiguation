from pathlib import Path

import torch
from transformers import DebertaV2Model

from disambiguation.parsing.model import BiaffineParser

WEIGHTS_PATH = Path(__file__).parent.parent.parent / "models" / "biaffine_parser.pt"


def load_parser(device: str = "cuda", dtype: torch.dtype = torch.float16) -> BiaffineParser:
    encoder = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
    parser = BiaffineParser(encoder=encoder)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    parser.load_state_dict(state_dict)
    parser.eval()
    return parser.to(device=device, dtype=dtype)
