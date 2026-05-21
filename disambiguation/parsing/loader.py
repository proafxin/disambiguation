import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
from onnx import numpy_helper
from transformers import DebertaV2Model

from disambiguation.parsing.model import BiaffineParser

REPO_ID = "ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt"


def _extract_state_dict() -> dict[str, torch.Tensor]:
    onnx_path = hf_hub_download(repo_id=REPO_ID, filename="model.fp16.onnx")
    model = onnx.load(onnx_path)

    onnx_weights = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init).astype(np.float32)
        onnx_weights[init.name] = torch.from_numpy(arr)

    matmul_map = {}
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            for out in node.output:
                for inp in node.input:
                    if "onnx::MatMul" in inp:
                        path = out.replace("/MatMul_output_0", "").replace("/MatMul_", "").lstrip("/")
                        path = path.replace("/", ".")
                        matmul_map[inp] = path

    state_dict = {}
    for name, tensor in onnx_weights.items():
        if name.startswith("m."):
            state_dict[name[2:]] = tensor
        elif "onnx::MatMul" in name and name in matmul_map:
            state_dict[matmul_map[name] + ".weight"] = tensor.T

    state_dict["word_proj.weight"] = state_dict.pop("graph_output_cast_2.weight")
    return state_dict


def load_parser(device: str = "cuda", dtype: torch.dtype = torch.float16) -> BiaffineParser:
    encoder = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
    parser = BiaffineParser(encoder=encoder)
    state_dict = _extract_state_dict()
    parser.load_state_dict(state_dict)
    parser.eval()
    return parser.to(device=device, dtype=dtype)
