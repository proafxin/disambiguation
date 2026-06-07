from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = CACHE_DIR / "models"
TENSORBOARD_DIR = CACHE_DIR / "tensorboard"
SPACY_TRF_DIR = DATA_DIR / "spacy_trf"
NOM_CACHE = DATA_DIR / "stage2_conll_nominals_v4.pkl"
SPAN_CTX_CACHE = DATA_DIR / "stage2_span_ctx_v7"
# conll+preco@8k canonical config: one nominal cache + single-file ctx (no per-doc shards)
NOM_CACHE_CP8K = DATA_DIR / "stage2_nominals_cp8k_v1.pkl"
SPAN_CTX_CP8K = DATA_DIR / "stage2_span_ctx_cp8k_v1"
SCORER_PL = Path.home() / "Projects" / "reference-coreference-scorers" / "scorer.pl"
