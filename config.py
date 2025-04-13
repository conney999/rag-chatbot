from pathlib import Path

DATA_DIR = Path(__file__).parent/"data"
RAW_DATA_402 = DATA_DIR / "raw/drdfs_expanded_402.xlsx"
RAW_DATA_28 = DATA_DIR / "raw/drdfs_original_28.xlsx"
PROCESSED_DATA_PATH = DATA_DIR / "processed/qa_pairs.json"


CACHED_DIR = "/scratch/cse592w25_class_root/cse592w25_class/connieli/hf_cache"

ALL_MINILM_MODEL_NAME = "all-MiniLM-L6-v2"
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

FLAN_T5_BASE_MODEL_NAME = "google/flan-t5-base"
T5_MODEL_NAME = "t5-base"
FLAN_T5_XL_MODEL_NAME ="google/flan-t5-xl"
GEMMA2_9B_MODEL_NAME = "google/gemma-2-9b-it"
GEMMA2_2B_MODEL_NAME = "google/gemma-2-2b-it"

TOP_N = 2
CONTEXT_JOIN_STR = " "

