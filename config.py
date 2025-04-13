from pathlib import Path

DATA_DIR = Path(__file__).parent/"data"
RAW_DATA_402 = DATA_DIR / "raw/drdfs_expanded_402.xlsx"
RAW_DATA_28 = DATA_DIR / "raw/drdfs_original_28.xlsx"
PROCESSED_DATA_PATH = DATA_DIR / "processed/qa_pairs.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FLAN_T5_MODEL_NAME = "google/flan-t5-base"
T5_MODEL_NAME = "t5-base"
GEMMA2_9B = "google/gemma-2-9b-it"

TOP_N = 2
CONTEXT_JOIN_STR = " "

