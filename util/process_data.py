import pandas as pd
import json
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_402, RAW_DATA_28, PROCESSED_DATA_PATH

qa_pairs = []
def process_file(file_path):
    df = pd.read_excel(file_path, header=None)
    for _, row in df.iterrows():
        if pd.isna(row[0]):
            continue
            
        question = str(row[0]).strip()
        answer_parts = [str(cell).strip() for cell in row[1:] if not pd.isna(cell)]
        answer = ' '.join(answer_parts)
        
        qa_pairs.append({
            "question": question,
            "answer": answer
        })


process_file(RAW_DATA_28)
process_file(RAW_DATA_402)

with open(PROCESSED_DATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
