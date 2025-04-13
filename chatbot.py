from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models import T5Model, FlanT5Model, EmbeddingModel, Gemma2_9B_Model
import json
import sys
from config import PROCESSED_DATA_PATH

with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)


def main():
    if sys.argv[1] not in ['t5', 'flan_t5', 'gemma2_9b']:
        print("wrong model")
        sys.exit(1)

    model_type = sys.argv[1]
    if model_type == 't5':
        model = T5Model()
    elif model_type == 'flan_t5':
        model = FlanT5Model()
    elif model_type == 'gemma2_9b':
        model = Gemma2_9B_Model()

    embedding_model = EmbeddingModel()

    # knowledge_base = [pair["answer"] for pair in qa_pairs] # approach 1
    knowledge_base = [pair["question"] + " " + pair["answer"] for pair in qa_pairs] # approach 2
    answer_embeddings = embedding_model.encode(knowledge_base)

    print("RAG QA chatbot ('quit')")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'quit':
            break

        query_embedding = embedding_model.encode([user_query])[0]
        similarities = cosine_similarity([query_embedding], answer_embeddings)[0]
        top_n = 2
        top_indices = np.argsort(similarities)[::-1][:top_n]

        context = ""
        for index in top_indices:
            context += knowledge_base[index] + " "

        print("\nRetrieved Context:")
        print(context)
        print("-" * 20)

        answer = model.generate_answer(context, user_query)
        print("\nGenerated Answer:")
        print(answer)
        print("-" * 20)


if __name__== "__main__":
    main()