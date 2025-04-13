import sys
import json
import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

from config import PROCESSED_DATA_PATH
from models import (
    T5Model,
    FlanT5Model,
    FlanT5XLModel,
    AllMiniLM_Model,
    Bge_Model,
    Mpnet_Model,
    Gemma2_9B_Model,
    Gemma2_2B_Model,
)


with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)


def setup_models(embed_model_name, gen_model_name):
    if gen_model_name == 't5':
        model = T5Model()
    elif gen_model_name == 'flan':
        model = FlanT5Model()
    elif gen_model_name == 'flanxl':
        model = FlanT5XLModel()
    elif gen_model_name == 'gemma2-9b':
        model = Gemma2_9B_Model()
    elif gen_model_name == 'gemma2-2b':
        model = Gemma2_2B_Model()

    if embed_model_name == 'all-minilm':
        embedding_model = AllMiniLM_Model()
    elif embed_model_name == 'bge':
        embedding_model = Bge_Model()
    elif embed_model_name == 'mpnet':
        embedding_model = Mpnet_Model()

    return embedding_model, model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_model', choices=['all-minilm', 'bge', 'mpnet'], required=True)
    parser.add_argument('--gen_model', choices=['t5', 'flan', 'flanxl','gemma2-9b', 'gemma2-2b'], required=True)
    qa_group = parser.add_mutually_exclusive_group()
    qa_group.add_argument('--question_answer', action='store_true', dest='qa_mode')
    qa_group.add_argument('--answer', action='store_false', dest='qa_mode')
    args = parser.parse_args()

    gen_model_name = args.gen_model
    embed_model_name = args.embed_model
    use_qa_pairs = args.qa_mode
    
    embedding_model, model = setup_models(embed_model_name, gen_model_name)

    if use_qa_pairs:
        knowledge_base = [pair["question"] + " " + pair["answer"] for pair in qa_pairs] # approach 2
    else:
        knowledge_base = [pair["answer"] for pair in qa_pairs] # approach 1

    print(len(knowledge_base))


    kb_embeddings = embedding_model.encode(knowledge_base)
    # norms = np.linalg.norm(kb_embeddings, axis=1)
    # print("norms of kb embeddings:", norms)

    print("chatbot ('quit')")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'quit':
            break

        if embed_model_name == "bge":
            query_embedding = embedding_model.encode([user_query], True)[0]
        else:
            query_embedding = embedding_model.encode([user_query])[0]


        # print(np.linalg.norm(query_embedding))

        similarities = cosine_similarity([query_embedding], kb_embeddings)[0]
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