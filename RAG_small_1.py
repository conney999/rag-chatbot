from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# --- 1. Prepare Your Knowledge Base ---

qa_pairs = [
    {"question": "If my student doesn’t go to one of the partner high schools, can they still join the program?", 
    "answer": "In order to join the Next Level Scholars program, a student must be enrolled at one of the DRDFS partner high schools."},
    {"question": "How can I get involved with DRDFS?",
    "answer": "There are a lot of ways to get involved with DRDFS! Fill out the form here to have a team member get in touch: https://drdfs.org/get-involved/volunteer/"}, 
    {"question": "Can I use my scholarship for room and board?",
    "answer": "No, it can only be used for tuition, fees, books and other direct educational expenses"},
    {"question": "Can I still use my scholarship even if I transfer schools?",
    "answer": "Yes! Just indicate that you have transferred when renewing your scholarship. As long as the school you have transferred to is accredited by the Department of Education, you can use your scholarship."},
    {"question": "What can I use my scholarship on?",
    "answer": "The scholarship can be used for tuition, fees, books and other direct educational expenses. It can’t be used for room/board."},
    {"question": "When will I receive my scholarship check?",
    "answer": "Once you complete your Award Agreement or Award Renewal form, your scholarship will be processed and mailed to your school about two weeks after the processing dates on our website."}, 
    {"question": "How will I know when my school has received my scholarship check?",
    "answer": "Reach out to your schools financial aid office or your financial aid adviser. Once the check has been mailed, it usually takes 2-3 weeks for the check to process and it be applied to your student account."},
    {"question": "How much is my scholarship?",
    "answer": "All Next Level Scholars who complete the high school program are awarded a $4,000 scholarship. Scholars decide how much they would like disbursed to their school based on completion of their Award Agreement."},
    {"question": "Can I change how much of my funds will be disbursed?",
    "answer": "Depending on your timeline to graduation, you may have your disbursement amount changed. Please reach out to Erin at etalbot@drdfs.org to chat further."},
    {"question": "How do I get my scholarship?",
    "answer": "You must complete an Award Agreement or Award Renewal form, that has been sent to your via text & email."},
    {"question": "Can I put my scholarship on hold?",
    "answer": "Yes, you can defer your scholarship for up to 27 months after high school graduation. Your scholarship must continue to be used each consecutive year after you start using it. If you are taking a semester off, you must reach out to Erin at etalbot@drdfs.org."}, 
    {"question": "Can I use my scholarship at a trade school?",
    "answer": "Yes, however, it must be accredited by an accrediting agency of the US Department of Education. You can just check if your school is accredited here: https://ope.ed.gov/dapip/#/home"}, 
    {"question": "Can I use my scholarship for cosmetology?",
    "answer": "Yes, however, it must be accredited by an accrediting agency of the US Department of Education. You can just check if your school is accredited here: https://ope.ed.gov/dapip/#/home"}, 
    {"question": "Can I use my scholarship for summer classes?",
    "answer": "Yes. Reach out to Erin at etalbot@drdfs.org to figure out details."},
    {"question": "Can I use my scholarship for grad school?",
    "answer": "No, right now you can only use your scholarship for undergrad."},
    {"question": "What do I do if I don’t need my scholarship?",
    "answer": "We’ll put your scholarship on hold (defer). If you don’t ever require the funds for tuition, then the funds will expire."},
    {"question": "Is my scholarship check mailed to me?",
    "answer": "No, your scholarship check is mailed directly to your school’s financial aid office."},
    {"question": "How do I decide which college is right for me?",
    "answer": "Consider factors like location, cost, available majors, campus culture, class size, and career services. Visit campuses if possible, talk to current students, and research online to find the best fit for your goals and preferences. Feel free to reach out to your coach if you’d like to explore different options."},
    {"question": "What factors should I consider when choosing a college?",
    "answer": "Important factors include academic programs, campus size, location, financial aid, extracurricular opportunities, and student support services."},
    {"question": "What is the difference between public and private colleges?",
    "answer": "Public colleges are funded by the state and often have lower tuition for in-state students. Private colleges rely on tuition and donations, tend to be smaller, and may offer more financial aid."},
    {"question": "Should I go to a community college before transferring to a four-year university?",
    "answer": "Community colleges can be a cost-effective way to start your education. Many offer transfer agreements with four-year universities, allowing students to complete general education requirements before transferring."},
    {"question": "What is early decision vs. early action?",
    "answer": "Early decision (ED) is binding—if accepted, you must attend. Early action (EA) is non-binding, allowing you to apply early and receive an earlier decision without commitment."},
    {"question": "How many colleges should I apply to?",
    "answer": "However many work for you! We recommend you aim for at least one reach school (more difficult to get into), match school (fit your academic profile), and safety school (you’re highly likely to be accepted). Your coach can help you decide which school are a reach, match, or safety for you."},
    {"question": "What is the FAFSA, and when should I fill it out?",
    "answer": "The FAFSA is a form that determines eligibility for federal financial aid. It normally opens on October 1st of each year, and you should submit it as soon as possible since some aid is first-come, first-served. The FAFSA is a DRDFS scholarship requirement that must be completed by March 1st to remain eligible."},
    {"question": "What is the difference between grants, scholarships, and loans?",
    "answer": "Grants and scholarships are free money you don’t have to repay. Loans must be paid back, usually with interest."},
    {"question": "What should I do if I’m undecided and don’t know what to major in?",
    "answer": "Take general education classes and explore different fields before declaring a major. Make sure to meet with your academic advisor to discuss your interests."},
    {"question": "How do I balance academics and social life?",
    "answer": "Find an organization system that works for you such as a planner, online calendar, sticky notes, etc. Be sure to set priorities and stick to a schedule. It’s great to get involved in clubs, but don’t overcommit. Time management is key!"},
    {"question": "How many credits should I take?",
    "answer": "Full time students are required to take 12 credits each semester. We generally recommend taking 14-16 credits each semester, but this can depend on your major and other responsibilities. You should talk to your academic advisor and/or coach to help set up a plan for each semester so you are not over working yourself."} 
    ]


# In this code, we try two difffernet approaches for preparing our knowledge_base for a RAG chatbot 

## Approach 1: Treat the Answers as Your Knowledge Base:

def Approach_1(qa_pairs):
    
    knowledge_base = [pair["answer"] for pair in qa_pairs]
    original_questions = [pair["question"] for pair in qa_pairs]

    return knowledge_base

## Approach 2: Treat Each Question-Answer Pair as a Document:
    
def Approach_2(qa_pairs):

    knowledge_base = [f"Question: {pair['question']} Answer: {pair['answer']}" for pair in qa_pairs]

    return knowledge_base


# --- 2. Initialize Embedding Model ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
# --- 3. Generate Embeddings for the Knowledge Base ---
# knowledge_base = Approach_1(qa_pairs)
knowledge_base = Approach_2(qa_pairs)

document_embeddings = embedding_model.encode(knowledge_base)
print(f"Number of document embeddings: {len(document_embeddings)}")
print(f"Shape of each embedding: {document_embeddings[0].shape}")


# --- 5. Chatbot Interaction Loop ---
while True:
    user_query = input("Ask a question (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break

    # --- 6. Generate Embedding for the User Query ---
    query_embedding = embedding_model.encode([user_query])[0]

    # --- 7. Retrieve Relevant Documents ---
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_n = 2  # Number of top documents to retrieve
    top_indices = np.argsort(similarities)[::-1][:top_n]

    context = ""
    for index in top_indices:
        context += knowledge_base[index] + " "

    print("\nRetrieved Context:")
    print(context)
    print("-" * 20)

    # --- 8. Prepare Input for LLM --- &  --- 9. Get Answer from LLM ---

    def answer_question_t5(model_name, question, context):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        input_text = f"question: {question} context: {context}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)  # Ensure on the same device

        output_ids = model.generate(input_ids, max_new_tokens=100, num_return_sequences=1, no_repeat_ngram_size=2)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer

    def answer_question_DistilBERT(model_name, question, context):

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding="max_length")
        # print("\nInputs to DistilBERT:")
        # print(inputs)

        outputs = model(**inputs)
        # print("\nDistilBERT Outputs:")
        # print(outputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        answer_start_index = torch.argmax(start_logits)
        answer_end_index = torch.argmax(end_logits)

        if answer_end_index >= answer_start_index:
            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        else:
            answer = "" # Or some other indicator that no valid answer was found

        return answer

    model_name = "t5-base"
    # model_name = "distilbert-base-uncased"
    answer = answer_question_t5(model_name, user_query, context)
    # answer = answer_question_DistiBERT(user_query, context)

    print(f"Answer: {answer}")
    print("\n")

