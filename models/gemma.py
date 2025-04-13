from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import GEMMA2_9B

class Gemma2_9B_Model():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(GEMMA2_9B)
        self.model = AutoModelForCausalLM.from_pretrained(
            GEMMA2_9B,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )

    def generate_answer(self, context, user_query):
        content = (
            f"You are a helpful Q&A administrator for "
            f"Detroit Regional Dollars for Scholars (DRDFS), "
            f"a nonprofit that empowers students to succeed after high school. "
            f"Try your best to use the given information to concisely but completely answer the question "
            f"but if the information is not relevant to the question, acknowledge that then try your best to "
            f"answer in a concise, helpful manner for parents and students:\n\n"
            f"information: {context}\nquestion: {user_query}"
        )

        chat = [{ "role": "user", "content": content },]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        print(prompt)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=100, eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0])       