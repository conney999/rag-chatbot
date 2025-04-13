from transformers import T5ForConditionalGeneration, T5Tokenizer
from config import T5_MODEL_NAME, FLAN_T5_BASE_MODEL_NAME, FLAN_T5_XL_MODEL_NAME, CACHED_DIR



class BaseT5Model:
    def __init__(self, model_name):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=CACHED_DIR)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", cache_dir=CACHED_DIR)

    def generate_answer(self, input_text, generation_params=None):
        default_params = {
            'num_return_sequences': 1,
            'no_repeat_ngram_size': 2
        }
        if generation_params:
            default_params.update(generation_params)
            
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(input_ids, **default_params)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)



class T5Model(BaseT5Model):
    def __init__(self):
        super().__init__(T5_MODEL_NAME)


    def generate_answer(self, context, user_query):
        input_text = f"question: {user_query} context: {context}"
        generation_params = {
            'max_new_tokens': 300
        }
        return super().generate_answer(input_text, generation_params)



class FlanT5Model(BaseT5Model):
    def __init__(self, model_name=FLAN_T5_BASE_MODEL_NAME):
        super().__init__(model_name)


    def generate_answer(self, context, user_query):
        input_text = (
            f"You are a helpful Q&A assistant that answers questions using the provided context. "
            # f"if context is not relevant, say 'I don't have information on that' "
            f"context: {context} "
            f"question: {user_query} "
            f"answer:"
        )

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        # print("debug: input token length =", input_ids.shape[1])
        generation_params = {
            'min_length': 100,
            'max_length': 512
        }
        # max_new_tokens precedence over max_length, took out max_new_tokens for now
        return super().generate_answer(input_text, generation_params)


class FlanT5XLModel(FlanT5Model):
    def __init__(self):
        super().__init__(FLAN_T5_XL_MODEL_NAME)
