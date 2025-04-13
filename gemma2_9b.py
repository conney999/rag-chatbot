from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers

def main():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        quantization_config=quantization_config,
        device_map="auto",
    )

    print("Gemma2 QA ('quit')")
    while True:
        input_text = input("\nYour question: ")
        if input_text.lower() == 'quit':
            break


        chat = [{ "role": "user", "content": input_text },]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        print(prompt)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=300,eos_token_id=tokenizer.eos_token_id, )
        print(tokenizer.decode(outputs[0]))



if __name__== "__main__":
    main()



