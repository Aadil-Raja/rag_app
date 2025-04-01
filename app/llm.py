from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "models/mistral-7b-instruct-v0.1.Q2_K.gguf",
    model_type="mistral",  # use 'mistral' here
    max_new_tokens=512,
    threads=6
)

def generate_response(prompt):
    return llm(prompt)
