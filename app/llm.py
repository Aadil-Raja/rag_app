#llm.py
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "models/mistral-7b-instruct-v0.1.Q2_K.gguf",
    model_type="mistral",
    max_new_tokens=512,
    context_length=1024,  # Explicitly set a higher context length
    threads=6
)


def generate_response(prompt):
    return llm(prompt)
