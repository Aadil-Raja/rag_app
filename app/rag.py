from app.vectorstore import search
from app.llm import generate_response

def generate_answer(question: str):
    # ChromaDB handles embedding inside `search`
    relevant_chunks = search(question)
    context = "\n".join(relevant_chunks)

    prompt = f"""
You're an expert assistant. Use the context to answer the user's question as informatively as possible.
If the answer is not found in the context, say: "I don't know".

### Context:
{context}

### Question:
{question}

### Answer:
"""
    answer = generate_response(prompt)
    return answer, context
