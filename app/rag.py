from  embedder import embed_text
from vectorstore import search
from llm import generate_response

def generate_answer(question: str):
    query_vec = embed_text([question])[0]
    relevant_chunks = search(query_vec)
    context = "\n".join(relevant_chunks)

    prompt = f"""[INST] Answer the question based on the context below.
Context:
{context}

Question:
{question}
Answer:
[/INST]"""

    return generate_response(prompt)
