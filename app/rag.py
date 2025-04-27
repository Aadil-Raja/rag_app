#rag.py

from app.vectorstore import search
from app.llm import generate_response
from sentence_transformers import CrossEncoder

# Load re-ranker model once
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_chunks(question: str, chunks: list[str], top_n=3):
    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:top_n]]

def generate_answer(question: str):
    # Get more chunks from Chroma for better coverage
    retrieved_chunks = search(question, top_k=8)

    if not retrieved_chunks or retrieved_chunks == ["[No results found]"]:
        return "I couldn't find any relevant information in the documents.", ""

    # Re-rank using semantic similarity
    top_chunks = rerank_chunks(question, retrieved_chunks)

    # Combine for prompt
    context = "\n".join(top_chunks)
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
