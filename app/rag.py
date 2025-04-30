from app.vectorstore import search
from app.llm import generate_response
from sentence_transformers import CrossEncoder
from app.rag_evaluator import simple_retrieval_score, simple_faithfulness_score
from transformers import AutoTokenizer
from app import state 
# Load reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def trim_context_to_max_tokens(text: str, max_tokens: int = 512) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

def rerank_chunks(question: str, chunks: list[str], top_n=None):
    if top_n is None:
        top_n = state.current_topK_size_slider

    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    # Now each chunk has a score
    scored_chunks = list(zip(range(len(chunks)), chunks, scores))  
    scored_chunks = sorted(scored_chunks, key=lambda x: x[2], reverse=True)  

    top_indices = [idx for idx, _, _ in scored_chunks[:top_n]]
    top_chunks = [chunk for _, chunk, _ in scored_chunks[:top_n]]
    top_scores = [score for _, _, score in scored_chunks[:top_n]]

    return top_chunks, top_scores, top_indices



def generate_answer(question: str):
    print(state.current_topK_size_slider)
    print("\n==============================")
    print(f"❓ User's Question: {question}")
    print("==============================\n")

    # Step 1: Retrieve from vectorstore (documents + metadata)
    retrieved_results = search(question, top_k=15)

    if not retrieved_results or retrieved_results == ["[No results found]"]:
        print("⚠️ No relevant chunks retrieved!\n")
        return {
            "answer": "I couldn't find any relevant information.",
            "top_chunks": [],
            "context": "",
            "top_scores": [],
            "sources": [],
            "retrieval_score": 0,
            "faithfulness_score": 0
        }

    # Unpack documents and metadata
    retrieved_chunks, retrieved_metadata = zip(*retrieved_results)

    print(f"🔍 Retrieved {len(retrieved_chunks)} candidate chunks.\n")

    # Step 2: Rerank
    top_chunks, top_scores,top_indices = rerank_chunks(question, retrieved_chunks)
    trimmed_chunks = [trim_context_to_max_tokens(chunk, max_tokens=250) for chunk in top_chunks]

    print("🎯 Top Chunks After Re-ranking:\n")
    for idx, (chunk, score) in enumerate(zip(trimmed_chunks, top_scores), start=1):
        print(f"Top {idx} (Score: {score:.4f})\n{chunk}\n")

    # Step 3: Pick matching metadata
    top_metadatas = [retrieved_metadata[i] for i in top_indices]

    # Step 4: Build context
    context = "\n".join(trimmed_chunks)
    
    print(f"✂️ Trimmed Context Length (tokens): {len(tokenizer.encode(context))}\n")
    instructions=""
    # Step 5: Build prompt
    if state.current_pdf_type == "Resume/CV":
     instructions = """
You are a resume assistant.

Context is split by:
- name: person's name
- section type: e.g., Education, Skills
- details: the content

Answer using only this context. If the question mentions a name, use only that person's data.  
If not, find the best match. If info is missing, say: "Information not available."
"""

    elif state.current_pdf_type == "Q/A Style PDF":
       instructions = """You are an expert assistant answering based on a document that contains many question-answer pairs (Q:... A:... format).

1. If the user's query resembles or matches any question in the context, respond with the corresponding answer(s).
2. If the query is general or exploratory (e.g., "What kind of information is present in the document?"), summarize or list relevant Q/A pairs.
3. If no relevant information is found, say: "I don't know."

Use only the provided context. Do not guess or hallucinate outside it.
"""
    elif state.current_pdf_type == "PDF Slides":
       instructions = """You are an expert assistant helping interpret information from presentation slides.
Slides are short, bullet-style notes. Your job is to infer and expand on the key concepts in natural language.

Use the slide content below to answer the user's question.
If the slides do not provide enough information to answer, respond with: "The slides do not contain enough information."

Be accurate and detailed but avoid adding information not present or clearly implied from the context.
"""

    else:
        instructions = """You are an expert assistant. 
Use the given document context to answer the user's question as informatively as possible.
If the answer is not found, say: "I don't know."
"""
    print(instructions)
    prompt = f"""
{instructions}

Do NOT repeat the same sentence or phrase. Avoid verbose or filler words. Do not guess or make up details. Focus only on what is present in the text.

### Context:
{context}

### Question:
{question}

### Answer:
"""
    print(f"✂️ Trimmed Context Length (tokens): {len(tokenizer.encode(prompt))}\n")
    print("⚡ Prompt ready. Sending to LLM...\n")

    answer = generate_response(prompt)

    print("✅ Answer Generated!\n")

    # Step 6: Evaluate
    retrieval_score = simple_retrieval_score(question, top_chunks)
    faithfulness_score = simple_faithfulness_score(answer, top_chunks)

    print(f"📈 Retrieval Score: {retrieval_score:.4f}")
    print(f"📈 Faithfulness Score: {faithfulness_score:.4f}\n")

    # Step 7: Return
    return {
        "answer": answer,
        "top_chunks": top_chunks,
        "context": context,
        "top_scores": top_scores,
        "sources": top_metadatas,
        "retrieval_score": retrieval_score,
        "faithfulness_score": faithfulness_score
    }
