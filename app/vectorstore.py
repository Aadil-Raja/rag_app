import faiss
import numpy as np
import os
import pickle

index = faiss.IndexFlatL2(384)
doc_chunks = []

def load_index():
    global index, doc_chunks
    if os.path.exists("index.faiss") and os.path.exists("chunks.pkl"):
        index = faiss.read_index("index.faiss")
        with open("chunks.pkl", "rb") as f:
            doc_chunks.extend(pickle.load(f))

def save_index():
    faiss.write_index(index, "index.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(doc_chunks, f)

def add_to_index(vectors, chunks):
    index.add(np.array(vectors).astype('float32'))
    doc_chunks.extend(chunks)
    save_index()

def search(query_vector, top_k=3):
    if index.ntotal == 0:
        return ["[No documents found in the index]"]
    
    D, I = index.search(np.array([query_vector]).astype('float32'), top_k)
    return [doc_chunks[i] for i in I[0] if i < len(doc_chunks)]
