
#vectorstore.py
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

client = chromadb.PersistentClient(path="chroma_db")
#embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en")
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="intfloat/e5-base-v2"
)
collection = client.get_or_create_collection(
    name="rag_docs",
    embedding_function=embedding_fn
)
def add_to_index(chunks, metadatas):
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        metadatas=metadatas,  # 👈 attach metadata here
        ids=ids
    )


def search(query, top_k=3):
    query = f"query: {query}"
    results = collection.query(query_texts=[query], n_results=top_k)

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    return list(zip(documents, metadatas))  # Return both together

def clear_index():
    client.delete_collection("rag_docs")
    global collection
    collection = client.get_or_create_collection(
        name="rag_docs",
        embedding_function=embedding_fn
    )

# import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# client = chromadb.PersistentClient(path="chroma_db")
# embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en")

# collection = client.get_or_create_collection(
#     name="rag_docs",
#     embedding_function=embedding_fn
# )

# def add_to_index(chunks):
#     ids = [f"chunk-{i}" for i in range(len(chunks))]
#     collection.add(documents=chunks, ids=ids)

# def search(query, top_k=3):
#     results = collection.query(query_texts=[query], n_results=top_k)
#     print("Search results:", results)
#     return results["documents"][0] if results["documents"] else ["[No results found]"]