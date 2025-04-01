import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

client = chromadb.PersistentClient(path="chroma_db")
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en")

collection = client.get_or_create_collection(
    name="rag_docs",
    embedding_function=embedding_fn
)

def add_to_index(chunks):
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

def search(query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    print("Search results:", results)
    return results["documents"][0] if results["documents"] else ["[No results found]"]
