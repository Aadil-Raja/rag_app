from fastapi import FastAPI
from app.models import Query
from app.rag import generate_answer
import gradio as gr
from gradio.routes import mount_gradio_app  # NEW import

app = FastAPI()

@app.post("/ask")
async def ask_question(query: Query):
    answer, top_chunks = generate_answer(query.question)
    return {"answer": answer, "top_chunks": top_chunks}

# ✅ Define your Gradio function
def rag_chat(query):
    answer, top_chunks = generate_answer(query)
    return f"📚 Context:\n{top_chunks}\n\n🤖 Answer:\n{answer}"

# ✅ Create Gradio interface
gradio_interface = gr.Interface(fn=rag_chat, inputs="text", outputs="text")

# ✅ Mount Gradio to FastAPI (correct way)
app = mount_gradio_app(app, gradio_interface, path="/gradio")
