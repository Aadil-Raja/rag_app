# main.py

from fastapi import FastAPI
from app.models import Query
from app.rag import generate_answer
import gradio as gr
from gradio.routes import mount_gradio_app
from app.embed_pdf import embed_uploaded_pdfs  # 🔁 new helper function
import tempfile
import shutil
import os
from app.vectorstore import clear_index 
app = FastAPI()

@app.post("/ask")
async def ask_question(query: Query):
    answer, top_chunks = generate_answer(query.question)
    return {"answer": answer, "top_chunks": top_chunks}

def rag_chat(query):
    answer, top_chunks = generate_answer(query)
    return f"📚 Context:\n{top_chunks}\n\n🤖 Answer:\n{answer}"

# ✅ New function to handle uploaded PDFs
def handle_upload(files):
    clear_index()  # Clear existing index before embedding new PDFs
    temp_dir = tempfile.mkdtemp()
    for file in files:
        # Extract filename from full path and copy to temp dir
        file_name = os.path.basename(file.name)
        shutil.copy(file.name, os.path.join(temp_dir, file_name))

    # Run embedding logic
    chunks_count = embed_uploaded_pdfs(temp_dir)
    shutil.rmtree(temp_dir)
    return f"✅ Indexed {chunks_count} chunks from {len(files)} uploaded PDF(s)."


upload_interface = gr.Interface(
    fn=handle_upload,
    inputs=gr.File(file_types=[".pdf"], label="Upload PDF(s)", file_count="multiple"),
    outputs="text",
    title="📄 Upload PDFs to Embed into RAG"
)

chat_interface = gr.Interface(
    fn=rag_chat,
    inputs="text",
    outputs="text",
    title="💬 Ask Questions to RAG"
)

# ✅ Combine both into one tabbed app
demo = gr.TabbedInterface([upload_interface, chat_interface], ["📁 Upload PDFs", "💡 Ask Questions"])
app = mount_gradio_app(app, demo, path="/gradio")