from fastapi import FastAPI
from app.models import Query
from app.rag import generate_answer
import gradio as gr
from gradio.routes import mount_gradio_app
from app.embed_pdf import embed_uploaded_pdfs
import tempfile
import shutil
import os
from app.vectorstore import clear_index

app = FastAPI()

# ✅ FastAPI endpoint
@app.post("/ask")
async def ask_question(query: Query):
    result = generate_answer(query.question)
    return result

# ✅ Gradio RAG Chat
def rag_chat(query):
    result = generate_answer(query)

    answer = result["answer"]
    top_chunks = result["top_chunks"]
    context = result["context"]
    top_scores = result["top_scores"]
    sources = result.get("sources", [])
    retrieval_score = result.get("retrieval_score", 0)
    faithfulness_score = result.get("faithfulness_score", 0)

    # Build Answer Display
    answer_box = f"🤖 Final Answer:\n\n{answer}"

    # Build Top Chunks + Scores + Sources
    top_chunks_text = ""
    for idx, (chunk, score, source) in enumerate(zip(top_chunks, top_scores, sources), start=1):
        source_info = ""
        if isinstance(source, dict):
            source_info = " | ".join([f"{k}: {v}" for k, v in source.items()])
        
        top_chunks_text += f"Top {idx} (Score: {score:.4f})\n{chunk}\n(Source: {source_info})\n\n"

    # Build Context
    context_text = f"🧠 Final Context Sent to LLM:\n\n{context}"

    # Build Evaluation Metrics
    eval_text = (
        f"📈 Retrieval Score: {retrieval_score * 100:.2f}%\n"
        f"📈 Faithfulness Score: {faithfulness_score * 100:.2f}%"
    )

    return answer_box, top_chunks_text, context_text, eval_text

def handle_upload(files, pdf_type):
    clear_index()
    temp_dir = tempfile.mkdtemp()

    for file in files:
        file_name = os.path.basename(file.name)
        shutil.copy(file.name, os.path.join(temp_dir, file_name))

    chunks_count = embed_uploaded_pdfs(temp_dir, pdf_type)
    shutil.rmtree(temp_dir)
    return f"✅ Indexed {chunks_count} chunks from {len(files)} uploaded PDF(s) with '{pdf_type}' style."
upload_interface = gr.Interface(
    fn=handle_upload,
    inputs=[
        gr.File(file_types=[".pdf"], label="Upload PDF(s)", file_count="multiple"),
        gr.Dropdown(
            choices=["Auto Detect", "Q/A Style PDF", "Normal Paragraph PDF"],
            value="Auto Detect",
            label="Select PDF Type"
        )
    ],
    outputs="text",
    title="📁 Upload PDFs for RAG",
    description="Upload PDFs and specify their structure for best results."
)

chat_interface = gr.Interface(
    fn=rag_chat,
    inputs=gr.Textbox(label="Enter your question", placeholder="Ask about uploaded PDFs..."),
    outputs=[
        gr.Textbox(label="🤖 Final Answer"),
        gr.Textbox(label="📚 Top Retrieved Chunks (with Sources)"),
        gr.Textbox(label="🧠 Full Context Sent to LLM"),
        gr.Textbox(label="📈 Evaluation Metrics"),
    ],
    title="💬 Ask Questions to RAG",
    description="Ask anything and see retrieved chunks, sources, and evaluation scores."
)

# ✅ Mount Gradio App
demo = gr.TabbedInterface(
    [upload_interface, chat_interface],
    ["📁 Upload PDFs", "💬 Ask Questions"]
)

app = mount_gradio_app(app, demo, path="/gradio")
