# This is the updated main.py with proper handling for PDF Slides type
from fastapi import FastAPI
from app.models import Query
from app.rag import generate_answer
import gradio as gr
from gradio.routes import mount_gradio_app
from app.embed_pdf import embed_uploaded_pdfs
import tempfile
import shutil
from app import state
import os
from app.vectorstore import clear_index

app = FastAPI()

@app.post("/ask")
async def ask_question(query: Query):
    result = generate_answer(query.question)
    return result

def rag_chat(query):
    result = generate_answer(query)

    answer = result["answer"]
    top_chunks = result["top_chunks"]
    context = result["context"]
    top_scores = result["top_scores"]
    sources = result.get("sources", [])
    retrieval_score = result.get("retrieval_score", 0)
    faithfulness_score = result.get("faithfulness_score", 0)

    answer_box = f"🤖 Final Answer:\n\n{answer}"

    top_chunks_text = ""
    for idx, (chunk, score, source) in enumerate(zip(top_chunks, top_scores, sources), start=1):
        source_info = ""
        if isinstance(source, dict):
            source_info = " | ".join([f"{k}: {v}" for k, v in source.items()])
        top_chunks_text += f"Top {idx} (Score: {score:.4f})\n{chunk}\n(Source: {source_info})\n\n"

    context_text = f"🧠 Final Context Sent to LLM:\n\n{context}"

    eval_text = (
        f"📈 Retrieval Score: {retrieval_score * 100:.2f}%\n"
        f"📈 Faithfulness Score: {faithfulness_score * 100:.2f}%"
    )

    return answer_box, top_chunks_text, context_text, eval_text

def handle_upload(files, pdf_type, qa_chunk_size, slide_chunk_size):
    state.current_pdf_type = pdf_type
    clear_index()
    temp_dir = tempfile.mkdtemp()

    for file in files:
        file_name = os.path.basename(file.name)
        shutil.copy(file.name, os.path.join(temp_dir, file_name))

    chunks_count = embed_uploaded_pdfs(
        temp_dir,
        pdf_type=pdf_type,
        qa_chunk_size=int(qa_chunk_size),
        slide_chunk_size=int(slide_chunk_size)
    )
    shutil.rmtree(temp_dir)
    return f"✅ Indexed {chunks_count} chunks from {len(files)} uploaded PDF(s) with '{pdf_type}' style."

with gr.Blocks() as demo:
    with gr.Tab("📁 Upload PDFs"):
        pdf_upload = gr.File(file_types=[".pdf"], label="Upload PDF(s)", file_count="multiple")
        pdf_type_dropdown = gr.Dropdown(
            choices=["Auto Detect", "Q/A Style PDF", "Normal Paragraph PDF", "Resume/CV", "PDF Slides"],
            value="Auto Detect",
            label="Select PDF Type"
        )
        qa_chunk_size_slider = gr.Slider(
            minimum=1, maximum=10, step=1, value=2,
            label="Questions per Chunk",
            visible=False
        )
        slide_chunk_size_slider = gr.Slider(
            minimum=1, maximum=4, step=1, value=1,
            label="Slides per Chunk",
            visible=False
        )
        upload_button = gr.Button("Upload and Embed")
        output_text = gr.Textbox(label="Upload Output")

        def update_chunk_visibility(pdf_type):
            return (
                gr.update(visible=(pdf_type == "Q/A Style PDF")),
                gr.update(visible=(pdf_type == "PDF Slides"))
            )

        pdf_type_dropdown.change(
            update_chunk_visibility,
            inputs=[pdf_type_dropdown],
            outputs=[qa_chunk_size_slider, slide_chunk_size_slider]
        )

        upload_button.click(
            handle_upload,
            inputs=[pdf_upload, pdf_type_dropdown, qa_chunk_size_slider, slide_chunk_size_slider],
            outputs=[output_text]
        )

    with gr.Tab("💬 Ask Questions"):
        chat_input = gr.Textbox(label="Enter your question", placeholder="Ask about uploaded PDFs...")
        answer_box = gr.Textbox(label="🤖 Final Answer")
        top_chunks_box = gr.Textbox(label="📋 Top Retrieved Chunks (with Sources)")
        context_box = gr.Textbox(label="🧠 Full Context Sent to LLM")
        eval_box = gr.Textbox(label="📈 Evaluation Metrics")

        ask_button = gr.Button("Ask")
        ask_button.click(
            rag_chat,
            inputs=[chat_input],
            outputs=[answer_box, top_chunks_box, context_box, eval_box]
        )

app = mount_gradio_app(app, demo, path="/gradio")
