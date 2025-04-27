#emebed_pdf.py
import os
import PyPDF2
from app.vectorstore import add_to_index,clear_index

# 📌 Fixed-length character chunker
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

# ✅ Function to embed PDFs from a given directory
def embed_uploaded_pdfs(upload_dir):
    all_chunks = []
    clear_index()
    for file in os.listdir(upload_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(upload_dir, file)
            reader = PyPDF2.PdfReader(pdf_path)

            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text.replace("\n", " ") + " "

            # 🧠 Now chunk that giant string properly
            chunks = chunk_text(full_text, chunk_size=300, overlap=50)
            print(f"📄 {file} → {len(chunks)} chunks")
            all_chunks.extend(chunks)

    # ✅ Store in Chroma
    add_to_index(all_chunks)
    print(f"✅ Indexed {len(all_chunks)} chunks from PDFs")
    print(all_chunks[:3])  # Print first 3 chunks for verification
    return len(all_chunks)