import os
import PyPDF2
from vectorstore import add_to_index

# 📌 Fixed-length character chunker
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

data_dir = "data"
all_chunks = []

for file in os.listdir(data_dir):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(data_dir, file)
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
