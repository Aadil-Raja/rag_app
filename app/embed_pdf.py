import os
import re
import PyPDF2
from app.vectorstore import add_to_index

def is_qa_style(text: str) -> bool:
    """Auto detect if PDF looks like a Q/A style based on number of Q: and A: patterns."""
    return text.count('Q:') > 5 and text.count('A:') > 5
def chunk_qa_style(text: str,qa_chunk_size=1) -> list:
    """Split text into Q/A chunks correctly even if line breaks exist."""
    # New smarter pattern
    pattern = r"(Q\d*:.*?A:.*?)(?=(?:Q\d*:)|$)"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    chunks = []
    for i in range(0, len(matches), qa_chunk_size):
        group = matches[i:i+qa_chunk_size]
        cleaned = " ".join(q.strip().replace("\n", " ") for q in group)
        if cleaned:
            chunks.append(cleaned)
    return chunks


def chunk_paragraph_style(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """Split normal text into paragraph chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

def chunk_resume_sections(text: str) -> list :
    
    pattern = r"(Education|Skills|Experience|Projects|Certifications|Awards|Languages)"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    chunks = []
    for idx in range(len(matches)):
        start = matches[idx].start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        if section_text:
            chunks.append(section_text)
    return chunks

def embed_uploaded_pdfs(upload_dir: str, pdf_type: str = "Auto Detect",qa_chunk_size=1) -> int:
    """Embed PDFs inside a folder by either Q/A splitting or paragraph splitting."""
    all_chunks = []
    all_metadatas = []

    for file in os.listdir(upload_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(upload_dir, file)
            reader = PyPDF2.PdfReader(pdf_path)

            full_text = ""
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += text + "\n"


            # ✨ Decide splitting method
            if pdf_type == "Auto Detect":
                if is_qa_style(full_text):
                    chunks = chunk_qa_style(full_text)
                else:
                    chunks = chunk_paragraph_style(full_text)
            elif pdf_type == "Q/A Style PDF":
                chunks = chunk_qa_style(full_text,qa_chunk_size)
            elif pdf_type == "Resume/CV":
                chunks= chunk_resume_sections(full_text)
            else:  # "Normal Paragraph PDF"
                chunks = chunk_paragraph_style(full_text)

            for idx, chunk in enumerate(chunks, start=1):
                print(f"Chunk {idx}: {chunk}\n")
            print(f"📄 {file} ➔ {len(chunks)} chunks generated.")

            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    "filename": file,
                    "page_number": 1  # Full document considered page 1
                })

    # ✅ Now properly pass both chunks and metadata
    add_to_index(all_chunks, all_metadatas)
    print(f"✅ Indexed {len(all_chunks)} chunks with metadata!")
    return len(all_chunks)
