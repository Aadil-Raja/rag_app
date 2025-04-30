import os
import re
import PyPDF2
from app.vectorstore import add_to_index

def is_qa_style(text: str) -> bool:
    return text.count('Q:') > 5 and text.count('A:') > 5

def chunk_qa_style(text: str, qa_chunk_size=1) -> list:
    pattern = r"(Q\d*:.*?A:.*?)(?=(?:Q\d*:)|$)"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    chunks = []
    metadata = []
    for i in range(0, len(matches), qa_chunk_size):
        group = matches[i:i + qa_chunk_size]
        cleaned = " ".join(q.strip().replace("\n", " ") for q in group)
        if cleaned:
            chunks.append(cleaned)
            if qa_chunk_size == 1:
                label = f"question: Q{str(i + 1)}"
            else:
                q_start = i + 1
                q_end = min(i + qa_chunk_size, len(matches))
                label = f"questions: Q{q_start}-Q{q_end}"
            metadata.append(label)
    return list(zip(chunks, metadata))

def chunk_paragraph_style(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

def chunk_resume_sections(text: str) -> list:
    import re

    # Regex pattern to detect sections
    pattern = r"(Education|Skills|Experience|Projects|Certifications|Awards|Languages)"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    # Extract name from the top of the resume
    name_block = text[:matches[0].start()].strip() if matches else text.strip()
    name = name_block.splitlines()[0].strip() if name_block else "Unknown"

    chunks = []
    metadata = []

    # Add header section
    if name_block:
        header_details = f"name: {name}\nsection type: Header\ndetails: {name_block}"
        chunks.append(header_details)
        metadata.append("section: Header")

    # Add each labeled section with name and type
    for idx in range(len(matches)):
        start = matches[idx].start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            section_name = matches[idx].group(0).capitalize()
            labeled_chunk = f"name: {name}\nsection type: {section_name}\ndetails: {section_text}"
            chunks.append(labeled_chunk)
            metadata.append(f"section: {section_name}")

    return list(zip(chunks, metadata))

def chunk_slides(reader, slides_per_chunk=1) -> list:
    page_texts = []
    page_indices = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            page_texts.append(text.strip().replace("\n", " "))
            page_indices.append(i + 1)

    chunks = []
    metadata = []
    for i in range(0, len(page_texts), slides_per_chunk):
        chunk = " ".join(page_texts[i:i + slides_per_chunk])
        start_pg = page_indices[i]
        end_pg = page_indices[min(i + slides_per_chunk - 1, len(page_indices) - 1)]
        if start_pg == end_pg:
            label = f"page: {start_pg}"
        else:
            label = f"pages: {start_pg}-{end_pg}"
        chunks.append(chunk.strip())
        metadata.append(label)

    return list(zip(chunks, metadata))

def embed_uploaded_pdfs(upload_dir: str, pdf_type: str = "Auto Detect", qa_chunk_size=1, slide_chunk_size=1) -> int:
    all_chunks = []
    all_metadatas = []

    for file in os.listdir(upload_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(upload_dir, file)
            reader = PyPDF2.PdfReader(pdf_path)

            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

          
            if pdf_type == "Q/A Style PDF":
                qa_chunks = chunk_qa_style(full_text, qa_chunk_size)
                chunks, metadata = zip(*qa_chunks)
            elif pdf_type == "Resume/CV":
                resume_chunks = chunk_resume_sections(full_text)
                chunks, metadata = zip(*resume_chunks)
            elif pdf_type == "PDF Slides":
                slide_chunks = chunk_slides(reader, slide_chunk_size)
                chunks, metadata = zip(*slide_chunks)
            else:
                chunks = chunk_paragraph_style(full_text)
                metadata = ["page: 1"] * len(chunks)

            for idx, chunk in enumerate(chunks, start=1):
                print(f"Chunk {idx}: {chunk}\n")
            print(f"📄 {file} ➔ {len(chunks)} chunks generated.")

            for chunk, label in zip(chunks, metadata):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "filename": file,
                    "label": label
                })

    add_to_index(all_chunks, all_metadatas)
    print(f"✅ Indexed {len(all_chunks)} chunks with metadata!")
    return len(all_chunks)
