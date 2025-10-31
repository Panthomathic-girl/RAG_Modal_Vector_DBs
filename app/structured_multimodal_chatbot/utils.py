from fastapi import UploadFile, HTTPException
import io
from pypdf import PdfReader
import os
import tempfile

MAX_BYTES = 200 * 1024 * 1024  # 200 MB limit

def _enforce_size_limit(file: UploadFile, max_bytes: int = MAX_BYTES) -> bytes:
    """
    Read UploadFile stream safely and enforce a maximum size.
    Returns the file content as bytes.
    """
    buf = io.BytesIO()
    total = 0
    chunk_size = 1024 * 1024  # 1 MB

    while True:
        chunk = file.file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (> {max_bytes//(1024*1024)} MB).")
        buf.write(chunk)

    file.file.seek(0)
    return buf.getvalue()

def _extract_pdf_text(path: str) -> tuple[str, int]:
    reader = PdfReader(path)
    num_pages = len(reader.pages)
    texts = []
    for i in range(num_pages):
        try:
            page = reader.pages[i]
            texts.append(page.extract_text() or "")
        except Exception as e:
            texts.append("")
    return ("\n".join(texts).strip(), num_pages)

def _save_to_temp(content: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tmp.name, "wb") as f:
        f.write(content)
    return tmp.name