from fastapi import HTTPException, UploadFile
import os
import tempfile
import io
from pypdf import PdfReader
from pdf2image import convert_from_bytes

MAX_BYTES = 10 * 1024 * 1024  # 10MB limit

def _enforce_size_limit(file: UploadFile) -> bytes:
    """
    Read file content and enforce size limit (10MB).
    """
    buf = io.BytesIO()
    total = 0
    chunk_size = 1024 * 1024  # 1MB
    while True:
        chunk = file.file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large (> {MAX_BYTES//(1024*1024)} MB).")
        buf.write(chunk)
    file.file.seek(0)
    return buf.getvalue()

def _save_to_temp(content: bytes, suffix: str) -> str:
    """
    Save content to a temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name
    return temp_path

def _extract_pdf_text(pdf_path: str) -> tuple[str, int]:
    """
    Extract text from a PDF file using pypdf.
    Returns: (text, page_count)
    """
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            texts.append(text)
        return "\n".join(texts).strip(), num_pages
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}")

def _extract_pdf_images(pdf_bytes: bytes) -> list:
    """
    Extract images from a PDF file.
    Args:
        pdf_bytes: Bytes of the PDF file.
    Returns:
        List of (image_bytes, page_number) tuples.
    """
    try:
        images = convert_from_bytes(pdf_bytes, fmt="jpeg")
        image_data = []
        for i, image in enumerate(images):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")
            image_data.append((img_byte_arr.getvalue(), i + 1))
        return image_data
    except Exception as e:
        print(f"Error extracting images from PDF: {e}")
        return []