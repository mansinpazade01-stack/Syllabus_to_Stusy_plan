"""Step 1 - get raw text out of a PDF or a text file."""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

Source = Union[str, Path, bytes, bytearray, BinaryIO]


def extract_text(source: Source, filename: str = "") -> str:
    """Return the text of a syllabus.

    `source` can be a file path, raw bytes (e.g. Streamlit's `uploaded.getvalue()`),
    or a file-like object. PDFs are read with pdfplumber (PyPDF2 as fallback);
    everything else is decoded as plain text.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        data = source.read()
        filename = filename or getattr(source, "name", "")

    is_pdf = Path(filename).suffix.lower() == ".pdf" or data[:5] == b"%PDF-"
    text = _pdf_to_text(data) if is_pdf else _decode(data)
    if not text.strip():
        raise ValueError(
            "No text could be extracted. If this is a scanned PDF, it needs OCR "
            "(e.g. pytesseract) or you can paste the syllabus text instead."
        )
    return text


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _pdf_to_text(data: bytes) -> str:
    text = ""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        text = "\n".join(pages)
    except Exception:
        text = ""

    if len(text.strip()) < 30:           # pdfplumber failed or found nothing -> try PyPDF2 / pypdf
        for module_name in ("PyPDF2", "pypdf"):
            try:
                module = __import__(module_name)
                reader = module.PdfReader(io.BytesIO(data))
                alt = "\n".join((p.extract_text() or "") for p in reader.pages)
                if len(alt.strip()) > len(text.strip()):
                    text = alt
                    break
            except Exception:
                continue
    return text
