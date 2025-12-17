from __future__ import annotations

import fitz
from PIL import Image


def render_page(pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
    """Render a single PDF page to PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

    doc.close()
    return img


def extract_text_layer(pdf_path: str, page_num: int) -> str | None:
    """Extract embedded text from PDF page (if any)."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    text = page.get_text("text").strip()
    doc.close()
    return text if text else None


def get_page_count(pdf_path: str) -> int:
    """Get the number of pages in a PDF."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count
