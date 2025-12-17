from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PDFFile:
    path: Path
    sha256: str
    size_bytes: int


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def discover_pdfs(root: Path) -> list[PDFFile]:
    """
    Walk directory recursively and find all PDF files.

    Returns list of PDFFile with path and hash.
    """
    pdfs = []

    for pdf_path in root.rglob("*.pdf"):
        if pdf_path.is_file():
            sha256 = compute_sha256(pdf_path)
            pdfs.append(PDFFile(
                path=pdf_path,
                sha256=sha256,
                size_bytes=pdf_path.stat().st_size,
            ))

    return pdfs


def filter_unchanged(
    pdfs: list[PDFFile],
    existing_hashes: set[str],
) -> list[PDFFile]:
    """Filter out PDFs that have already been indexed (by hash)."""
    return [pdf for pdf in pdfs if pdf.sha256 not in existing_hashes]
