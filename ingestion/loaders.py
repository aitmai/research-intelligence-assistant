"""
Loads raw text out of uploaded files (used by Mode A: opportunity briefs)
and cleans up raw EDGAR HTML filings into plain text (used by Mode B).
"""
from __future__ import annotations
import re
from pathlib import Path

from pypdf import PdfReader


def load_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_upload(path: str) -> str:
    """Dispatches to the right loader based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf_file(path)
    if suffix in (".txt", ".md"):
        return load_text_file(path)
    if suffix in (".htm", ".html"):
        return strip_html(load_text_file(path))
    raise ValueError(f"Unsupported file type: {suffix}")


def strip_html(raw_html: str) -> str:
    """Minimal HTML-to-text cleanup for raw EDGAR filing documents."""
    text = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
