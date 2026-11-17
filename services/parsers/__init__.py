import os
from typing import List

from services.parsers.base import ExtractedBlock
from services.parsers.code_parser import CodeAndTextParser
from services.parsers.pdf_parser import PDFParser

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cpp",
    ".c",
    ".go",
    ".rs",
    ".json",
    ".yaml",
    ".yml",
    ".sql",
    ".html",
    ".css",
}


def parse_document(filename: str, content_bytes: bytes) -> List[ExtractedBlock]:
    """Dispatches the document bytes to the appropriate parser based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        ext = ".txt"

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        return PDFParser.parse(content_bytes, filename=filename)
    else:
        return CodeAndTextParser.parse(content_bytes, filename=filename)


__all__ = ["ExtractedBlock", "PDFParser", "CodeAndTextParser", "parse_document", "ALLOWED_EXTENSIONS"]
