from typing import List
import pymupdf

from services.parsers.base import ExtractedBlock


class PDFParser:
    """Extracts text and page-level metadata from PDF files using PyMuPDF."""

    @staticmethod
    def parse(content_bytes: bytes, filename: str = "document.pdf") -> List[ExtractedBlock]:
        blocks: List[ExtractedBlock] = []
        try:
            doc = pymupdf.open(stream=content_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Failed to open or parse PDF '{filename}': {str(e)}") from e

        total_pages = doc.page_count
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if text:
                blocks.append(
                    ExtractedBlock(
                        text=text,
                        metadata={
                            "page_number": page_num + 1,
                            "total_pages": total_pages,
                            "filename": filename,
                        },
                    )
                )

        doc.close()
        return blocks
