import os
from typing import List

from services.parsers.base import ExtractedBlock


class CodeAndTextParser:
    """Parses plain text, Markdown, and source code files with line-number metadata tracking."""

    @staticmethod
    def parse(
        content_bytes: bytes,
        filename: str = "source.txt",
        lines_per_block: int = 100,
    ) -> List[ExtractedBlock]:
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback decoding for legacy encoded files
            text = content_bytes.decode("latin-1", errors="replace")

        # Strip null bytes 0x00
        text = text.replace("\x00", "")

        lines = text.splitlines()
        if not lines:
            return []

        file_ext = os.path.splitext(filename)[1].lower()
        blocks: List[ExtractedBlock] = []

        # For small files (<= lines_per_block), keep as single block
        if len(lines) <= lines_per_block:
            blocks.append(
                ExtractedBlock(
                    text=text.strip(),
                    metadata={
                        "filename": filename,
                        "file_type": file_ext,
                        "start_line": 1,
                        "end_line": len(lines),
                        "total_lines": len(lines),
                    },
                )
            )
            return blocks

        # For larger code/text files, segment into logical blocks with line references
        for i in range(0, len(lines), lines_per_block):
            block_lines = lines[i : i + lines_per_block]
            block_text = "\n".join(block_lines).strip()
            if block_text:
                blocks.append(
                    ExtractedBlock(
                        text=block_text,
                        metadata={
                            "filename": filename,
                            "file_type": file_ext,
                            "start_line": i + 1,
                            "end_line": min(i + lines_per_block, len(lines)),
                            "total_lines": len(lines),
                        },
                    )
                )

        return blocks
