from dataclasses import dataclass, field
from typing import Any, Dict, List
from services.parsers.base import ExtractedBlock


@dataclass
class TextChunk:
    """Represents an overlapping text chunk with associated document metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0


class DocumentChunker:
    """Splits extracted document blocks into overlapping text chunks while preserving metadata."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_blocks(self, blocks: List[ExtractedBlock]) -> List[TextChunk]:
        all_chunks: List[TextChunk] = []
        global_index = 0

        for block in blocks:
            text = block.text.replace("\x00", "").strip()
            if not text:
                continue

            # If the block fits comfortably in a single chunk, do not split
            if len(text) <= self.chunk_size:
                chunk = TextChunk(
                    content=text,
                    metadata=dict(block.metadata),
                    chunk_index=global_index,
                )
                all_chunks.append(chunk)
                global_index += 1
                continue

            # Sliding window with natural boundary search
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + self.chunk_size, text_len)

                # If not at the end of the text, look backwards for a clean sentence or paragraph break
                if end < text_len:
                    # Look for paragraph break
                    break_point = text.rfind("\n\n", start, end)
                    if break_point != -1 and break_point > start + (self.chunk_size // 2):
                        end = break_point + 2
                    else:
                        # Look for sentence break (period, exclamation, question mark + space/newline)
                        for sep in [". ", ".\n", "? ", "! ", "\n"]:
                            break_point = text.rfind(sep, start, end)
                            if break_point != -1 and break_point > start + (self.chunk_size // 2):
                                end = break_point + len(sep)
                                break

                chunk_str = text[start:end].strip()
                if chunk_str:
                    chunk_meta = dict(block.metadata)
                    chunk_meta["chunk_char_start"] = start
                    chunk_meta["chunk_char_end"] = end

                    all_chunks.append(
                        TextChunk(
                            content=chunk_str,
                            metadata=chunk_meta,
                            chunk_index=global_index,
                        )
                    )
                    global_index += 1

                # Advance window by (chunk_size - chunk_overlap)
                step = end - start - self.chunk_overlap
                if step <= 0:
                    start = end
                else:
                    start += step

        return all_chunks
