import os
import uuid
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func

from db.models import Document, DocumentChunk, DocumentStatus
from services.chunker import DocumentChunker
from services.embedding import embedding_service
from services.parsers import parse_document

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


async def process_document_sync(
    db: AsyncSession,
    user_id: uuid.UUID,
    filename: str,
    content_bytes: bytes,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> Document:
    """Parses, chunks, embeds, and indexes a document synchronously inside PostgreSQL + pgvector."""
    if len(content_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f}MB.")

    ext = os.path.splitext(filename)[1].lower()

    # 1. Initialize Document record in PENDING / PROCESSING status
    doc = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename=filename,
        file_type=ext or "text",
        status=DocumentStatus.PROCESSING,
        error_message=None,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    try:
        # 2. Extract structured blocks with page / line metadata
        extracted_blocks = parse_document(filename=filename, content_bytes=content_bytes)
        if not extracted_blocks:
            raise ValueError("Document contains no readable text content.")

        # 3. Create overlapping text chunks
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = chunker.chunk_blocks(extracted_blocks)

        if not text_chunks:
            raise ValueError("No text chunks generated from document.")

        # 4. Generate 768-dimensional embeddings in batches
        chunk_texts = [c.content for c in text_chunks]
        embeddings = embedding_service.generate_embeddings_batch(chunk_texts)

        # 5. Save chunks into database
        is_postgresql = "postgresql" in str(db.bind.engine.url if db.bind else "")

        db_chunks = []
        for i, chunk in enumerate(text_chunks):
            embedding_val = embeddings[i] if i < len(embeddings) else None

            db_chunk = DocumentChunk(
                id=uuid.uuid4(),
                document_id=doc.id,
                user_id=user_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                chunk_metadata=chunk.metadata,
                embedding=embedding_val,
                search_vector=func.to_tsvector("english", chunk.content) if is_postgresql else None,
            )
            db_chunks.append(db_chunk)

        db.add_all(db_chunks)

        # 6. Mark document as COMPLETED
        doc.status = DocumentStatus.COMPLETED
        await db.commit()
        await db.refresh(doc)
        return doc

    except Exception as e:
        await db.rollback()
        # Update document status to FAILED with error message
        doc.status = DocumentStatus.FAILED
        doc.error_message = str(e)
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        raise
