import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DocumentChunk
from services.embedding import embedding_service


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    metadata: Dict[str, Any]
    score: float


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors in pure Python."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class VectorRetrievalService:
    """Executes vector similarity search using Gemini embeddings and tenant-isolated pgvector filtering."""

    @staticmethod
    async def search_similar_chunks(
        db: AsyncSession,
        user_id: uuid.UUID,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[uuid.UUID]] = None,
    ) -> List[SearchResult]:
        if not query or not query.strip():
            return []

        # 1. Generate query embedding
        query_embedding = embedding_service.generate_embedding(query)

        # Detect PostgreSQL vs SQLite environment
        dialect_name = db.bind.dialect.name if db.bind else "sqlite"
        is_postgresql = "postgresql" in dialect_name

        if is_postgresql:
            # Native PostgreSQL pgvector cosine distance: embedding <=> query_embedding
            distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
            stmt = select(
                DocumentChunk,
                distance_expr,
            ).where(DocumentChunk.user_id == user_id)

            if document_ids:
                stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

            stmt = stmt.order_by(distance_expr.asc()).limit(top_k)
            result = await db.execute(stmt)
            rows = result.all()

            search_results = []
            for chunk, distance in rows:
                score = 1.0 - float(distance) if distance is not None else 0.0
                search_results.append(
                    SearchResult(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        metadata=chunk.chunk_metadata or {},
                        score=round(score, 4),
                    )
                )
            return search_results

        else:
            # Fallback for SQLite / unit testing environment: compute in Python
            stmt = select(DocumentChunk).where(DocumentChunk.user_id == user_id)
            if document_ids:
                stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

            result = await db.execute(stmt)
            chunks = result.scalars().all()

            scored = []
            for chunk in chunks:
                if chunk.embedding:
                    sim = _cosine_similarity(query_embedding, chunk.embedding)
                    scored.append((chunk, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_matches = scored[:top_k]

            return [
                SearchResult(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    content=c.content,
                    metadata=c.chunk_metadata or {},
                    score=round(sim, 4),
                )
                for c, sim in top_matches
            ]


retrieval_service = VectorRetrievalService()
