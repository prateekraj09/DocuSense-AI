import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DocumentChunk
from services.embedding import embedding_service

# Standard Reciprocal Rank Fusion smoothing constant
RRF_K = 60


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_type: str = "hybrid"  # "vector", "keyword", or "hybrid"


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


class HybridRetrievalService:
    """Production single-store hybrid retrieval engine combining pgvector semantic search

    and PostgreSQL full-text search via Reciprocal Rank Fusion (RRF).
    """

    @classmethod
    async def search_vector_candidates(
        cls,
        db: AsyncSession,
        user_id: uuid.UUID,
        query_embedding: List[float],
        candidate_k: int = 20,
        document_ids: Optional[List[uuid.UUID]] = None,
        is_postgresql: bool = True,
    ) -> List[DocumentChunk]:
        """Retrieves top candidates ranked purely by dense vector cosine similarity."""
        if is_postgresql:
            if hasattr(DocumentChunk.embedding, "cosine_distance"):
                distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
            else:
                distance_expr = DocumentChunk.embedding.op("<=>")(query_embedding).label("distance")
            stmt = select(DocumentChunk).where(DocumentChunk.user_id == user_id)
            if document_ids:
                stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
            stmt = stmt.order_by(distance_expr.asc()).limit(candidate_k)
            res = await db.execute(stmt)
            return list(res.scalars().all())
        else:
            # SQLite fallback: calculate cosine similarity in memory
            stmt = select(DocumentChunk).where(DocumentChunk.user_id == user_id)
            if document_ids:
                stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
            res = await db.execute(stmt)
            chunks = list(res.scalars().all())
            scored = [
                (c, _cosine_similarity(query_embedding, c.embedding))
                for c in chunks
                if c.embedding
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in scored[:candidate_k]]

    @classmethod
    async def search_keyword_candidates(
        cls,
        db: AsyncSession,
        user_id: uuid.UUID,
        query: str,
        candidate_k: int = 20,
        document_ids: Optional[List[uuid.UUID]] = None,
        is_postgresql: bool = True,
    ) -> List[DocumentChunk]:
        """Retrieves top candidates ranked purely by full-text keyword matching (BM25 / ts_rank)."""
        if not query.strip():
            return []

        if is_postgresql:
            tsquery = func.plainto_tsquery("english", query)
            rank_expr = func.ts_rank(DocumentChunk.search_vector, tsquery).label("rank")
            stmt = (
                select(DocumentChunk)
                .where(
                    DocumentChunk.user_id == user_id,
                    DocumentChunk.search_vector.op("@@")(tsquery),
                )
            )
            if document_ids:
                stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
            stmt = stmt.order_by(rank_expr.desc()).limit(candidate_k)
            res = await db.execute(stmt)
            return list(res.scalars().all())
        else:
            # SQLite / Test fallback: regex and token frequency matching
            tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
            if not tokens:
                return []

            stmt = select(DocumentChunk).where(DocumentChunk.user_id == user_id)
            if document_ids:
                stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
            res = await db.execute(stmt)
            chunks = list(res.scalars().all())

            scored = []
            for c in chunks:
                content_lower = c.content.lower()
                token_matches = sum(content_lower.count(t) for t in tokens)
                if token_matches > 0:
                    scored.append((c, token_matches))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in scored[:candidate_k]]

    @classmethod
    async def hybrid_search(
        cls,
        db: AsyncSession,
        user_id: uuid.UUID,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[uuid.UUID]] = None,
        rrf_k: int = RRF_K,
    ) -> List[SearchResult]:
        """Executes Hybrid Search combining Vector Cosine Similarity and Keyword Full-Text Search

        merged via Reciprocal Rank Fusion (RRF).
        """
        if not query or not query.strip():
            return []

        # Detect database dialect
        dialect_name = db.bind.dialect.name if db.bind else "sqlite"
        is_postgresql = "postgresql" in dialect_name

        # 1. Generate dense vector embedding for query
        query_embedding = embedding_service.generate_embedding(query)

        # 2. Fetch top candidates from both retrieval channels
        vector_candidates = await cls.search_vector_candidates(
            db=db,
            user_id=user_id,
            query_embedding=query_embedding,
            candidate_k=top_k * 3,
            document_ids=document_ids,
            is_postgresql=is_postgresql,
        )

        keyword_candidates = await cls.search_keyword_candidates(
            db=db,
            user_id=user_id,
            query=query,
            candidate_k=top_k * 3,
            document_ids=document_ids,
            is_postgresql=is_postgresql,
        )

        # 3. Reciprocal Rank Fusion (RRF) algorithm
        # rrf_score(d) = sum( 1 / (rrf_k + rank(d)) )
        rrf_scores: Dict[uuid.UUID, float] = {}
        chunk_map: Dict[uuid.UUID, DocumentChunk] = {}
        retrieval_sources: Dict[uuid.UUID, set] = {}

        # Add vector ranks (1-indexed)
        for rank, chunk in enumerate(vector_candidates, start=1):
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (rrf_k + rank))
            retrieval_sources.setdefault(chunk.id, set()).add("vector")

        # Add keyword ranks (1-indexed)
        for rank, chunk in enumerate(keyword_candidates, start=1):
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (rrf_k + rank))
            retrieval_sources.setdefault(chunk.id, set()).add("keyword")

        if not rrf_scores:
            return []

        # Sort chunks by merged RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_ids = sorted_chunk_ids[:top_k]

        # Normalize score relative to theoretical maximum
        max_possible_rrf = (1.0 / (rrf_k + 1)) * 2  # Appears at rank 1 in both

        results: List[SearchResult] = []
        for cid in top_ids:
            chunk = chunk_map[cid]
            raw_score = rrf_scores[cid]
            norm_score = min(round(raw_score / max_possible_rrf, 4), 1.0)
            sources = retrieval_sources[cid]

            if "vector" in sources and "keyword" in sources:
                retrieval_type = "hybrid"
            elif "vector" in sources:
                retrieval_type = "vector"
            else:
                retrieval_type = "keyword"

            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata=chunk.chunk_metadata or {},
                    score=norm_score,
                    retrieval_type=retrieval_type,
                )
            )

        return results

    # Backward compatibility alias for Milestone 2
    search_similar_chunks = hybrid_search


retrieval_service = HybridRetrievalService()
