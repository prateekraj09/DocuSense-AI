import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from core.database import get_db
from db.models import QueryLog, User
from schemas.chat import (
    ChatQueryRequest,
    ChatQueryResponse,
    CitationModel,
    RetrievedChunkInfo,
)
from services.generation import generation_service
from services.retrieval import retrieval_service

router = APIRouter(prefix="/chat", tags=["Chat & Grounded Intelligence"])


@router.post(
    "/query",
    response_model=ChatQueryResponse,
    summary="Query private documents with grounded citations",
    description="Executes single-store hybrid search (vector + full-text) and generates verifiable answers with source citations.",
)
async def query_chat(
    query_in: ChatQueryRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChatQueryResponse:
    start_time = time.time()

    # 1. Hybrid Search (Vector + Full-Text via RRF)
    chunks = await retrieval_service.hybrid_search(
        db=db,
        user_id=current_user.id,
        query=query_in.query,
        top_k=query_in.top_k,
        document_ids=query_in.document_ids,
    )

    # 2. Grounded LLM Generation with strict citation guardrails
    if not chunks:
        answer = "Insufficient information in uploaded documents to answer this question."
        latency_ms = int((time.time() - start_time) * 1000)
        citations = []
        tokens_used = 0
    else:
        try:
            gen_res = generation_service.generate_grounded_answer(
                query=query_in.query,
                retrieved_chunks=chunks,
            )
            answer = gen_res.answer
            citations = [
                CitationModel(source=c.source, location=c.location, snippet=c.snippet)
                for c in gen_res.citations
            ]
            tokens_used = gen_res.prompt_tokens + gen_res.completion_tokens
            latency_ms = int((time.time() - start_time) * 1000)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Grounded response generation failed: {str(e)}",
            )

    # 3. Log query audit record into PostgreSQL query_logs table
    log_entry = QueryLog(
        id=uuid.uuid4(),
        user_id=current_user.id,
        query_text=query_in.query,
        response_text=answer,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cache_hit=False,
    )
    db.add(log_entry)
    await db.commit()

    # 4. Format retrieved chunks for transparency
    chunk_infos = [
        RetrievedChunkInfo(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            content=c.content,
            score=c.score,
            retrieval_type=c.retrieval_type,
            metadata=c.metadata,
        )
        for c in chunks
    ]

    return ChatQueryResponse(
        query=query_in.query,
        answer=answer,
        citations=citations,
        retrieved_chunks=chunk_infos,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cache_hit=False,
    )
