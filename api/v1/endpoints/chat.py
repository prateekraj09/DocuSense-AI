import json
import time
import uuid
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from core.database import get_db
from core.rate_limiter import limiter
from db.models import QueryLog, User
from schemas.chat import (
    ChatQueryRequest,
    ChatQueryResponse,
    CitationModel,
    RetrievedChunkInfo,
)
from services.cache import cache_service
from services.generation import generation_service
from services.retrieval import retrieval_service

router = APIRouter(prefix="/chat", tags=["Chat & Grounded Intelligence"])


@router.post(
    "/query",
    response_model=ChatQueryResponse,
    summary="Query private documents with caching and citations",
    description="Checks Redis cache first. If missed, runs hybrid search + Gemini generation, caches result, and logs audit metrics.",
)
@limiter.limit("30/minute")
async def query_chat(
    request: Request,
    query_in: ChatQueryRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChatQueryResponse:
    start_time = time.time()

    # Step 1: Redis Query Cache Check
    cache_key = cache_service.generate_query_cache_key(
        user_id=current_user.id,
        query=query_in.query,
        document_ids=query_in.document_ids,
    )
    cached_data = cache_service.get_cached_query(cache_key)

    if cached_data:
        latency_ms = int((time.time() - start_time) * 1000)
        # Log cache hit in PostgreSQL query_logs
        log_entry = QueryLog(
            id=uuid.uuid4(),
            user_id=current_user.id,
            query_text=query_in.query,
            response_text=cached_data["answer"],
            latency_ms=latency_ms,
            tokens_used=0,  # 0 tokens used on cache hit!
            cache_hit=True,
        )
        db.add(log_entry)
        await db.commit()

        cached_data["cache_hit"] = True
        cached_data["latency_ms"] = latency_ms
        return ChatQueryResponse(**cached_data)

    # Step 2: Hybrid Search (Vector + Full-Text via RRF)
    chunks = await retrieval_service.hybrid_search(
        db=db,
        user_id=current_user.id,
        query=query_in.query,
        top_k=query_in.top_k,
        document_ids=query_in.document_ids,
    )

    # Step 3: Grounded LLM Generation with strict citation guardrails
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

    # Step 4: Log query audit record into PostgreSQL
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

    # Step 5: Format retrieved chunks
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

    response_data = {
        "query": query_in.query,
        "answer": answer,
        "citations": [c.model_dump(mode="json") for c in citations],
        "retrieved_chunks": [ci.model_dump(mode="json") for ci in chunk_infos],
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
        "cache_hit": False,
    }

    # Step 6: Store in Redis Cache (TTL = 3600 seconds)
    cache_service.set_cached_query(cache_key, response_data, expire_seconds=3600)

    return ChatQueryResponse(
        query=query_in.query,
        answer=answer,
        citations=citations,
        retrieved_chunks=chunk_infos,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cache_hit=False,
    )


@router.post(
    "/stream",
    summary="Stream grounded answer tokens via Server-Sent Events (SSE)",
    description="Streams LLM tokens incrementally, followed by citations and completion events.",
)
@limiter.limit("20/minute")
async def stream_chat(
    request: Request,
    query_in: ChatQueryRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    # Step 1: Hybrid Search
    chunks = await retrieval_service.hybrid_search(
        db=db,
        user_id=current_user.id,
        query=query_in.query,
        top_k=query_in.top_k,
        document_ids=query_in.document_ids,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # Initial search metadata event
        meta_event = {
            "type": "retrieved_chunks",
            "count": len(chunks),
            "chunks": [
                {
                    "chunk_id": str(c.chunk_id),
                    "score": c.score,
                    "retrieval_type": c.retrieval_type,
                    "metadata": c.metadata,
                }
                for c in chunks
            ],
        }
        yield f"data: {json.dumps(meta_event)}\n\n"

        # Stream tokens from Gemini
        stream = generation_service.generate_grounded_answer_stream(
            query=query_in.query,
            retrieved_chunks=chunks,
        )
        for chunk in stream:
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
