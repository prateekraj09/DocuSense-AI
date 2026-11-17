import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CitationModel(BaseModel):
    source: str
    location: str
    snippet: str


class RetrievedChunkInfo(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float
    retrieval_type: str
    metadata: Dict[str, Any]


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask against uploaded documents")
    document_ids: Optional[List[uuid.UUID]] = Field(
        default=None, description="Optional document UUID filter"
    )
    top_k: int = Field(default=5, ge=1, le=15, description="Number of hybrid chunks to retrieve")


class ChatQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]
    retrieved_chunks: List[RetrievedChunkInfo]
    latency_ms: int
    tokens_used: int
    cache_hit: bool = False
