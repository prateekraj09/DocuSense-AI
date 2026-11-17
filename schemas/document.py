import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from db.models import DocumentStatus


class DocumentResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    filename: str
    file_type: str
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime
    chunk_count: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class DocumentChunkDetail(BaseModel):
    id: uuid.UUID
    chunk_index: int
    content: str
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class SimilaritySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question or search query string")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top chunks to retrieve")
    document_ids: Optional[List[uuid.UUID]] = Field(
        default=None, description="Optional list of document IDs to scope the search"
    )


class SimilaritySearchItem(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    metadata: Dict[str, Any]
    score: float


class SimilaritySearchResponse(BaseModel):
    query: str
    count: int
    results: List[SimilaritySearchItem]
