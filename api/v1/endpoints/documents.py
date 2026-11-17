import os
import uuid
from typing import Annotated, List

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from core.database import get_db
from db.models import Document, DocumentChunk, DocumentStatus, User
from schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    SimilaritySearchItem,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)
from services.ingestion import process_document_sync
from services.parsers import ALLOWED_EXTENSIONS
from services.retrieval import retrieval_service

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and ingest a document asynchronously",
    description="Validates file, creates PENDING document record, enqueues processing job to Celery/Redis, and returns HTTP 202.",
)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    sync: bool = False,
) -> DocumentResponse:
    import base64
    from workers.tasks import ingest_document_task

    filename = file.filename or "uploaded_file.txt"
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    if sync:
        # Synchronous execution mode (for tests or instant CLI workflows)
        try:
            doc = await process_document_sync(
                db=db,
                user_id=current_user.id,
                filename=filename,
                content_bytes=file_bytes,
            )
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {str(e)}",
            )

        chunk_count_res = await db.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == doc.id)
        )
        chunk_count = chunk_count_res.scalar() or 0

        return DocumentResponse(
            id=doc.id,
            user_id=doc.user_id,
            filename=doc.filename,
            file_type=doc.file_type,
            status=doc.status,
            error_message=doc.error_message,
            created_at=doc.created_at,
            chunk_count=chunk_count,
        )

    # Production Asynchronous Ingestion (HTTP 202 Accepted + Celery Worker)
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        user_id=current_user.id,
        filename=filename,
        file_type=ext or "text",
        status=DocumentStatus.PENDING,
        error_message=None,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    content_b64 = base64.b64encode(file_bytes).decode("utf-8")

    try:
        # Push background task to Celery via Redis broker
        ingest_document_task.delay(
            document_id=str(doc.id),
            user_id=str(current_user.id),
            filename=filename,
            content_b64=content_b64,
        )
    except Exception as e:
        # If Celery broker is offline in local dev, mark and warn
        doc.status = DocumentStatus.FAILED
        doc.error_message = f"Failed to enqueue task to worker: {str(e)}"
        await db.commit()
        await db.refresh(doc)

    return DocumentResponse(
        id=doc.id,
        user_id=doc.user_id,
        filename=doc.filename,
        file_type=doc.file_type,
        status=doc.status,
        error_message=doc.error_message,
        created_at=doc.created_at,
        chunk_count=0,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List user documents",
    description="Retrieves all documents uploaded by the authenticated user with status and chunk count.",
)
async def list_documents(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentListResponse:
    # Strictly query documents matching authenticated user ID
    stmt = (
        select(Document, func.count(DocumentChunk.id).label("chunk_count"))
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .where(Document.user_id == current_user.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.all()

    documents: List[DocumentResponse] = []
    for doc, count in rows:
        documents.append(
            DocumentResponse(
                id=doc.id,
                user_id=doc.user_id,
                filename=doc.filename,
                file_type=doc.file_type,
                status=doc.status,
                error_message=doc.error_message,
                created_at=doc.created_at,
                chunk_count=count,
            )
        )

    return DocumentListResponse(documents=documents, total=len(documents))


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    description="Fetches a specific document by UUID, verifying owner authorization.",
)
async def get_document(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentResponse:
    stmt = (
        select(Document, func.count(DocumentChunk.id).label("chunk_count"))
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .where(Document.id == document_id, Document.user_id == current_user.id)
        .group_by(Document.id)
    )
    result = await db.execute(stmt)
    row = result.first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied.",
        )

    doc, count = row
    return DocumentResponse(
        id=doc.id,
        user_id=doc.user_id,
        filename=doc.filename,
        file_type=doc.file_type,
        status=doc.status,
        error_message=doc.error_message,
        created_at=doc.created_at,
        chunk_count=count,
    )


@router.delete(
    "/{document_id}",
    summary="Delete a document",
    description="Deletes a document and cascades deletion to all associated chunks and vector embeddings.",
)
async def delete_document(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    # Verify ownership before deletion
    stmt = select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied.",
        )

    await db.delete(doc)
    await db.commit()
    return {
        "detail": "Document and all associated vector embeddings successfully deleted.",
        "document_id": str(document_id),
    }


@router.post(
    "/search",
    response_model=SimilaritySearchResponse,
    summary="Vector similarity search",
    description="Executes semantic vector similarity search against the user's private knowledge base.",
)
async def search_documents(
    search_in: SimilaritySearchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SimilaritySearchResponse:
    results = await retrieval_service.search_similar_chunks(
        db=db,
        user_id=current_user.id,
        query=search_in.query,
        top_k=search_in.top_k,
        document_ids=search_in.document_ids,
    )

    items = [
        SimilaritySearchItem(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            content=r.content,
            metadata=r.metadata,
            score=r.score,
        )
        for r in results
    ]

    return SimilaritySearchResponse(
        query=search_in.query,
        count=len(items),
        results=items,
    )
