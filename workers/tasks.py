import asyncio
import base64
import logging
import uuid
from typing import Any, Dict

from celery import Task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import Document, DocumentStatus
from services.ingestion import process_document_sync
from workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _get_worker_async_session() -> async_sessionmaker[AsyncSession]:
    """Creates a dedicated database engine and session factory for worker tasks."""
    engine = create_async_engine(
        settings.get_database_url(),
        pool_pre_ping=True,
        echo=False,
    )
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


class IngestionTask(Task):
    """Custom Celery Task base class with automatic failure logging."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")


@celery_app.task(
    name="ingest_document_task",
    base=IngestionTask,
    bind=True,
    max_retries=3,
    default_retry_delay=10,
)
def ingest_document_task(
    self,
    document_id: str,
    user_id: str,
    filename: str,
    content_b64: str,
) -> Dict[str, Any]:
    """Background task that parses, chunks, generates embeddings, and indexes a document."""
    doc_uuid = uuid.UUID(document_id)
    user_uuid = uuid.UUID(user_id)
    file_bytes = base64.b64decode(content_b64)

    async def _run() -> Dict[str, Any]:
        session_factory = _get_worker_async_session()
        async with session_factory() as db:
            try:
                # Update status to PROCESSING
                stmt = select(Document).where(Document.id == doc_uuid)
                res = await db.execute(stmt)
                doc = res.scalar_one_or_none()
                if doc:
                    doc.status = DocumentStatus.PROCESSING
                    await db.commit()

                # Process document
                await process_document_sync(
                    db=db,
                    user_id=user_uuid,
                    filename=filename,
                    content_bytes=file_bytes,
                )
                logger.info(f"Successfully processed document {document_id}")
                return {"status": "SUCCESS", "document_id": document_id}

            except Exception as exc:
                logger.error(f"Error in background ingestion for doc {document_id}: {exc}")
                stmt = select(Document).where(Document.id == doc_uuid)
                res = await db.execute(stmt)
                doc = res.scalar_one_or_none()
                if doc:
                    doc.status = DocumentStatus.FAILED
                    doc.error_message = str(exc)
                    await db.commit()
                raise exc

    return asyncio.run(_run())
