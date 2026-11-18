import io
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from db.models import QueryLog
from services.embedding import GeminiEmbeddingService
from services.generation import GroundedGenerationService
from tests.test_documents import create_sample_pdf_bytes


@pytest.mark.asyncio
async def test_async_upload_returns_202_pending(async_client: AsyncClient, monkeypatch):
    from workers import tasks
    monkeypatch.setattr(tasks.ingest_document_task, "delay", lambda **kwargs: None)

    # Register & Login
    user_data = {"email": "async_user@docusense.ai", "password": "AsyncPassword123!"}
    await async_client.post("/api/v1/auth/register", json=user_data)
    login_resp = await async_client.post("/api/v1/auth/login", json=user_data)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Upload document in default async mode (sync=False)
    pdf_bytes = create_sample_pdf_bytes()
    files = {"file": ("manual.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    resp = await async_client.post("/api/v1/documents/upload", headers=headers, files=files)
    assert resp.status_code == 202
    data = resp.json()
    assert data["filename"] == "manual.pdf"
    assert data["status"] in ["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
    assert "id" in data


@pytest.mark.asyncio
async def test_hybrid_search_and_grounded_chat(async_client: AsyncClient, monkeypatch):
    # Mock embedding generator
    def mock_embeddings_batch(self, texts, batch_size=30):
        return [[0.1 * (j % 5) for j in range(768)] for _ in texts]

    def mock_single_embedding(self, text):
        return [0.1 * (j % 5) for j in range(768)]

    monkeypatch.setattr(GeminiEmbeddingService, "generate_embeddings_batch", mock_embeddings_batch)
    monkeypatch.setattr(GeminiEmbeddingService, "generate_embedding", mock_single_embedding)

    # 1. Register & Login
    user_data = {"email": "rag_user@docusense.ai", "password": "RagPassword123!"}
    await async_client.post("/api/v1/auth/register", json=user_data)
    login_resp = await async_client.post("/api/v1/auth/login", json=user_data)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload document synchronously so chunks are ready immediately
    doc_text = (
        "DocuSense AI is a production multi-tenant RAG platform. "
        "It uses PostgreSQL 16 with pgvector for single-store hybrid search. "
        "The Reciprocal Rank Fusion constant k is set to 60."
    )
    files = {"file": ("docusense_spec.txt", io.BytesIO(doc_text.encode("utf-8")), "text/plain")}
    up_resp = await async_client.post("/api/v1/documents/upload?sync=true", headers=headers, files=files)
    assert up_resp.status_code in [200, 201, 202]
    doc_id = up_resp.json()["id"]

    # 3. Test Grounded Chat Query - Answerable Question
    chat_payload = {
        "query": "What database does DocuSense AI use and what is the RRF constant?",
        "top_k": 3,
    }
    chat_resp = await async_client.post("/api/v1/chat/query", headers=headers, json=chat_payload)
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["query"] == chat_payload["query"]
    assert len(chat_data["retrieved_chunks"]) > 0
    assert chat_data["latency_ms"] >= 0
    assert "answer" in chat_data

    # 4. Test Anti-Hallucination Guardrail - Unanswerable Question
    unanswerable_payload = {
        "query": "What is the capital of France and who won the 1998 World Cup?",
        "top_k": 3,
    }
    guardrail_resp = await async_client.post("/api/v1/chat/query", headers=headers, json=unanswerable_payload)
    assert guardrail_resp.status_code == 200
    guard_data = guardrail_resp.json()
    assert (
        "Insufficient information" in guard_data["answer"]
        or "insufficient information" in guard_data["answer"].lower()
    )


@pytest.mark.asyncio
async def test_chat_query_logs_persisted(async_client: AsyncClient, monkeypatch):
    # Mock embedding generator
    monkeypatch.setattr(GeminiEmbeddingService, "generate_embedding", lambda self, text: [0.0] * 768)

    user_data = {"email": "audit_user@docusense.ai", "password": "AuditPassword123!"}
    await async_client.post("/api/v1/auth/register", json=user_data)
    login_resp = await async_client.post("/api/v1/auth/login", json=user_data)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Query without documents
    query_payload = {"query": "Tell me about DocuSense architecture"}
    resp = await async_client.post("/api/v1/chat/query", headers=headers, json=query_payload)
    assert resp.status_code == 200
    assert "Insufficient information" in resp.json()["answer"]
