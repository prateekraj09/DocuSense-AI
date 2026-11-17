import io
import pytest
import pymupdf
from httpx import AsyncClient

from services.chunker import DocumentChunker
from services.parsers import CodeAndTextParser, PDFParser, parse_document


def create_sample_pdf_bytes() -> bytes:
    """Creates a 2-page in-memory PDF for testing."""
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((50, 72), "DocuSense AI is a multi-tenant production RAG platform.")
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Page 2 covers hybrid search using pgvector and PostgreSQL tsvector.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_parser_page_extraction():
    pdf_bytes = create_sample_pdf_bytes()
    blocks = PDFParser.parse(pdf_bytes, filename="test.pdf")
    assert len(blocks) == 2
    assert blocks[0].metadata["page_number"] == 1
    assert "DocuSense AI" in blocks[0].text
    assert blocks[1].metadata["page_number"] == 2
    assert "Page 2" in blocks[1].text


def test_code_parser_line_tracking():
    code_text = "def hello():\n    print('Hello World')\n    return True\n"
    blocks = CodeAndTextParser.parse(code_text.encode("utf-8"), filename="app.py")
    assert len(blocks) == 1
    assert blocks[0].metadata["filename"] == "app.py"
    assert blocks[0].metadata["start_line"] == 1
    assert "def hello():" in blocks[0].text


def test_chunker_overlapping_and_boundaries():
    text = (
        "Paragraph one discusses systems engineering and asynchronous job queues. "
        "Paragraph two discusses PostgreSQL pgvector and vector embeddings. "
        "Paragraph three discusses Redis caching and latency optimization."
    )
    blocks = CodeAndTextParser.parse(text.encode("utf-8"), filename="notes.txt")
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=30)
    chunks = chunker.chunk_blocks(blocks)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= 120
        assert "filename" in c.metadata


@pytest.mark.asyncio
async def test_document_upload_and_multitenant_isolation(async_client: AsyncClient, monkeypatch):
    # Mock embedding generator to ensure fast, deterministic tests without external API calls
    def mock_embeddings_batch(self, texts, batch_size=30):
        # Return deterministic 768-dim vectors
        return [[0.05 * (j % 10) for j in range(768)] for _ in texts]

    def mock_single_embedding(self, text):
        return [0.05 * (j % 10) for j in range(768)]

    from services.embedding import GeminiEmbeddingService
    monkeypatch.setattr(GeminiEmbeddingService, "generate_embeddings_batch", mock_embeddings_batch)
    monkeypatch.setattr(GeminiEmbeddingService, "generate_embedding", mock_single_embedding)

    # 1. Register and login User A
    user_a_data = {"email": "user_a@docusense.ai", "password": "UserAPassword123!"}
    await async_client.post("/api/v1/auth/register", json=user_a_data)
    login_a_resp = await async_client.post("/api/v1/auth/login", json=user_a_data)
    token_a = login_a_resp.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # 2. User A uploads a PDF document
    pdf_bytes = create_sample_pdf_bytes()
    files = {"file": ("architecture.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_resp = await async_client.post("/api/v1/documents/upload", headers=headers_a, files=files)
    assert upload_resp.status_code == 201
    doc_a = upload_resp.json()
    assert doc_a["filename"] == "architecture.pdf"
    assert doc_a["status"] == "COMPLETED"
    assert doc_a["chunk_count"] >= 2
    doc_a_id = doc_a["id"]

    # 3. User A lists documents -> sees 1 document
    list_a_resp = await async_client.get("/api/v1/documents", headers=headers_a)
    assert list_a_resp.status_code == 200
    assert list_a_resp.json()["total"] == 1

    # 4. Register and login User B
    user_b_data = {"email": "user_b@docusense.ai", "password": "UserBPassword123!"}
    await async_client.post("/api/v1/auth/register", json=user_b_data)
    login_b_resp = await async_client.post("/api/v1/auth/login", json=user_b_data)
    token_b = login_b_resp.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 5. User B lists documents -> MUST BE EMPTY (Strict Multi-Tenancy Isolation!)
    list_b_resp = await async_client.get("/api/v1/documents", headers=headers_b)
    assert list_b_resp.status_code == 200
    assert list_b_resp.json()["total"] == 0

    # 6. User B attempts to access User A's document -> 404 Forbidden/Not Found
    get_unauth = await async_client.get(f"/api/v1/documents/{doc_a_id}", headers=headers_b)
    assert get_unauth.status_code == 404

    # 7. User B attempts to delete User A's document -> 404 Forbidden/Not Found
    del_unauth = await async_client.delete(f"/api/v1/documents/{doc_a_id}", headers=headers_b)
    assert del_unauth.status_code == 404

    # 8. User A searches for similar chunks
    search_payload = {"query": "Tell me about DocuSense AI and pgvector", "top_k": 3}
    search_a_resp = await async_client.post("/api/v1/documents/search", headers=headers_a, json=search_payload)
    assert search_a_resp.status_code == 200
    search_data_a = search_a_resp.json()
    assert search_data_a["count"] > 0
    assert "DocuSense AI" in search_data_a["results"][0]["content"]

    # 9. User B searches with identical query -> 0 results because User B has no documents
    search_b_resp = await async_client.post("/api/v1/documents/search", headers=headers_b, json=search_payload)
    assert search_b_resp.status_code == 200
    assert search_b_resp.json()["count"] == 0

    # 10. User A deletes their document -> cascade delete
    del_resp = await async_client.delete(f"/api/v1/documents/{doc_a_id}", headers=headers_a)
    assert del_resp.status_code == 200
    # Confirm document list is now empty for User A
    list_after_del = await async_client.get("/api/v1/documents", headers=headers_a)
    assert list_after_del.json()["total"] == 0
