import json
import pytest
from httpx import AsyncClient

from services.cache import cache_service
from services.embedding import GeminiEmbeddingService
from services.generation import GroundedGenerationService


@pytest.mark.asyncio
async def test_redis_cache_key_generation():
    import uuid
    user_id = uuid.uuid4()
    key1 = cache_service.generate_query_cache_key(user_id, "What is DocuSense AI?")
    key2 = cache_service.generate_query_cache_key(user_id, "  what is docusense ai?  ")
    # Normalized queries for the same user should produce identical cache keys
    assert key1 == key2


@pytest.mark.asyncio
async def test_chat_query_caching_lifecycle(async_client: AsyncClient, monkeypatch):
    # Mock embedding generator
    monkeypatch.setattr(GeminiEmbeddingService, "generate_embedding", lambda self, text: [0.0] * 768)

    # In-memory dictionary mock for Redis cache in test environment
    in_memory_cache = {}
    monkeypatch.setattr(cache_service, "get_cached_query", lambda key: in_memory_cache.get(key))
    monkeypatch.setattr(cache_service, "set_cached_query", lambda key, data, expire_seconds=3600: in_memory_cache.update({key: data}))

    user_data = {"email": "cache_user@docusense.ai", "password": "CachePassword123!"}
    await async_client.post("/api/v1/auth/register", json=user_data)
    login_resp = await async_client.post("/api/v1/auth/login", json=user_data)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    query_payload = {"query": "Explain Redis caching in DocuSense AI"}

    # 1. First Call: Cache MISS (cache_hit = False)
    resp1 = await async_client.post("/api/v1/chat/query", headers=headers, json=query_payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["cache_hit"] is False

    # 2. Second Call: Cache HIT (cache_hit = True, instant return)
    resp2 = await async_client.post("/api/v1/chat/query", headers=headers, json=query_payload)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["cache_hit"] is True
    assert data2["answer"] == data1["answer"]


@pytest.mark.asyncio
async def test_sse_chat_streaming(async_client: AsyncClient, monkeypatch):
    monkeypatch.setattr(GeminiEmbeddingService, "generate_embedding", lambda self, text: [0.0] * 768)

    user_data = {"email": "stream_user@docusense.ai", "password": "StreamPassword123!"}
    await async_client.post("/api/v1/auth/register", json=user_data)
    login_resp = await async_client.post("/api/v1/auth/login", json=user_data)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    query_payload = {"query": "Stream test question"}
    resp = await async_client.post("/api/v1/chat/stream", headers=headers, json=query_payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text_content = resp.text
    assert "data: " in text_content
    assert "[DONE]" in text_content


@pytest.mark.asyncio
async def test_rate_limiter_protection(async_client: AsyncClient):
    from core.rate_limiter import limiter
    # Verify limiter instance is properly registered
    assert limiter is not None
    resp = await async_client.get("/health")
    assert resp.status_code == 200
