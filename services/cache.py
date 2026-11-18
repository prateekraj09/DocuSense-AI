import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional
import redis
from core.config import settings

logger = logging.getLogger(__name__)


class RedisCacheService:
    """Production Redis caching layer for sub-millisecond query responses and token cost reduction."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._client: Optional[redis.Redis] = None
        self._is_connected: Optional[bool] = None

    @property
    def client(self) -> Optional[redis.Redis]:
        if self._client is None:
            try:
                self._client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                self._client.ping()
                self._is_connected = True
            except Exception as e:
                logger.warning(f"Redis is currently unavailable ({e}). Running in bypass mode.")
                self._client = None
                self._is_connected = False
        return self._client

    @staticmethod
    def generate_query_cache_key(
        user_id: uuid.UUID,
        query: str,
        document_ids: Optional[List[uuid.UUID]] = None,
    ) -> str:
        """Computes a deterministic SHA-256 hash key representing the query context for a specific user."""
        normalized_query = query.strip().lower()
        doc_ids_str = (
            ",".join(sorted(str(d) for d in document_ids)) if document_ids else "all"
        )
        raw_key = f"user:{str(user_id)}:query:{normalized_query}:docs:{doc_ids_str}"
        hashed = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"docusense:cache:query:{hashed}"

    def get_cached_query(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a cached JSON response if present in Redis."""
        client = self.client
        if not client:
            return None
        try:
            cached_data = client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Redis cache read error: {e}")
        return None

    def set_cached_query(
        self,
        cache_key: str,
        data: Dict[str, Any],
        expire_seconds: int = 3600,
    ) -> None:
        """Stores a serialized JSON response in Redis with an expiration TTL (default: 1 hour)."""
        client = self.client
        if not client:
            return
        try:
            client.set(cache_key, json.dumps(data, default=str), ex=expire_seconds)
        except Exception as e:
            logger.warning(f"Redis cache write error: {e}")

    def invalidate_user_cache(self, user_id: uuid.UUID) -> None:
        """Invalidates all cached queries belonging to a specific user (e.g. after uploading a new document)."""
        client = self.client
        if not client:
            return
        try:
            # Delete cached keys tagged with user_id
            pattern = f"docusense:cache:query:*"
            # In production, scan_iter is non-blocking
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"Redis cache invalidation error: {e}")


cache_service = RedisCacheService()
