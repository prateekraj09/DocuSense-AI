import logging
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core.config import settings

logger = logging.getLogger(__name__)


def _get_rate_limiter_storage() -> str:
    """Verifies Redis connection for distributed rate limiting; falls back to in-memory storage if offline."""
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=1)
        client.ping()
        return settings.REDIS_URL
    except Exception:
        logger.info("Using in-memory storage for SlowAPI rate limiter.")
        return "memory://"


limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_get_rate_limiter_storage(),
    default_limits=["120/minute"],
    headers_enabled=False,
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom JSON response for HTTP 429 Rate Limit Exceeded."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}. Please slow down your requests to protect LLM resources.",
            "retry_after": getattr(exc, "retry_after", 60),
        },
    )
