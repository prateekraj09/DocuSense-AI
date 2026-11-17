import logging
from typing import List, Optional
from google import genai
from google.genai import types

from core.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddingService:
    """Service for generating high-dimensional dense vector embeddings using Google GenAI SDK."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not configured in environment or settings.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_embedding(self, text: str) -> List[float]:
        """Generates a 768-dimensional embedding for a single text string."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        try:
            config = types.EmbedContentConfig(output_dimensionality=self.dimension)
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=config,
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise RuntimeError(f"Gemini embedding API error: {str(e)}") from e

    def generate_embeddings_batch(
        self, texts: List[str], batch_size: int = 30
    ) -> List[List[float]]:
        """Generates 768-dimensional embeddings for a batch of text strings, chunked by batch_size."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Replace empty strings with a single space to avoid API validation errors
            sanitized_batch = [t if t.strip() else " " for t in batch]

            try:
                config = types.EmbedContentConfig(output_dimensionality=self.dimension)
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=sanitized_batch,
                    config=config,
                )
                for item in response.embeddings:
                    all_embeddings.append(item.values)
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings (batch {i}): {e}")
                raise RuntimeError(f"Gemini batch embedding error: {str(e)}") from e

        return all_embeddings


embedding_service = GeminiEmbeddingService()
