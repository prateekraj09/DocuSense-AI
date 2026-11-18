import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from core.config import settings
from services.retrieval import SearchResult

logger = logging.getLogger(__name__)

STRICT_GUARDRAIL_SYSTEM_INSTRUCTION = """
You are DocuSense AI, a rigorous enterprise document and code intelligence engine.

STRICT INSTRUCTIONS:
1. Answer the user's question using the factual context provided in the CHUNKS below.
2. You may synthesize, compare, contrast, or summarize across the provided chunks, provided every statement is factually grounded in the text.
3. If the provided context contains NO relevant information to address the question, you MUST respond EXACTLY with:
   "Insufficient information in uploaded documents to answer this question."
4. Do NOT extrapolate, speculate, or draw from outside world knowledge not found in the chunks.
5. Whenever you reference a fact, line of code, or conclusion, you MUST cite the source inline using the exact format:
   - For PDFs: [filename, Page X]
   - For code/text: [filename, Lines Y-Z]
6. Keep your answer concise, precise, and directly relevant to the user query.
"""


@dataclass
class Citation:
    source: str
    location: str  # e.g., "Page 2" or "Lines 10-25"
    snippet: str


@dataclass
class GenerationResult:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


class GroundedGenerationService:
    """Orchestrates grounded LLM generation with anti-hallucination guardrails and source citations."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = settings.GENERATION_MODEL
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not configured in environment or settings.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _format_context_chunks(chunks: List[SearchResult]) -> str:
        """Formats retrieved chunks with clear source metadata tags for the LLM."""
        formatted_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata or {}
            filename = meta.get("filename", "unknown_document")
            if "page_number" in meta:
                location = f"Page {meta['page_number']}"
            elif "start_line" in meta and "end_line" in meta:
                location = f"Lines {meta['start_line']}-{meta['end_line']}"
            else:
                location = "Unknown Location"

            block = f"--- CHUNK {i} [{filename}, {location}] ---\n{chunk.content}"
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    @staticmethod
    def _extract_citations(answer: str, chunks: List[SearchResult]) -> List[Citation]:
        """Parses inline citations like [filename, Page X] from the LLM answer."""
        citations: List[Citation] = []
        # Pattern matching: [filename.pdf, Page 1] or [app.py, Lines 1-20]
        pattern = r"\[([^,\]]+),\s*([^\]]+)\]"
        matches = re.findall(pattern, answer)

        seen = set()
        for filename, location in matches:
            key = (filename.strip(), location.strip())
            if key not in seen:
                seen.add(key)
                # Find matching chunk snippet
                matching_snippet = ""
                for c in chunks:
                    c_meta = c.metadata or {}
                    if c_meta.get("filename") == filename.strip():
                        matching_snippet = c.content[:200]
                        break

                citations.append(
                    Citation(
                        source=filename.strip(),
                        location=location.strip(),
                        snippet=matching_snippet,
                    )
                )

        return citations

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
    ) -> GenerationResult:
        """Sends the question + retrieved context to Gemini with anti-hallucination guardrails."""
        if not retrieved_chunks:
            return GenerationResult(
                answer="Insufficient information in uploaded documents to answer this question.",
                citations=[],
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
            )

        context_str = self._format_context_chunks(retrieved_chunks)
        user_prompt = (
            f"CONTEXT DOCUMENTS:\n{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Remember: Answer ONLY from the context above. Cite every fact as [filename, Page X] or [filename, Lines Y-Z]. "
            f"If not in the context, reply 'Insufficient information in uploaded documents to answer this question.'"
        )

        start_time = time.time()
        try:
            config = types.GenerateContentConfig(
                system_instruction=STRICT_GUARDRAIL_SYSTEM_INSTRUCTION,
                temperature=0.0,  # Zero temperature for deterministic factual extraction
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            answer = response.text.strip() if response.text else "Insufficient information in uploaded documents to answer this question."

            # Calculate token counts if available
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = usage.prompt_token_count if usage else len(user_prompt) // 4
            completion_tokens = usage.candidates_token_count if usage else len(answer) // 4

            citations = self._extract_citations(answer, retrieved_chunks)

            return GenerationResult(
                answer=answer,
                citations=citations,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            latency_ms = int((time.time() - start_time) * 1000)
            raise RuntimeError(f"LLM Generation Error: {str(e)}") from e

    def generate_grounded_answer_stream(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
    ):
        """Streams generated tokens and citations incrementally for Server-Sent Events (SSE)."""
        if not retrieved_chunks:
            yield {"type": "token", "content": "Insufficient information in uploaded documents to answer this question."}
            yield {"type": "citations", "citations": []}
            yield {"type": "done"}
            return

        context_str = self._format_context_chunks(retrieved_chunks)
        user_prompt = (
            f"CONTEXT DOCUMENTS:\n{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Remember: Answer ONLY from the context above. Cite every fact as [filename, Page X] or [filename, Lines Y-Z]. "
            f"If not in the context, reply 'Insufficient information in uploaded documents to answer this question.'"
        )

        try:
            config = types.GenerateContentConfig(
                system_instruction=STRICT_GUARDRAIL_SYSTEM_INSTRUCTION,
                temperature=0.0,
            )
            accumulated_text = ""
            for response_chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=user_prompt,
                config=config,
            ):
                if response_chunk.text:
                    accumulated_text += response_chunk.text
                    yield {"type": "token", "content": response_chunk.text}

            citations = self._extract_citations(accumulated_text, retrieved_chunks)
            yield {
                "type": "citations",
                "citations": [
                    {"source": c.source, "location": c.location, "snippet": c.snippet}
                    for c in citations
                ],
            }
            yield {"type": "done", "full_answer": accumulated_text}

        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield {"type": "error", "message": str(e)}


generation_service = GroundedGenerationService()
