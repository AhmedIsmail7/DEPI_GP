"""
LLM interface for VidEx — wraps Google's Gemini API via the google-genai SDK.
Handles prompt assembly from retrieved context, exponential backoff on
rate-limit/transient errors, and an LRU cache to avoid re-billing identical
queries against a limited free-tier quota.
"""

import time
import random
import hashlib
import functools
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config import GEMINI_API_KEY, GEMINI_MODEL
from schemas import RetrievalResult, LLMAnswer


SYSTEM_PROMPT = (
    "You are a helpful educational AI assistant. "
    "Use the provided context, which includes text and timestamps from a "
    "video, to answer the user's question. "
    "Always cite the source timestamps when explaining. "
    "If the answer is not in the context, clearly state that you don't "
    "have enough information."
)

MAX_RETRIES = 5
BASE_DELAY_SECONDS = 1.0
CACHE_MAXSIZE = 256


def _with_backoff(func):
    """
    Retries transient/rate-limit errors with exponential backoff + jitter.
    Re-raises immediately on non-retryable errors (e.g. bad API key,
    invalid request) so those fail fast instead of retrying uselessly.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except APIError as e:
                last_exc = e
                status = getattr(e, "code", None)
                # Only retry on rate limits (429) and server-side errors (5xx)
                if status is not None and status not in (429, 500, 502, 503, 504):
                    raise
                delay = BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"[LLM Handler] Retryable error ({status}), "
                      f"attempt {attempt + 1}/{MAX_RETRIES}, waiting {delay:.1f}s...")
                time.sleep(delay)
        raise last_exc
    return wrapper


class GeminiLLMHandler:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL
        self._cache: dict[str, LLMAnswer] = {}
        self._cache_order: list[str] = []  # tracks insertion order for LRU eviction
        print(f"--- [LLM Engine] Gemini Handler Initialized ({self.model}) ---")

    def _cache_key(self, query: str, context_chunks: list[RetrievalResult]) -> str:
        """
        Builds a stable hash key from the query + retrieved context so that
        identical (query, context) pairs hit the cache instead of the API.
        Context is included because the same query against updated retrieval
        results should NOT be served a stale cached answer.
        """
        raw = query + "||" + "||".join(
            f"{c.text}:{c.timestamp}" for c in context_chunks
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> LLMAnswer | None:
        return self._cache.get(key)

    def _cache_set(self, key: str, value: LLMAnswer):
        if key in self._cache:
            self._cache_order.remove(key)
        self._cache[key] = value
        self._cache_order.append(key)
        if len(self._cache_order) > CACHE_MAXSIZE:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

    @_with_backoff
    def _call_gemini(self, full_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )
        return response.text

    def generate_response(
        self, query: str, context_chunks: list[RetrievalResult], video_id: str
    ) -> LLMAnswer:
        """
        Builds a prompt from retrieved context and calls Gemini,
        with caching and retry handling applied.
        """
        cache_key = self._cache_key(query, context_chunks)
        cached = self._cache_get(cache_key)
        if cached is not None:
            print("[LLM Handler] Cache hit, skipping API call.")
            return cached

        context_text = "\n\n".join(
            f"[Source {i+1} | Timestamp: {chunk.timestamp}s]: {chunk.text}"
            for i, chunk in enumerate(context_chunks)
        )
        full_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        answer_text = self._call_gemini(full_prompt)

        result = LLMAnswer(
            answer=answer_text,
            source_timestamps=[c.timestamp for c in context_chunks],
            video_id=video_id,
        )
        self._cache_set(cache_key, result)
        return result


# Singleton
llm_handler = GeminiLLMHandler()

if __name__ == "__main__":
    sample_context = [
        RetrievalResult(
            video_id="test_video",
            text="Defining functions in python uses the 'def' keyword.",
            timestamp=72.5,
            combined_score=0.91,
        ),
        RetrievalResult(
            video_id="test_video",
            text="You can return values from functions using the return keyword.",
            timestamp=632.5,
            combined_score=0.88,
        ),
    ]
    query = "How do I define a function and return a value?"

    answer = llm_handler.generate_response(query, sample_context, video_id="test_video")
    print(f"\n--- [LLM Response] ---\n{answer.answer}")
    print(f"Sources: {answer.source_timestamps}")