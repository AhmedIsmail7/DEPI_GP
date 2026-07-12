"""
LLM handler for VidEx — wraps Gemini.

Takes the video chunks we found and the user's question, builds a solid prompt, 
and gets Gemini to answer acting like the professor from the video.

It handles:
  - Normal text questions ("what did the prof say?")
  - Visual math questions ("solve the equation on the board")
  - Screenshot questions ("explain what I am looking at right now")
"""

import os
import hashlib
import functools

from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, wait_exponential, stop_after_attempt

from config import GEMINI_API_KEY, GEMINI_MODEL
from schemas import RetrievalResult, LLMAnswer


# Quick cache so we don't spam the API with the exact same question twice.
# Saves money and makes it way faster.
CACHE_MAXSIZE = 128


SYSTEM_PROMPT = (
    "You are the original professor teaching this video lecture. "
    "Answer the student's question directly, clearly, and conversationally.\n"
    "Rules:\n"
    "1. Prioritize the provided video transcripts and visual slides. "
    "They're your primary source of truth.\n"
    "2. If the student asks about an equation, diagram, or anything visual, "
    "analyze the provided images carefully and walk them through it step by step.\n"
    "3. If the video context alone isn't enough to fully answer the question, "
    "you CAN use your general knowledge — but you MUST tell the student: "
    "\"This part goes beyond what was covered in the video.\"\n"
    "4. Cite video sources using bracket indices (e.g., [1], [2]) when "
    "referencing specific timestamps.\n"
    "5. If a current video frame is provided, the student is looking at that "
    "exact moment right now. Reference it directly in your answer.\n"
    "6. Keep explanations clear and conversational — you're teaching, not "
    "writing a paper."
)


class VidExGenerator:
    """
    The core engine. Builds the prompt, talks to Gemini, and handles 
    retries if the API decides to randomly fail.
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Check your .env file or environment."
            )
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_id = GEMINI_MODEL

        # Response cache: avoids hitting the API for identical queries
        self._cache: dict[str, str] = {}
        self._cache_order: list[str] = []

        print(f"[LLM] Gemini handler ready — model: {self.model_id}")

    def _make_cache_key(self, query: str, contexts: list, frame_path: str | None) -> str:
        """Create a unique hash for the question + context so we can cache it."""
        raw = query + "||"
        for item in contexts:
            text = getattr(item, "text", str(item))
            ts = getattr(item, "timestamp", 0.0)
            raw += f"{text}:{ts}||"
        if frame_path:
            raw += frame_path
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        return self._cache.get(key)

    def _cache_put(self, key: str, value: str):
        if key in self._cache:
            self._cache_order.remove(key)
        self._cache[key] = value
        self._cache_order.append(key)
        # kick out the oldest cache entry if we're storing too many
        if len(self._cache_order) > CACHE_MAXSIZE:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_gemini(self, contents: list, system_instruction: str) -> str:
        """Actually hits the Gemini API. Auto-retries a few times if it fails."""
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        return response.text

    def generate_answer(self, user_query: str, contexts: list,
                        current_frame_path: str | None = None) -> str:
        """
        Builds the prompt and asks Gemini.
        
        It knows how to deal with:
          1. Just text (no images)
          2. Slide frames retrieved from the DB
          3. A screenshot the user just took
        """
        # Check cache first
        cache_key = self._make_cache_key(user_query, contexts, current_frame_path)
        cached = self._cache_get(cache_key)
        if cached is not None:
            print("[LLM] Cache hit — skipping API call.")
            return cached

        # Build the text context block from retrieved chunks
        context_block = ""
        for i, item in enumerate(contexts, start=1):
            text = getattr(item, "text", "No transcript available.")
            timestamp = getattr(item, "timestamp", 0.0)
            context_block += f"[{i}] Timestamp {timestamp}s: {text}\n"

        prompt_text = f"Student Question: {user_query}\n\nVideo Transcripts:\n{context_block}"

        # Assemble the multimodal content array
        api_contents = [prompt_text]

        # Scenario A: user uploaded a screenshot
        # (e.g., "solve the equation that's on screen right now")
        if current_frame_path and os.path.exists(current_frame_path):
            try:
                frame_img = Image.open(current_frame_path)
                api_contents.append(frame_img)
                api_contents.append(
                    "The student is currently paused on the frame above. "
                    "If their question relates to what's visible, reference it directly."
                )
            except Exception as e:
                print(f"[LLM] Couldn't load current frame ({current_frame_path}): {e}")

        # Scenario B: throw in any slide frames we got from the DB
        for i, item in enumerate(contexts, start=1):
            img_path = getattr(item, "image_path", None)
            if img_path and os.path.exists(img_path):
                try:
                    api_contents.append(f"Retrieved slide for context [{i}]:")
                    api_contents.append(Image.open(img_path))
                except Exception:
                    # if the image is missing or broken, just ignore it and keep going
                    continue

        answer = self._call_gemini(api_contents, SYSTEM_PROMPT)
        self._cache_put(cache_key, answer)
        return answer


# -----------------------------------------------
# Adapter: Hook it up to app.py
# -----------------------------------------------
class LLMAdapter:
    """
    Wraps the generator so app.py can just call `generate_response` 
    and get back exactly what the frontend expects.
    """

    def __init__(self):
        self.engine = VidExGenerator()

    def generate_response(self, query: str, context_chunks: list[RetrievalResult],
                          video_id: str | None = None,
                          current_frame_path: str | None = None) -> LLMAnswer:
        raw_answer = self.engine.generate_answer(
            user_query=query,
            contexts=context_chunks,
            current_frame_path=current_frame_path,
        )

        # Pull the timestamps out so we can display them nicely in the UI
        timestamps = []
        for chunk in context_chunks:
            ts = getattr(chunk, "timestamp", None)
            if ts is not None:
                timestamps.append(float(ts))

        return LLMAnswer(
            answer=raw_answer,
            source_timestamps=timestamps,
            video_id=video_id,
        )


# Singleton — this is what app.py imports
llm_handler = LLMAdapter()