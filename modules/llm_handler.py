"""
LLM Integration and Generation Engine.

Constructs multimodal prompts utilizing retrieved video context and 
interfaces with the Gemini API to generate contextual answers.
Supports text queries, visual context analysis, and chat history rolling summarization.
"""

import os
import hashlib
import functools

from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, wait_exponential, stop_after_attempt

from config import GEMINI_API_KEY, GEMINI_MODEL, CHAT_HISTORY_TOKEN_LIMIT
from schemas import RetrievalResult, LLMAnswer


# In-memory LRU cache to reduce latency and API consumption for identical queries.
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
    "4. Format your output clearly and naturally using markdown. Do not insert numeric citation brackets (e.g. [1], [2]) into your text.\n"
    "5. When mentioning a specific time in the video, always use the MM:SS format (e.g., 2:05) instead of raw seconds.\n"
    "6. If a current video frame is provided, the student is looking at that "
    "exact moment right now. Reference it directly in your answer.\n"
    "7. Keep explanations clear and conversational — you're teaching, not "
    "writing a paper."
)


class VidExGenerator:
    """
    Manages prompt construction, Gemini API execution, and transient failure retries.
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Check your .env file or environment."
            )
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_id = GEMINI_MODEL

        # Initialize response cache for identical contextual queries
        self._cache: dict[str, str] = {}
        self._cache_order: list[str] = []

        print(f"[LLM] Gemini handler ready — model: {self.model_id}")

    def _make_cache_key(self, query: str, contexts: list, frame_path: str | None, chat_history: list[dict] | None = None, rolling_summary: str = "") -> str:
        """Generates a unique SHA-256 hash identifying the query and its full context."""
        raw = query + "||"
        for item in contexts:
            text = getattr(item, "text", str(item))
            ts = getattr(item, "timestamp", 0.0)
            raw += f"{text}:{ts}||"
        if frame_path:
            raw += frame_path
        if rolling_summary:
            raw += f"Summary:{rolling_summary}||"
        if chat_history:
            # We don't hash all history, just what is passed in (which is token-capped)
            for msg in chat_history:
                if isinstance(msg, dict):
                    raw += f"{msg.get('role', '')}:{msg.get('content', '')}||"
                else:
                    raw += f"{msg}||"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation: 1 token ≈ 4 characters."""
        return len(text) // 4

    def _cache_get(self, key: str) -> str | None:
        return self._cache.get(key)

    def _cache_put(self, key: str, value: str):
        if key in self._cache:
            self._cache_order.remove(key)
        self._cache[key] = value
        self._cache_order.append(key)
        # Evict the oldest entry to maintain max size
        if len(self._cache_order) > CACHE_MAXSIZE:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_gemini(self, contents: list, system_instruction: str, model_id: str | None = None) -> str:
        """Executes the generate_content API call with exponential backoff."""
        target_model = model_id if model_id else self.model_id
        response = self.client.models.generate_content(
            model=target_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        return response.text

    def generate_answer(self, user_query: str, contexts: list,
                        current_frame_path: str | None = None,
                        chat_history: list[dict] | None = None,
                        rolling_summary: str = "") -> tuple[str, str | None]:
        """
        Constructs a multimodal prompt utilizing retrieved contexts and current user state.
        Returns a tuple of (generated_answer, updated_rolling_summary).
        """
        # 1. Budgeting Logic
        recent_history = []
        overflow_history = []
        
        if chat_history:
            budget = CHAT_HISTORY_TOKEN_LIMIT
            # Walk backwards through Q/A pairs (assuming length is even, or just stepping backwards)
            # We add messages in pairs if possible to maintain context
            i = len(chat_history) - 1
            while i >= 0:
                msg = chat_history[i]
                role_label = "Student" if msg["role"] == "user" else "Professor"
                formatted_msg = f"{role_label}: {msg['content']}\n\n"
                msg_tokens = self._estimate_tokens(formatted_msg)
                
                if budget - msg_tokens >= 0:
                    recent_history.insert(0, formatted_msg)
                    budget -= msg_tokens
                    i -= 1
                else:
                    break
            
            # Any messages that didn't fit are overflow
            if i >= 0:
                overflow_history = chat_history[:i+1]
                
        # 2. Rolling Summarization (Synchronous Blocking Call)
        # Note: This executes synchronously on the critical path, adding latency. 
        # However, it only fires intermittently when the token budget overflows.
        new_summary = None
        if overflow_history:
            print(f"[LLM] History exceeded budget. Summarizing {len(overflow_history)} overflow messages...")
            overflow_text = ""
            for msg in overflow_history:
                role = "Student" if msg["role"] == "user" else "Professor"
                overflow_text += f"{role}: {msg['content']}\n"
                
            summary_prompt = (
                "You are an AI maintaining conversational memory.\n"
                "Update the existing rolling summary with the new overflow conversation turns.\n"
                "Keep it strictly under 150 tokens (1-3 sentences): focus on the core topic, key entities, and any unresolved questions.\n\n"
                f"Existing Summary: {rolling_summary if rolling_summary else 'None'}\n\n"
                f"New Overflow Turns:\n{overflow_text}\n\n"
                "Return ONLY the updated summary text."
            )
            
            try:
                # Use flash-lite explicitly for summarization (fastest/cheapest tier)
                raw_new_summary = self._call_gemini([summary_prompt], "Maintain the rolling summary.", model_id="gemini-3.1-flash-lite")
                
                # Hard truncate backstop in case the model ignores the prompt length instruction (approx 150 tokens = 600 chars)
                if len(raw_new_summary) > 600:
                    raw_new_summary = raw_new_summary[:597] + "..."
                    
                new_summary = raw_new_summary
                rolling_summary = new_summary
            except Exception as e:
                print(f"[LLM] Warning: Summarization failed ({e}). Keeping existing summary.")
                # Fallback: keep previous summary, don't break the user's answer
                new_summary = rolling_summary

        # Check cache first (using the updated rolling summary and token-capped recent history)
        # Note: The cache is global per VidExGenerator instance. 
        # If chat_history is empty, rolling_summary is empty, so standalone questions bypass the summary
        # completely and hit the cache reliably across all users/sessions.
        cache_key = self._make_cache_key(user_query, contexts, current_frame_path, recent_history, rolling_summary)
        cached = self._cache_get(cache_key)
        if cached is not None:
            print("[LLM] Cache hit — skipping API call.")
            return cached, new_summary

        # Build the text context block from retrieved chunks
        context_block = ""
        for idx, item in enumerate(contexts, start=1):
            text = getattr(item, "text", "No transcript available.")
            timestamp = getattr(item, "timestamp", 0.0)
            minutes, seconds = divmod(int(timestamp), 60)
            context_block += f"[{idx}] Timestamp {minutes}:{seconds:02d}: {text}\n"

        # Build history context block
        history_text = ""
        if rolling_summary:
            history_text += f"Previous Conversation Summary:\n{rolling_summary}\n\n"
            
        if recent_history:
            history_text += "Recent Conversation History:\n" + "".join(recent_history)
            
        if history_text:
            history_text += "---\n\n"

        prompt_text = f"{history_text}Student Question: {user_query}\n\nVideo Transcripts:\n{context_block}"

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

        answer_text = self._call_gemini(api_contents, SYSTEM_PROMPT)
        self._cache_put(cache_key, answer_text)
        return answer_text, new_summary


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
                          current_frame_path: str | None = None,
                          chat_history: list[dict] | None = None,
                          rolling_summary: str = "") -> LLMAnswer:
        raw_answer, new_summary = self.engine.generate_answer(
            user_query=query,
            contexts=context_chunks,
            current_frame_path=current_frame_path,
            chat_history=chat_history,
            rolling_summary=rolling_summary
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
            new_summary=new_summary,
        )


# Singleton — this is what app.py imports
llm_handler = LLMAdapter()