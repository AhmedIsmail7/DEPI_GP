"""
this file handles all the ai and llm stuff.
it talks to gemini api, builds the prompts with the video text/images,
and summarizes the chat history so we don't go over token limits.
"""

import os
import hashlib
from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, wait_exponential, stop_after_attempt

from config import GEMINI_API_KEY, GEMINI_MODEL, CHAT_HISTORY_TOKEN_LIMIT
from schemas import RetrievalResult, LLMAnswer


# simple cache in memory so we don't call api for same question twice
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
    main class for gemini stuff. handles prompts and retries if api fails.
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Check your .env file or environment."
            )
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_id = GEMINI_MODEL

        # setup cache for same questions
        self._cache: dict[str, str] = {}
        self._cache_order: list[str] = []

        print(f"[LLM] Gemini handler ready — model: {self.model_id}")

    def _make_cache_key(self, query: str, contexts: list, frame_path: str | None, chat_history: list[dict] | None = None, rolling_summary: str = "") -> str:
        """make a unique hash for the question and its context so we can cache it"""
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
            # We don't hash all history, just what is passed in
            for msg in chat_history:
                if isinstance(msg, dict):
                    raw += f"{msg.get('role', '')}:{msg.get('content', '')}||"
                else:
                    raw += f"{msg}||"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """guess tokens: 1 token is like 4 chars roughly"""
        return len(text) // 4

    def _cache_get(self, key: str) -> str | None:
        return self._cache.get(key)

    def _cache_put(self, key: str, value: str):
        if key in self._cache:
            self._cache_order.remove(key)
        self._cache[key] = value
        self._cache_order.append(key)
        # remove oldest item if cache is full
        if len(self._cache_order) > CACHE_MAXSIZE:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_gemini(self, contents: list, system_instruction: str, model_id: str | None = None) -> str:
        """call gemini api, and retry automatically if it fails"""
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
        mixes video text, chat history and current frame to get an answer.
        returns the answer and the new summary.
        """
        # 1. Budget Logic since we are poor :()
        recent_history = []
        overflow_history = []
        
        if chat_history:
            budget = CHAT_HISTORY_TOKEN_LIMIT
            # go back in history to count tokens
            # add msg pairs to keep context logic
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
            
            # whatever didn't fit goes to overflow
            if i >= 0:
                overflow_history = chat_history[:i+1]
                
        # 2. summarize old history if too long, runs blocking but only when overflow
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
                # use flash-lite for summary because its cheap and fast
                raw_new_summary = self._call_gemini([summary_prompt], "Maintain the rolling summary.", model_id="gemini-3.1-flash-lite")
                
                # hard cut if model talks too much just in case
                if len(raw_new_summary) > 600:
                    raw_new_summary = raw_new_summary[:597] + "..."
                    
                new_summary = raw_new_summary
                rolling_summary = new_summary
            except Exception as e:
                print(f"[LLM] Warning: Summarization failed ({e}). Keeping existing summary.")
                # if fail just keep old summary so app doesn't crash
                new_summary = rolling_summary

        # check cache first using the summary and recent history
        cache_key = self._make_cache_key(user_query, contexts, current_frame_path, recent_history, rolling_summary)
        cached = self._cache_get(cache_key)
        if cached is not None:
            print("[LLM] Cache hit — skipping API call.")
            return cached, new_summary

        # put text context together from chunks
        context_block = ""
        for idx, item in enumerate(contexts, start=1):
            text = getattr(item, "text", "No transcript available.")
            timestamp = getattr(item, "timestamp", 0.0)
            minutes, seconds = divmod(int(timestamp), 60)
            context_block += f"[{idx}] Timestamp {minutes}:{seconds:02d}: {text}\n"

        # put history text together
        history_text = ""
        if rolling_summary:
            history_text += f"Previous Conversation Summary:\n{rolling_summary}\n\n"
            
        if recent_history:
            history_text += "Recent Conversation History:\n" + "".join(recent_history)
            
        if history_text:
            history_text += "---\n\n"

        prompt_text = f"{history_text}Student Question: {user_query}\n\nVideo Transcripts:\n{context_block}"

        # mix text and images together for api
        api_contents = [prompt_text]

        # Case A: user paused and sent a screenshot (like "explain this")
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

        # Case B: add slide images from DB if available
        for i, item in enumerate(contexts, start=1):
            img_path = getattr(item, "image_path", None)
            if img_path and os.path.exists(img_path):
                try:
                    api_contents.append(f"Retrieved slide for context [{i}]:")
                    api_contents.append(Image.open(img_path))
                except Exception:
                    # if image is broken just skip it
                    continue

        answer_text = self._call_gemini(api_contents, SYSTEM_PROMPT)
        self._cache_put(cache_key, answer_text)
        return answer_text, new_summary


# -----------------------------------------------
# Adapter: connector to app.py
# -----------------------------------------------
class LLMAdapter:
    """
    wrapper so app.py doesn't have to deal with the messy generator logic directly.
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

        # get timestamps to show in frontend ui
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


# this is what app.py imports
llm_handler = LLMAdapter()