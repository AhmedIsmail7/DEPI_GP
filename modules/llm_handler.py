try:
    import cohere
except Exception:  # pragma: no cover - optional dependency
    cohere = None

from config import (
    COHERE_API_KEY,
    COHERE_MODEL,
)


class CohereLLMHandler:

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or COHERE_API_KEY
        self.model = model or COHERE_MODEL
        self._client = None
        print("--- [LLM Engine] Cohere Handler Initialized ---")

    def _get_client(self):
        if self._client is None:
            if cohere is None:
                raise ImportError("cohere is required to use the LLM handler")
            if not self.api_key:
                raise ValueError("COHERE_API_KEY is not configured")
            self._client = cohere.ClientV2(api_key=self.api_key)
        return self._client

    def generate_response(
        self,
        query: str,
        context_chunks: list
    ):
        """
        Generate an answer using the retrieved context.
        """

        context_text = "\n\n".join(
            [
                f"[Source {i + 1} | Timestamp: {chunk['timestamp']}s]\n"
                f"{chunk['text']}"
                for i, chunk in enumerate(context_chunks)
            ]
        )

        system_prompt = (
            "You are a helpful educational AI assistant. "
            "Answer ONLY using the provided context. "
            "Always cite the timestamp of the source. "
            "If the answer is not found in the context, "
            "say that you do not have enough information."
        )

        client = self._get_client()
        response = client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_text}\n\n"
                        f"Question:\n{query}"
                    ),
                },
            ],
        )

        content = getattr(getattr(response, "message", None), "content", None)
        if content is None and isinstance(response, dict):
            content = response.get("message", {}).get("content", [])

        if not content:
            raise ValueError("LLM response did not contain any content")

        first_item = content[0]
        if isinstance(first_item, dict):
            return first_item.get("text", "")
        return getattr(first_item, "text", "")


_llm_handler = None


def get_llm_handler():
    """Return a singleton CohereLLMHandler, initializing lazily."""
    global _llm_handler
    if _llm_handler is None:
        _llm_handler = CohereLLMHandler()
    return _llm_handler


if __name__ == "__main__":

    sample_context = [
        {
            "text": "Defining functions in Python uses the 'def' keyword.",
            "timestamp": 72.5,
        },
        {
            "text": "Use the return keyword to return a value.",
            "timestamp": 632.5,
        },
    ]

    question = "How do I define a function and return a value?"

    handler = get_llm_handler()
    answer = handler.generate_response(
        question,
        sample_context,
    )

    print("\n--- LLM Response ---\n")
    print(answer)