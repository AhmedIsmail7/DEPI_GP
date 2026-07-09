import os
import cohere
from dotenv import load_dotenv

load_dotenv()

class CohereLLMHandler:
    def __init__(self):
        self.client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        self.model = "command-a-03-2025"
        print("--- [LLM Engine] Cohere Handler Initialized ---")

    def generate_response(self, query: str, context_chunks: list):
        """
        Creates a prompt with retrieved context and calls Cohere.
        """
        context_text = "\n\n".join([
            f"[Source {i+1} | Timestamp: {chunk['timestamp']}s]: {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        system_prompt = (
            "You are a helpful educational AI assistant. "
            "Use the provided context (which includes text and timestamps from a video) to answer the user's question. "
            "Always cite the source timestamps when explaining. "
            "If the answer is not in the context, clearly state that you don't have enough information."
        )

        full_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ]
        )

        return response.message.content[0].text

# Singleton
llm_handler = CohereLLMHandler()

if __name__ == "__main__":
    sample_context = [
        {"text": "Defining functions in python uses the 'def' keyword.", "timestamp": 72.5},
        {"text": "You can return values from functions using the return keyword.", "timestamp": 632.5}
    ]
    query = "How do I define a function and return a value?"
    
    answer = llm_handler.generate_response(query, sample_context)
    print(f"\n--- [LLM Response] ---\n{answer}")