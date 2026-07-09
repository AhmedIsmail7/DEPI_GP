import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import DATABASE

load_dotenv()

class SemanticRetriever:
    def __init__(self, collection_name=None):
        self.client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        self.collection_name = collection_name or getattr(DATABASE, "text_collection", "text_chunks")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("--- [Retriever] Semantic Engine Initialized ---")

    def retrieve(self, query: str, top_k: int = 3):
        """
        1. Encode query to 384-dim vector
        2. Search in Qdrant
        3. Return top matches with payload
        """
        # Encode query
        query_vector = self.encoder.encode(query).tolist()

        # Search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        results = []
        for hit in search_results:
            results.append({
                "text": hit.payload.get("transcript") or hit.payload.get("text") or "",
                "timestamp": hit.payload.get("start_time") if hit.payload.get("start_time") is not None else hit.payload.get("timestamp", 0.0),
                "similarity": hit.score
            })
        
        return results

# Singleton
retriever = SemanticRetriever()

if __name__ == "__main__":
    test_query = "How to define a function in python?"
    matches = retriever.retrieve(test_query)
    
    print(f"--- [Test Query]: {test_query} ---")
    for m in matches:
        print(f"Found: {m['text'][:50]}... | Time: {m['timestamp']}s | Score: {m['similarity']:.4f}")