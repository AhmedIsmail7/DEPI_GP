import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

class SemanticRetriever:
    def __init__(self, collection_name="vedex_knowledge"):
        self.client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        self.collection_name = collection_name
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
            query_vector=("text_vector", query_vector),
            limit=top_k
        )

        results = []
        for hit in search_results:
            results.append({
                "text": hit.payload.get("text"),
                "timestamp": hit.payload.get("timestamp"),
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