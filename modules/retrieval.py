try:
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    TEXT_EMBEDDING_MODEL,
)


class SemanticRetriever:

    def __init__(self):
        if QdrantClient is None or SentenceTransformer is None:
            raise ImportError("qdrant-client and sentence-transformers are required for retrieval")

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        self.collection_name = COLLECTION_NAME

        self.encoder = SentenceTransformer(
            TEXT_EMBEDDING_MODEL
        )

        print("--- [Retriever] Semantic Engine Initialized ---")

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ):
        """
        Encode the query, search Qdrant,
        and return the top matching chunks.
        """

        query_vector = self.encoder.encode(query)
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=(
                "text_vector",
                query_vector
            ),
            limit=top_k
        )

        results = []

        for hit in search_results:

            results.append({
                "text": hit.payload["text"],
                "timestamp": hit.payload["timestamp"],
                "similarity": hit.score,
            })

        return results


retriever = None


def get_retriever():
    global retriever
    if retriever is None:
        retriever = SemanticRetriever()
    return retriever


if __name__ == "__main__":

    test_query = "How do I define a function in Python?"

    matches = get_retriever().retrieve(test_query)

    print(f"\nQuery: {test_query}\n")

    for i, match in enumerate(matches, start=1):

        print(f"Result {i}")
        print(f"Timestamp : {match['timestamp']} s")
        print(f"Similarity: {match['similarity']:.4f}")
        print(f"Text      : {match['text']}")
        print("-" * 50)