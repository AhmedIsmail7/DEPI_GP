import torch
from qdrant_client import QdrantClient, models
from modules.embeddings import embedding_manager

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, DEFAULT_TOP_K
from schemas import RetrievalResult


# Fusion weights: text search tends to be more semantically precise for
# lecture content, visual search is a supporting signal. Adjust after
# testing against real queries.
TEXT_WEIGHT = 0.7
VISUAL_WEIGHT = 0.3


def fuse_search_results(text_hits, visual_hits, text_weight: float, visual_weight: float) -> list[dict]:
    """
    Pure fusion logic, extracted from retrieve() so it's testable without
    a live Qdrant connection. text_hits/visual_hits are lists of objects
    with .id, .payload, .score (matching qdrant_client's ScoredPoint shape).
    Returns a list of dicts with combined scores, sorted descending.
    """
    merged: dict = {}

    for hit in text_hits:
        merged[hit.id] = {"payload": hit.payload, "text_score": hit.score, "visual_score": None}

    for hit in visual_hits:
        if hit.id in merged:
            merged[hit.id]["visual_score"] = hit.score
        else:
            merged[hit.id] = {"payload": hit.payload, "text_score": None, "visual_score": hit.score}

    results = []
    for entry in merged.values():
        text_score = entry["text_score"] or 0.0
        visual_score = entry["visual_score"] or 0.0
        combined = (text_weight * text_score) + (visual_weight * visual_score)
        results.append({**entry, "combined_score": combined})

    results.sort(key=lambda r: r["combined_score"], reverse=True)
    return results


class SemanticRetriever:
    def __init__(self, collection_name: str = QDRANT_COLLECTION_NAME):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = collection_name

    def _search_named_vector(
        self, vector_name: str, query_vector: list[float], top_k: int, video_id: str | None,
    ):
        query_filter = None
        if video_id is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=vector_name,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return response.points

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, video_id: str | None = None) -> list[RetrievalResult]:
        """
        Unified-space retrieval: a single SigLIP query embedding is compared
        directly against both text_vector and image_vector, since both now
        live in the same 768-dim space. Results are fused by point ID with
        a weighted score combination.
        """
        query_vector = embedding_manager.get_text_embedding(query)

        fetch_k = max(top_k * 2, top_k)
        text_hits = self._search_named_vector("text_vector", query_vector, fetch_k, video_id)
        visual_hits = self._search_named_vector("image_vector", query_vector, fetch_k, video_id)

        fused = fuse_search_results(text_hits, visual_hits, TEXT_WEIGHT, VISUAL_WEIGHT)

        results = []
        for entry in fused[:top_k]:
            payload = entry["payload"]
            results.append(RetrievalResult(
                video_id=payload["video_id"],
                text=payload["text"],
                timestamp=payload["timestamp"],
                text_score=entry["text_score"],
                visual_score=entry["visual_score"],
                combined_score=entry["combined_score"],
            ))
        return results


retriever = SemanticRetriever()

if __name__ == "__main__":
    test_query = "How to define a function in python?"
    matches = retriever.retrieve(test_query)

    print(f"--- [Test Query]: {test_query} ---")
    for m in matches:
        print(
            f"'{m.text[:50]}...' | t={m.timestamp}s | "
            f"text={m.text_score} visual={m.visual_score} combined={m.combined_score:.4f}"
        )