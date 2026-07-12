"""
Dual-modality retrieval for VidEx.

We hit both the text index (Whisper) and the image index (SigLIP) in Qdrant,
mash the results together by timestamp so we don't get duplicates,
and hand the best context chunks back to the LLM.

Everything uses the shared 768-dim SigLIP model so we're comparing apples to apples.
"""

from qdrant_client import QdrantClient, models

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, DEFAULT_TOP_K
from schemas import RetrievalResult


# How we balance the scores.
# Text is usually better for spoken lecture content, but visual search is 
# awesome for catching equations on the whiteboard that the prof never reads out loud.
TEXT_WEIGHT = 0.7
VISUAL_WEIGHT = 0.3


class VidExRetriever:
    """
    Does the heavy lifting of talking to Qdrant.
    There's a mock mode in here if you want to test without spinning up a live cluster.
    """

    def __init__(self, collection_name: str = QDRANT_COLLECTION_NAME, use_mock: bool = False):
        self.use_mock = use_mock
        self.collection_name = collection_name

        if not self.use_mock:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            self.client = None
            print("[Retrieval] Running in mock mode — no Qdrant connection.")

    def _build_filter(self, video_id: str | None) -> models.Filter | None:
        """Limits the search to a specific video so we don't pull answers from random lectures."""
        if video_id is None:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
        )

    def _search_vector(self, vector_name: str, query_vector: list[float],
                       limit: int, video_id: str | None) -> list:
        """Helper to run a basic search against one specific index in Qdrant."""
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=(vector_name, query_vector),
            query_filter=self._build_filter(video_id),
            limit=limit,
            with_payload=True,
        )

    def search_multimodal_context(self, user_query: str, limit: int = 3,
                                  video_id: str | None = None) -> list[dict]:
        """
        The magic dual-search method. It hits both text and image indices with the exact
        same vector, then fuses them by timestamp. If the same chunk pops up in both, 
        we merge them so we don't feed the LLM duplicate info.
        """
        if self.use_mock:
            print(f"[Retrieval] Mock search for: '{user_query[:50]}...'")
            return [
                {
                    "text": "When we analyze a binary search tree, if the tree becomes "
                            "unbalanced like a single linked list, our lookups degrade. "
                            "In the absolute worst case, searching takes O(n) linear time.",
                    "timestamp": 120.0,
                    "video_id": video_id or "mock_video",
                    "text_score": 0.92,
                    "visual_score": None,
                    "combined_score": 0.92,
                },
                {
                    "text": "In a perfectly balanced binary search tree, every split "
                            "eliminates half of the remaining elements. This gives us "
                            "an ideal lookup speed of O(log n).",
                    "timestamp": 95.0,
                    "video_id": video_id or "mock_video",
                    "text_score": 0.78,
                    "visual_score": None,
                    "combined_score": 0.78,
                },
            ]

        # Since SigLIP handles both text and images in the same 768-dim space,
        # we only need to encode the query once.
        # Importing here inside the function so PyTorch doesn't slow down the whole app on boot.
        from modules.embeddings import embedding_manager
        query_vector = embedding_manager.get_text_embedding(user_query)

        # Grab extra hits from each index because fusion will likely merge a few of them
        fetch_k = max(limit * 2, 6)
        text_hits = self._search_vector("text_vector", query_vector, fetch_k, video_id)
        visual_hits = self._search_vector("image_vector", query_vector, fetch_k, video_id)

        # --- Fusion time ---
        # If the same timestamp shows up in both results, merge them and keep the highest score.
        fused: dict[float, dict] = {}

        for hit in text_hits:
            ts = hit.payload.get("timestamp", 0.0)
            if ts not in fused:
                fused[ts] = {
                    "text": hit.payload.get("text", ""),
                    "timestamp": ts,
                    "video_id": hit.payload.get("video_id", video_id),
                    "text_score": hit.score,
                    "visual_score": None,
                }
            else:
                # same timestamp already seen — keep the higher text score
                existing = fused[ts]["text_score"]
                if existing is None or hit.score > existing:
                    fused[ts]["text_score"] = hit.score

        for hit in visual_hits:
            ts = hit.payload.get("timestamp", 0.0)
            if ts not in fused:
                fused[ts] = {
                    "text": hit.payload.get("text", ""),
                    "timestamp": ts,
                    "video_id": hit.payload.get("video_id", video_id),
                    "text_score": None,
                    "visual_score": hit.score,
                }
            else:
                existing = fused[ts]["visual_score"]
                if existing is None or hit.score > existing:
                    fused[ts]["visual_score"] = hit.score

        # Math out the final combined score and sort the results
        for entry in fused.values():
            t_score = entry["text_score"] or 0.0
            v_score = entry["visual_score"] or 0.0
            entry["combined_score"] = (TEXT_WEIGHT * t_score) + (VISUAL_WEIGHT * v_score)

        results = sorted(fused.values(), key=lambda x: x["combined_score"], reverse=True)
        return results[:limit]


# --------------------------------------------------
# Adapter: Connects the heavy lifting to app.py
# --------------------------------------------------
class RetrieverAdapter:
    """
    Wraps the retriever so app.py can just call:
        retriever.retrieve(query)
    without worrying about the inner vector math.
    """

    def __init__(self):
        self.engine = VidExRetriever(use_mock=False)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K,
                 video_id: str | None = None) -> list[RetrievalResult]:
        raw = self.engine.search_multimodal_context(
            user_query=query, limit=top_k, video_id=video_id,
        )

        results = []
        for entry in raw:
            results.append(RetrievalResult(
                video_id=entry.get("video_id"),
                text=entry.get("text", ""),
                timestamp=float(entry.get("timestamp", 0.0)),
                text_score=entry.get("text_score"),
                visual_score=entry.get("visual_score"),
                combined_score=entry.get("combined_score"),
            ))
        return results


# Singleton — this is what app.py imports
retriever = RetrieverAdapter()