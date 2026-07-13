"""
Multimodal Retrieval Module.

Executes parallel searches against both text and visual vector indices in Qdrant.
Results are deduplicated and merged by timestamp to provide the LLM with 
a unified contextual context window.
"""

from qdrant_client import QdrantClient, models

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, DEFAULT_TOP_K
from schemas import RetrievalResult


# Hyperparameters for score fusion.
# Text provides primary narrative context; visual catches non-verbal data.
TEXT_WEIGHT = 0.7
VISUAL_WEIGHT = 0.3


class VidExRetriever:
    """
    Interface for executing vector database queries and fusing multimodal results.
    Includes an optional mock mode for isolated testing.
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
        """Generates a Qdrant filter to isolate searches to a specific video identifier."""
        if video_id is None:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
        )

    def _search_vector(self, vector_name: str, query_vector: list[float],
                       limit: int, video_id: str | None) -> list:
        """Executes a similarity search against a targeted vector index."""
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
        Executes parallel searches across textual and visual vector spaces.
        Results are deduplicated via timestamp clustering to prevent redundant 
        context from consuming LLM token limits.
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

        # Generate a unified embedding query. Both indices share the same vector space.
        # Lazy-load the embedding manager to optimize application startup time.
        from modules.embeddings import embedding_manager
        query_vector = embedding_manager.get_text_embedding(user_query)

        # Fetch extra candidates since timestamp fusion will deduplicate results
        fetch_k = max(limit * 2, 6)
        text_hits = self._search_vector("text_vector", query_vector, fetch_k, video_id)
        visual_hits = self._search_vector("image_vector", query_vector, fetch_k, video_id)

        # --- Fusion time (Reciprocal Rank Fusion) ---
        candidates = {} # timestamp -> payload

        # 1. Process text hits (rank by chunk_index to avoid redundant frames flooding text ranks)
        text_ranks = {} # chunk_index -> (rank, score)
        t_rank = 0
        seen_chunks = set()
        for hit in text_hits:
            c_idx = hit.payload.get("chunk_index")
            if c_idx not in seen_chunks:
                text_ranks[c_idx] = (t_rank, hit.score)
                seen_chunks.add(c_idx)
                t_rank += 1
            ts = hit.payload.get("timestamp", 0.0)
            if ts not in candidates:
                candidates[ts] = hit.payload

        # 2. Process visual hits (rank directly by timestamp)
        visual_ranks = {} # timestamp -> (rank, score)
        for rank, hit in enumerate(visual_hits):
            ts = hit.payload.get("timestamp", 0.0)
            visual_ranks[ts] = (rank, hit.score)
            if ts not in candidates:
                candidates[ts] = hit.payload

        # 3. Combine scores
        fused = []
        for ts, payload in candidates.items():
            c_idx = payload.get("chunk_index")
            t_data = text_ranks.get(c_idx)
            v_data = visual_ranks.get(ts)

            rrf = 0.0
            t_score, v_score = None, None

            if t_data is not None:
                t_score = t_data[1]
                rrf += 1.0 / (60 + t_data[0])
            if v_data is not None:
                v_score = v_data[1]
                rrf += 1.0 / (60 + v_data[0])

            fused.append({
                "text": payload.get("text", ""),
                "timestamp": ts,
                "video_id": payload.get("video_id", video_id),
                "chunk_index": c_idx,
                "text_score": t_score,
                "visual_score": v_score,
                "combined_score": rrf # Use combined_score key to avoid changing downstream code
            })

        # 4. Deduplicate by chunk_index, keeping the timestamp with the highest RRF score
        best_per_chunk = {}
        for entry in fused:
            c_idx = entry.get("chunk_index")
            # If we already have a hit for this chunk, keep the one with the higher RRF score
            if c_idx not in best_per_chunk or entry["combined_score"] > best_per_chunk[c_idx]["combined_score"]:
                best_per_chunk[c_idx] = entry

        results = sorted(best_per_chunk.values(), key=lambda x: x["combined_score"], reverse=True)
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