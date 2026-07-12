"""
Tests for the retrieval fusion logic.

Uses mock Qdrant hits (SimpleNamespace objects that mimic ScoredPoint)
to verify that the VidExRetriever's search_multimodal_context correctly
merges text and visual results by timestamp.
"""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


def make_hit(score, video_id="vid1", text="sample text", timestamp=10.0):
    """Fakes a Qdrant ScoredPoint with the payload shape database.py produces."""
    return SimpleNamespace(
        score=score,
        payload={"video_id": video_id, "text": text, "timestamp": timestamp},
    )


class TestMultimodalFusion:
    """Tests for the dual-search fusion in VidExRetriever."""

    def _make_retriever(self, text_hits, visual_hits):
        """
        Creates a VidExRetriever in non-mock mode but patches the Qdrant
        client so we can inject fake search results without a live cluster.
        """
        from modules.retrieval import VidExRetriever

        retriever = VidExRetriever.__new__(VidExRetriever)
        retriever.use_mock = False
        retriever.collection_name = "test_collection"
        retriever.client = MagicMock()

        # First call = text search, second call = visual search
        retriever.client.search.side_effect = [text_hits, visual_hits]

        return retriever

    @patch("modules.embeddings.embedding_manager")
    def test_same_chunk_in_both_modalities_gets_merged(self, mock_emb):
        mock_emb.get_text_embedding.return_value = [0.0] * 768

        text_hits = [make_hit(0.9, timestamp=120.0)]
        visual_hits = [make_hit(0.5, timestamp=120.0)]
        retriever = self._make_retriever(text_hits, visual_hits)

        results = retriever.search_multimodal_context("test query", limit=5)

        # Same timestamp → should be merged into one result
        assert len(results) == 1
        assert results[0]["text_score"] == 0.9
        assert results[0]["visual_score"] == 0.5
        # combined = 0.7 * 0.9 + 0.3 * 0.5 = 0.78
        assert results[0]["combined_score"] == pytest.approx(0.78)

    @patch("modules.embeddings.embedding_manager")
    def test_text_only_hit_gets_zero_visual(self, mock_emb):
        mock_emb.get_text_embedding.return_value = [0.0] * 768

        text_hits = [make_hit(0.8, timestamp=50.0)]
        visual_hits = []
        retriever = self._make_retriever(text_hits, visual_hits)

        results = retriever.search_multimodal_context("test query", limit=5)

        assert len(results) == 1
        assert results[0]["text_score"] == 0.8
        assert results[0]["visual_score"] is None
        # combined = 0.7 * 0.8 + 0.3 * 0 = 0.56
        assert results[0]["combined_score"] == pytest.approx(0.56)

    @patch("modules.embeddings.embedding_manager")
    def test_visual_only_hit_gets_zero_text(self, mock_emb):
        mock_emb.get_text_embedding.return_value = [0.0] * 768

        text_hits = []
        visual_hits = [make_hit(0.6, timestamp=200.0)]
        retriever = self._make_retriever(text_hits, visual_hits)

        results = retriever.search_multimodal_context("test query", limit=5)

        assert len(results) == 1
        assert results[0]["text_score"] is None
        assert results[0]["visual_score"] == 0.6
        # combined = 0.7 * 0 + 0.3 * 0.6 = 0.18
        assert results[0]["combined_score"] == pytest.approx(0.18)

    @patch("modules.embeddings.embedding_manager")
    def test_results_sorted_by_combined_score_descending(self, mock_emb):
        mock_emb.get_text_embedding.return_value = [0.0] * 768

        text_hits = [
            make_hit(0.3, timestamp=10.0),
            make_hit(0.95, timestamp=50.0),
        ]
        visual_hits = []
        retriever = self._make_retriever(text_hits, visual_hits)

        results = retriever.search_multimodal_context("test query", limit=5)

        assert len(results) == 2
        assert results[0]["combined_score"] > results[1]["combined_score"]
        assert results[0]["timestamp"] == 50.0

    @patch("modules.embeddings.embedding_manager")
    def test_no_hits_returns_empty(self, mock_emb):
        mock_emb.get_text_embedding.return_value = [0.0] * 768

        retriever = self._make_retriever([], [])
        results = retriever.search_multimodal_context("test query", limit=5)

        assert results == []

    @patch("modules.embeddings.embedding_manager")
    def test_limit_is_respected(self, mock_emb):
        mock_emb.get_text_embedding.return_value = [0.0] * 768

        text_hits = [
            make_hit(0.9, timestamp=10.0),
            make_hit(0.8, timestamp=20.0),
            make_hit(0.7, timestamp=30.0),
            make_hit(0.6, timestamp=40.0),
        ]
        retriever = self._make_retriever(text_hits, [])

        results = retriever.search_multimodal_context("test query", limit=2)
        assert len(results) == 2
