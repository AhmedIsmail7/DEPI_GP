import pytest
from types import SimpleNamespace
from modules.retrieval import fuse_search_results


def make_hit(id_, score, video_id="vid1", text="sample", timestamp=10.0):
    return SimpleNamespace(id=id_, score=score, payload={"video_id": video_id, "text": text, "timestamp": timestamp})


class TestFuseSearchResults:
    def test_point_in_both_modalities_combines_scores(self):
        text_hits = [make_hit("A", 0.9)]
        visual_hits = [make_hit("A", 0.5)]
        result = fuse_search_results(text_hits, visual_hits, text_weight=0.7, visual_weight=0.3)
        assert len(result) == 1
        assert result[0]["combined_score"] == pytest.approx(0.7 * 0.9 + 0.3 * 0.5)

    def test_text_only_hit_included_with_zero_visual_contribution(self):
        text_hits = [make_hit("A", 0.8)]
        visual_hits = []
        result = fuse_search_results(text_hits, visual_hits, text_weight=0.7, visual_weight=0.3)
        assert result[0]["combined_score"] == pytest.approx(0.7 * 0.8)
        assert result[0]["visual_score"] is None

    def test_visual_only_hit_included_with_zero_text_contribution(self):
        text_hits = []
        visual_hits = [make_hit("A", 0.6)]
        result = fuse_search_results(text_hits, visual_hits, text_weight=0.7, visual_weight=0.3)
        assert result[0]["combined_score"] == pytest.approx(0.3 * 0.6)
        assert result[0]["text_score"] is None

    def test_results_sorted_descending_by_combined_score(self):
        text_hits = [make_hit("A", 0.2), make_hit("B", 0.9)]
        visual_hits = []
        result = fuse_search_results(text_hits, visual_hits, text_weight=1.0, visual_weight=0.0)
        assert [r["payload"]["video_id"] for r in result] == ["vid1", "vid1"]  # sanity
        assert result[0]["combined_score"] > result[1]["combined_score"]

    def test_no_hits_returns_empty_list(self):
        result = fuse_search_results([], [], text_weight=0.7, visual_weight=0.3)
        assert result == []
