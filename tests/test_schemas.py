import pytest
from pydantic import ValidationError
from schemas import TranscriptChunk, VisualChunk, QdrantPoint, TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM


class TestTranscriptChunk:
    def _valid_kwargs(self, **overrides):
        base = dict(
            video_id="vid123", index=0, start=0.0, end=30.0,
            text="hello world", embedding=[0.1] * TEXT_EMBEDDING_DIM,
        )
        base.update(overrides)
        return base

    def test_valid_chunk_passes(self):
        chunk = TranscriptChunk(**self._valid_kwargs())
        assert chunk.video_id == "vid123"

    def test_wrong_embedding_dim_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptChunk(**self._valid_kwargs(embedding=[0.1] * 100))

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptChunk(**self._valid_kwargs(start=30.0, end=10.0))

    def test_end_equal_start_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptChunk(**self._valid_kwargs(start=10.0, end=10.0))


class TestVisualChunk:
    def _valid_kwargs(self, **overrides):
        base = dict(
            video_id="vid123", chunk_index=0, timestamp=15.0,
            embedding=[0.1] * IMAGE_EMBEDDING_DIM, similarity_score=0.28,
        )
        base.update(overrides)
        return base

    def test_valid_chunk_passes(self):
        chunk = VisualChunk(**self._valid_kwargs())
        assert chunk.chunk_index == 0

    def test_wrong_embedding_dim_rejected(self):
        with pytest.raises(ValidationError):
            VisualChunk(**self._valid_kwargs(embedding=[0.1] * 512 + [0.1]))  # 513-dim

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            VisualChunk(**self._valid_kwargs(similarity_score=1.5))

    def test_score_at_boundary_accepted(self):
        chunk = VisualChunk(**self._valid_kwargs(similarity_score=1.0))
        assert chunk.similarity_score == 1.0


class TestQdrantPointId:
    def _make_point(self, video_id="vid1", chunk_index=0):
        return QdrantPoint(
            video_id=video_id, chunk_index=chunk_index, start=0.0, end=30.0,
            text="x", timestamp=15.0, similarity_score=0.3,
        )

    def test_same_inputs_produce_same_id(self):
        p1 = self._make_point(video_id="vid1", chunk_index=5)
        p2 = self._make_point(video_id="vid1", chunk_index=5)
        assert p1.point_id() == p2.point_id()

    def test_different_video_ids_produce_different_ids(self):
        p1 = self._make_point(video_id="vid1", chunk_index=0)
        p2 = self._make_point(video_id="vid2", chunk_index=0)
        assert p1.point_id() != p2.point_id()
        # This is the exact bug we hit and fixed early in the project —
        # cross-video point collisions. Guard against regression.

    def test_different_chunk_indices_produce_different_ids(self):
        p1 = self._make_point(video_id="vid1", chunk_index=0)
        p2 = self._make_point(video_id="vid1", chunk_index=1)
        assert p1.point_id() != p2.point_id()

    def test_id_is_valid_uuid_format(self):
        import uuid
        point_id = self._make_point().point_id()
        uuid.UUID(point_id)  # raises ValueError if not a valid UUID string