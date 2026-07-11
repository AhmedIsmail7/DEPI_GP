"""
Shared data contracts for VidEx.
Every module that produces or consumes chunk/vector data should import
from here rather than passing around raw dicts. This is what prevents
silent field-mismatch bugs (e.g. a missing video_id) between modules
built by different people.
"""

from pydantic import BaseModel, field_validator
from typing import Optional
import uuid


TEXT_EMBEDDING_DIM = 384   # all-MiniLM-L6-v2
IMAGE_EMBEDDING_DIM = 512  # CLIP ViT-B/32


class TranscriptChunk(BaseModel):
    """Output of transcribe.py — one sliding-window audio segment."""
    video_id: str
    index: int
    start: float          # seconds
    end: float             # seconds
    text: str
    embedding: list[float]

    @field_validator("embedding")
    @classmethod
    def check_text_embedding_dim(cls, v):
        if len(v) != TEXT_EMBEDDING_DIM:
            raise ValueError(
                f"Expected {TEXT_EMBEDDING_DIM}-dim text embedding, got {len(v)}"
            )
        return v

    @field_validator("end")
    @classmethod
    def check_end_after_start(cls, v, info):
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"end ({v}) must be after start ({start})")
        return v


class VisualChunk(BaseModel):
    """Output of vision.py — best-matching frame for a transcript chunk."""
    video_id: str
    chunk_index: int       # must match TranscriptChunk.index
    timestamp: float        # seconds, absolute position in video
    embedding: list[float]
    similarity_score: float

    @field_validator("embedding")
    @classmethod
    def check_image_embedding_dim(cls, v):
        if len(v) != IMAGE_EMBEDDING_DIM:
            raise ValueError(
                f"Expected {IMAGE_EMBEDDING_DIM}-dim image embedding, got {len(v)}"
            )
        return v

    @field_validator("similarity_score")
    @classmethod
    def check_score_range(cls, v):
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"Cosine similarity out of range: {v}")
        return v



# Fixed namespace UUID for VidEx — any valid UUID works here, it just needs
# to stay constant across the project so the same (video_id, chunk_index)
# always maps to the same point ID, run after run.
VIDEX_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


class QdrantPoint(BaseModel):
    """What actually gets upserted to Qdrant — merged text + visual data."""
    video_id: str
    chunk_index: int
    start: float
    end: float
    text: str
    timestamp: float
    similarity_score: float

    def point_id(self) -> str:
        """
        Deterministic, Qdrant-valid UUID across videos.
        uuid5 with a fixed namespace guarantees the same
        (video_id, chunk_index) pair always produces the same UUID,
        so re-ingesting a video overwrites its own points instead of
        colliding with a different video's points at the same index.
        """
        raw = f"{self.video_id}_{self.chunk_index}"
        return str(uuid.uuid5(VIDEX_NAMESPACE, raw))


class RetrievalResult(BaseModel):
    """Output of retrieval.py after dual-modality search + fusion ranking."""
    video_id: str
    text: str
    timestamp: float
    text_score: Optional[float] = None
    visual_score: Optional[float] = None
    combined_score: float


class LLMAnswer(BaseModel):
    """Final response returned to app.py."""
    answer: str
    source_timestamps: list[float]
    video_id: str