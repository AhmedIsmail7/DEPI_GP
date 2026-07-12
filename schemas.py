"""
Shared data contracts for VidEx.
Everyone import from here instead of passing around raw dicts. 
This is how we avoid annoying bugs where someone forgets a video_id field.
"""

from pydantic import BaseModel, field_validator
from typing import Optional
import uuid


TEXT_EMBEDDING_DIM = 768   # We use SigLIP now, so text and image are both 768
IMAGE_EMBEDDING_DIM = 768  


class TranscriptChunk(BaseModel):
    """What comes out of our audio transcription — one chunk of speech."""
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
            raise ValueError(f"Bro, text embedding should be {TEXT_EMBEDDING_DIM}-dim, but got {len(v)}")
        return v

    @field_validator("end")
    @classmethod
    def check_end_after_start(cls, v, info):
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"End time ({v}) has to be after start time ({start}), obviously")
        return v


class VisualChunk(BaseModel):
    """The best-matching frame that goes with a specific transcript chunk."""
    video_id: str
    chunk_index: int       # Make sure this matches TranscriptChunk.index!
    timestamp: float        # seconds, exact spot in the video
    embedding: list[float]
    similarity_score: float

    @field_validator("embedding")
    @classmethod
    def check_image_embedding_dim(cls, v):
        if len(v) != IMAGE_EMBEDDING_DIM:
            raise ValueError(f"Image embedding needs to be {IMAGE_EMBEDDING_DIM}-dim, got {len(v)}")
        return v

    @field_validator("similarity_score")
    @classmethod
    def check_score_range(cls, v):
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"Cosine similarity is broken: {v}")
        return v



# Fixed UUID so that generating the same video chunk gives us the same ID every time.
VIDEX_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


class QdrantPoint(BaseModel):
    """This is the final package we ship off to Qdrant — text + visual combined."""
    video_id: str
    chunk_index: int
    start: float
    end: float
    text: str
    timestamp: float
    similarity_score: float

    def point_id(self) -> str:
        """
        Gives us a reliable ID for Qdrant.
        If we re-ingest a video, this ensures we overwrite the old data 
        instead of making duplicates.
        """
        raw = f"{self.video_id}_{self.chunk_index}_{self.timestamp}"
        return str(uuid.uuid5(VIDEX_NAMESPACE, raw))


class RetrievalResult(BaseModel):
    """What the retriever spits back out after finding the best matches."""
    video_id: Optional[str] = None
    text: str
    timestamp: float
    text_score: Optional[float] = None
    visual_score: Optional[float] = None
    combined_score: Optional[float] = None
    image_path: Optional[str] = None


class LLMAnswer(BaseModel):
    """The final payload we send back to the frontend UI."""
    answer: str
    source_timestamps: list[float]
    video_id: Optional[str] = None
    new_summary: Optional[str] = None