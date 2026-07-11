"""
Qdrant collection management — schema creation, dimension validation, and
resilient batch upsert with retry on transient failures.

Retry logic + pre-upload dimension validation adapted from a teammate's
more defensive storage.py: both are cheap insurance against wasting a
full GPU-minutes ingestion run (Whisper + SigLIP) on a transient network
error, or against silently trying to upload embeddings of the wrong
dimension into a collection created under an old schema (exactly the bug
class hit once already during the SigLIP migration).
"""

import time
import random
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    UPSERT_MAX_RETRIES, UPSERT_BASE_DELAY_SECONDS,
)
from schemas import TranscriptChunk, VisualChunk, QdrantPoint, TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM


@dataclass
class IndexingResult:
    """Summary returned from upsert_data() for logging/inspection."""
    points_attempted: int = 0
    points_uploaded: int = 0
    skipped_no_visual_match: int = 0
    retries_used: int = 0
    errors: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.points_uploaded > 0 and not self.errors


class DimensionMismatchError(Exception):
    """Raised when an existing collection's vector dimensions don't match
    what the current embedding model produces — e.g. if the collection was
    created under an old embedding model (like the pre-SigLIP MiniLM/CLIP
    setup) and never recreated after switching models."""
    pass


class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = QDRANT_COLLECTION_NAME

    def init_collection(self):
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text_vector": models.VectorParams(size=TEXT_EMBEDDING_DIM, distance=models.Distance.COSINE),
                    "image_vector": models.VectorParams(size=IMAGE_EMBEDDING_DIM, distance=models.Distance.COSINE),
                },
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="video_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print("Created payload index on 'video_id'.")
        else:
            self._validate_dimensions()
            print(f"Collection {self.collection_name} already exists.")

    def _validate_dimensions(self):
        """
        Confirms the existing collection's vector sizes match what the
        current embedding model produces. Catches the exact class of bug
        hit during the SigLIP migration — old collection, new embedding
        dimensions — with a clear error instead of a cryptic upload failure.
        """
        info = self.client.get_collection(self.collection_name)
        vectors_config = info.config.params.vectors

        existing_text_dim = vectors_config["text_vector"].size
        existing_image_dim = vectors_config["image_vector"].size

        if existing_text_dim != TEXT_EMBEDDING_DIM or existing_image_dim != IMAGE_EMBEDDING_DIM:
            raise DimensionMismatchError(
                f"Collection '{self.collection_name}' was created with "
                f"text_vector={existing_text_dim}, image_vector={existing_image_dim}, "
                f"but the current embedding model produces "
                f"text_vector={TEXT_EMBEDDING_DIM}, image_vector={IMAGE_EMBEDDING_DIM}. "
                f"The embedding model has likely changed since this collection was "
                f"created. Delete the collection and re-ingest, or point at a "
                f"different collection name."
            )

    def _upsert_with_retry(self, points: list[models.PointStruct]) -> int:
        """Uploads points with exponential backoff on transient failures.
        Returns the number of retries actually used."""
        retries_used = 0
        for attempt in range(UPSERT_MAX_RETRIES):
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
                return retries_used
            except Exception as e:
                retries_used += 1
                if attempt == UPSERT_MAX_RETRIES - 1:
                    raise
                delay = UPSERT_BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"[Database] Upsert failed ({e}), retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{UPSERT_MAX_RETRIES})...")
                time.sleep(delay)
        return retries_used  # unreachable, but keeps type-checkers happy

    def upsert_data(self, transcript_chunks: list[TranscriptChunk], visual_chunks: list[VisualChunk]) -> IndexingResult:
        """Merges transcript + visual data by chunk_index and uploads to Qdrant."""
        visual_by_index = {v.chunk_index: v for v in visual_chunks}
        result = IndexingResult()

        points = []
        for chunk in transcript_chunks:
            vis = visual_by_index.get(chunk.index)
            if vis is None:
                result.skipped_no_visual_match += 1
                continue

            qpoint = QdrantPoint(
                video_id=chunk.video_id,
                chunk_index=chunk.index,
                start=chunk.start,
                end=chunk.end,
                text=chunk.text,
                timestamp=vis.timestamp,
                similarity_score=vis.similarity_score,
            )

            points.append(models.PointStruct(
                id=qpoint.point_id(),
                vector={
                    "text_vector": chunk.embedding,
                    "image_vector": vis.embedding,
                },
                payload=qpoint.model_dump(exclude={"embedding"}, mode="json"),
            ))

        result.points_attempted = len(points)
        print(f"Uploading {len(points)} points to Qdrant...")

        try:
            result.retries_used = self._upsert_with_retry(points)
            result.points_uploaded = len(points)
            print(f"Upload completed successfully"
                  f"{f' after {result.retries_used} retr(ies)' if result.retries_used else ''}.")
        except Exception as e:
            result.errors.append(str(e))
            print(f"Upload FAILED after {UPSERT_MAX_RETRIES} attempts: {e}")
            raise

        return result

    def get_available_video_ids(self) -> list[str]:
        """Returns all distinct video_ids currently stored, for populating
        the query UI's video picker."""
        ids = set()
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["video_id"],
                limit=100,
                offset=next_offset,
            )
            for p in points:
                vid = p.payload.get("video_id")
                if vid:
                    ids.add(vid)
            if next_offset is None:
                break
        return sorted(ids)


db_manager = QdrantManager()

if __name__ == "__main__":
    import os
    from config import TEMP_ASSETS_DIR
    import json

    db_manager.init_collection()

    with open(os.path.join(TEMP_ASSETS_DIR, "transcript_chunks.json"), "r") as f:
        transcript_chunks = [TranscriptChunk(**c) for c in json.load(f)]
    with open(os.path.join(TEMP_ASSETS_DIR, "visual_embeddings.json"), "r") as f:
        visual_chunks = [VisualChunk(**v) for v in json.load(f)]

    result = db_manager.upsert_data(transcript_chunks, visual_chunks)
    print(f"Result: {result}")