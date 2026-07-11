import json
from qdrant_client import QdrantClient, models

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME
from schemas import TranscriptChunk, VisualChunk, QdrantPoint, TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM


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
            # Required for retrieval.py's video_id filtering to work —
            # Qdrant needs an explicit index on any payload field used in filters.
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="video_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print("Created payload index on 'video_id'.")
        else:
            print(f"Collection {self.collection_name} already exists.")

    def upsert_data(self, transcript_chunks: list[TranscriptChunk], visual_chunks: list[VisualChunk]):
        """Merges transcript + visual data by chunk_index and uploads to Qdrant."""
        visual_by_index = {v.chunk_index: v for v in visual_chunks}

        points = []
        for chunk in transcript_chunks:
            vis = visual_by_index.get(chunk.index)
            if vis is None:
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
                id=qpoint.point_id(),  # deterministic hash of video_id + chunk_index
                vector={
                    "text_vector": chunk.embedding,
                    "image_vector": vis.embedding,
                },
                payload=qpoint.model_dump(exclude={"embedding"}, mode="json"),
            ))

        print(f"Uploading {len(points)} points to Qdrant...")
        self.client.upsert(collection_name=self.collection_name, points=points)
        print("Upload completed successfully.")

    def upsert_from_files(self, transcript_path: str, visual_path: str):
        """Convenience loader for running this module standalone / from JSON files."""
        with open(transcript_path, "r") as f:
            transcript_chunks = [TranscriptChunk(**c) for c in json.load(f)]
        with open(visual_path, "r") as f:
            visual_chunks = [VisualChunk(**v) for v in json.load(f)]
        self.upsert_data(transcript_chunks, visual_chunks)

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
    from config import TEMP_ASSETS_DIR
    import os
    db_manager.init_collection()
    db_manager.upsert_from_files(
        os.path.join(TEMP_ASSETS_DIR, "transcript_chunks.json"),
        os.path.join(TEMP_ASSETS_DIR, "visual_embeddings.json"),
    )