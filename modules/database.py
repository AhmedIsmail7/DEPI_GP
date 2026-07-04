import json
import os

try:
    from qdrant_client import QdrantClient, models
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None
    models = None

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    TEXT_VECTOR_SIZE,
    IMAGE_VECTOR_SIZE,
    TRANSCRIPT_OUTPUT,
    VISUAL_OUTPUT,
)


class QdrantManager:

    def __init__(self):
        if QdrantClient is None or models is None:
            raise ImportError("qdrant-client is required for database operations")

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        self.collection_name = COLLECTION_NAME

    def init_collection(self):
        """
        Create the Qdrant collection if it doesn't already exist.
        """

        if self.client.collection_exists(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
            return

        print(f"Creating collection '{self.collection_name}'...")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text_vector": models.VectorParams(
                    size=TEXT_VECTOR_SIZE,
                    distance=models.Distance.COSINE
                ),
                "image_vector": models.VectorParams(
                    size=IMAGE_VECTOR_SIZE,
                    distance=models.Distance.COSINE
                ),
            },
        )

        print("Collection created successfully.")

    def upsert_data(
        self,
        transcript_path=TRANSCRIPT_OUTPUT,
        visual_path=VISUAL_OUTPUT
    ):
        """
        Merge transcript and visual embeddings,
        then upload them to Qdrant.
        """

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        with open(visual_path, "r", encoding="utf-8") as f:
            visual = json.load(f)

        visual_lookup = {
            item["chunk_index"]: item
            for item in visual
        }

        points = []

        for i, chunk in enumerate(transcript):

            vis_data = visual_lookup.get(i)

            if vis_data is None:
                continue

            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        "text_vector": chunk["embedding"],
                        "image_vector": vis_data["embedding"],
                    },
                    payload={
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "text": chunk["text"],
                        "timestamp": vis_data["timestamp"],
                        "similarity_score": vis_data["similarity_score"],
                    },
                )
            )

        print(f"Uploading {len(points)} points...")

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print("Upload completed successfully.")


def export_results_to_json(
    transcript_path=TRANSCRIPT_OUTPUT,
    visual_path=VISUAL_OUTPUT,
    output_path=None,
):
    """
    Create a JSON export under Video_result containing the combined
    transcript and visual metadata for the uploaded Qdrant records.
    """
    if output_path is None:
        output_path = os.path.join("Video_result", "qdrant_results.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    with open(visual_path, "r", encoding="utf-8") as f:
        visual = json.load(f)

    visual_lookup = {item.get("chunk_index"): item for item in visual}
    combined = []

    for i, chunk in enumerate(transcript):
        vis_data = visual_lookup.get(i)
        if vis_data is None:
            continue

        combined.append({
            "id": i,
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "text": chunk.get("text"),
            "timestamp": vis_data.get("timestamp"),
            "similarity_score": vis_data.get("similarity_score"),
            "source": "qdrant",
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    print(f"Qdrant JSON export saved to: {output_path}")
    return output_path


db_manager = QdrantManager()


if __name__ == "__main__":

    db_manager.init_collection()
    db_manager.upsert_data()