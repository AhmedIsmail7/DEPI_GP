import os
import json
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# Load env variables
load_dotenv()

class QdrantManager:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.collection_name = "vedex_knowledge"

    def init_collection(self):
        """Creates the collection with Named Vectors."""
        # Check if collection exists
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text_vector": models.VectorParams(size=384, distance=models.Distance.COSINE),
                    "image_vector": models.VectorParams(size=512, distance=models.Distance.COSINE),
                }
            )
        else:
            print(f"Collection {self.collection_name} already exists.")

    def upsert_data(self, transcript_path, visual_path):
        """Merges and uploads data to Qdrant."""
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
        with open(visual_path, 'r') as f:
            visual = json.load(f)

        points = []
        for i, chunk in enumerate(transcript):
            # Find the corresponding visual embedding (matching by index or timestamp)
            vis_data = next((item for item in visual if item['chunk_index'] == i), None)
            
            if vis_data:
                point = models.PointStruct(
                    id=i,
                    vector={
                        "text_vector": chunk['embedding'], # 384 dim
                        "image_vector": vis_data['embedding'] # 512 dim
                    },
                    payload={
                        "start": chunk['start'],
                        "end": chunk['end'],
                        "text": chunk['text'],
                        "timestamp": vis_data['timestamp'],
                        "similarity_score": vis_data['similarity_score']
                    }
                )
                points.append(point)

        print(f"Uploading {len(points)} points to Qdrant...")
        self.client.upsert(collection_name=self.collection_name, points=points)
        print("Upload completed successfully.")

# Singleton
db_manager = QdrantManager()

if __name__ == "__main__":
    db_manager.init_collection()
    db_manager.upsert_data("temp_assets/transcript_chunks.json", "temp_assets/visual_embeddings.json")