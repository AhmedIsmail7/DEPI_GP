import os

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    F = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover - optional dependency
    CLIPModel = None
    CLIPProcessor = None

from config import (
    CLIP_MODEL,
    NUM_FRAMES_PER_CHUNK,
    TEMP_DIR,
    TRANSCRIPT_OUTPUT,
    VISUAL_OUTPUT,
)

from modules.utils import (
    save_json,
    load_json,
    file_exists,
)


class SemanticVisionProcessor:
    def __init__(self, model_name: str = CLIP_MODEL):
        self.model_name = model_name
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

        print(f"CLIP model '{model_name}' will load on first use on {self.device}.")

    def _ensure_models_loaded(self):
        if self.model is not None and self.processor is not None:
            return

        if cv2 is None or np is None or torch is None or F is None or Image is None or CLIPModel is None or CLIPProcessor is None:
            raise ImportError("OpenCV, numpy, torch, PIL, and transformers are required for vision processing")

        print(f"Loading CLIP model '{self.model_name}' on {self.device}...")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        print("CLIP loaded successfully.")

    def get_text_embedding(self, text: str):
        self._ensure_models_loaded()

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
            embedding /= embedding.norm(dim=-1, keepdim=True)

        return embedding.squeeze(0)

    def get_frame_at_time(self, cap, timestamp_sec: float):
        if cv2 is None or Image is None:
            raise ImportError("OpenCV and PIL are required for frame extraction")

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        success, frame = cap.read()

        if not success:
            return None

        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def process_video_blocks(
        self,
        video_path: str,
        transcript_chunks: list,
    ) -> list:
        self._ensure_models_loaded()

        if not file_exists(video_path):
            raise FileNotFoundError(video_path)

        if cv2 is None:
            raise ImportError("OpenCV is required for vision processing")

        cap = cv2.VideoCapture(video_path)

        visual_metadata = []

        print(f"Processing {len(transcript_chunks)} transcript chunks...")

        try:
            for i, chunk in enumerate(transcript_chunks):
                text_embedding = self.get_text_embedding(chunk["text"])

                timestamps = np.linspace(
                    chunk["start"],
                    chunk["end"],
                    NUM_FRAMES_PER_CHUNK,
                )

                frames = []
                valid_timestamps = []

                for timestamp in timestamps:
                    frame = self.get_frame_at_time(cap, timestamp)
                    if frame is not None:
                        frames.append(frame)
                        valid_timestamps.append(timestamp)

                if not frames:
                    print(f"Chunk {i}: No valid frames.")
                    continue

                inputs = self.processor(images=frames, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    image_embeddings = self.model.get_image_features(**inputs)
                    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

                similarities = F.cosine_similarity(image_embeddings, text_embedding.unsqueeze(0))
                best_index = similarities.argmax().item()

                visual_metadata.append(
                    {
                        "chunk_index": i,
                        "timestamp": round(valid_timestamps[best_index], 2),
                        "embedding": image_embeddings[best_index].cpu().numpy().tolist(),
                        "similarity_score": round(similarities[best_index].item(), 4),
                    }
                )

                print(f"Chunk {i:03d} | Score = {similarities[best_index]:.4f}")

        finally:
            cap.release()

        print("Semantic alignment completed.")
        return visual_metadata


vision_engine = SemanticVisionProcessor()


if __name__ == "__main__":

    try:

        if not file_exists(TRANSCRIPT_OUTPUT):
            raise FileNotFoundError(TRANSCRIPT_OUTPUT)

        transcript = load_json(TRANSCRIPT_OUTPUT)

        videos = [
            file
            for file in os.listdir(TEMP_DIR)
            if file.endswith(".mp4")
        ]

        if not videos:
            raise FileNotFoundError("No video found.")

        video_path = os.path.join(TEMP_DIR, videos[0])

        results = vision_engine.process_video_blocks(video_path, transcript)

        save_json(results, VISUAL_OUTPUT)

        print(f"Results saved to {VISUAL_OUTPUT}")

    except Exception as e:
        print(f"Vision Error: {e}")