import os
import json
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from config import CLIP_MODEL_NAME, TEMP_ASSETS_DIR
from schemas import TranscriptChunk, VisualChunk


class SemanticVisionProcessor:
    def __init__(self, model_name=CLIP_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    @property
    def model(self):
        if self._model is None:
            print(f"Loading CLIP model '{self.model_name}' on {self.device}...")
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor

    def get_text_embedding(self, text: str):
        """Encodes text using CLIP's text encoder to get 512-dim embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model.get_text_features(**inputs)
            # transformers 5.x returns BaseModelOutputWithPooling instead of a
            # plain tensor (breaking change vs 4.x). .pooler_output still holds
            # the fully projected 512-dim embedding either way — confirmed
            # against the transformers source.
            text_features = output.pooler_output if hasattr(output, "pooler_output") else output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze(0)

    def get_frame_at_time(self, cap, timestamp_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ret, frame = cap.read()
        if ret:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None

    def process_video_blocks(
            self, video_path: str, video_id: str, transcript_chunks: list[TranscriptChunk],
        ) -> list[VisualChunk]:
            """Iterates over transcript chunks, extracts 5 candidate frames per
            chunk window, and picks the one most semantically aligned with the
            chunk's text via CLIP similarity."""
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            cap = cv2.VideoCapture(video_path)
            results: list[VisualChunk] = []

            print(f"--- [Vision Engine] Aligning {len(transcript_chunks)} chunks for video_id={video_id} ---")

            for chunk in transcript_chunks:
                text_emb = self.get_text_embedding(chunk.text).to(self.device)
                timestamps = np.linspace(chunk.start, chunk.end, 5)

                frames, valid_timestamps = [], []
                for ts in timestamps:
                    frame = self.get_frame_at_time(cap, ts)
                    if frame:
                        frames.append(frame)
                        valid_timestamps.append(ts)

                if not frames:
                    print(f"Chunk {chunk.index}: No frames extracted, skipping.")
                    continue

                inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.model.get_image_features(**inputs)
                    image_features = output.pooler_output if hasattr(output, "pooler_output") else output
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarities = F.cosine_similarity(image_features, text_emb.unsqueeze(0))
                best_idx = torch.argmax(similarities).item()

                results.append(VisualChunk(
                    video_id=video_id,
                    chunk_index=chunk.index,
                    timestamp=round(valid_timestamps[best_idx], 2),
                    embedding=image_features[best_idx].cpu().numpy().tolist(),
                    similarity_score=round(similarities[best_idx].item(), 4),
                ))

                print(f"Chunk {chunk.index}: best frame @ {round(valid_timestamps[best_idx], 2)}s "
                    f"| sim: {similarities[best_idx].item():.4f}")

            cap.release()
            return results


vision_engine = SemanticVisionProcessor()

if __name__ == "__main__":
    try:
        transcript_path = os.path.join(TEMP_ASSETS_DIR, "transcript_chunks.json")
        if not os.path.exists(transcript_path):
            print("Error: transcript_chunks.json not found. Run transcriber first.")
        else:
            with open(transcript_path, "r") as f:
                raw_chunks = json.load(f)
            chunks = [TranscriptChunk(**c) for c in raw_chunks]
            video_id = chunks[0].video_id

            video_path = os.path.join(TEMP_ASSETS_DIR, f"{video_id}.mp4")
            results = vision_engine.process_video_blocks(video_path, video_id, chunks)

            out_path = os.path.join(TEMP_ASSETS_DIR, "visual_embeddings.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in results], f, indent=4)

            print(f"Visual pipeline complete. Data saved to {out_path}")
    except Exception as e:
        print(f"Vision error: {e}")