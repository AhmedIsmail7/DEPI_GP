import cv2
import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class SemanticVisionProcessor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- [Vision Engine] Loading CLIP model '{model_name}' on {self.device} ---")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        print("--- [Vision Engine] Model loaded successfully. ---")

    def get_text_embedding(self, text: str):
        """Encodes text using CLIP's text encoder to get 512-dim embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize to unit sphere for cosine similarity
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze(0) # Returns (512,)

    def get_frame_at_time(self, cap, timestamp_sec):
        """Helper to seek and extract a single frame."""
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ret, frame = cap.read()
        if ret:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None

    def process_video_blocks(self, video_path: str, transcript_chunks: list):
        """
        Main logic: Iterate over chunks, extract 5 candidate frames, 
        and select the best one based on CLIP text-image similarity.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        visual_metadata = []

        print(f"--- [Vision Engine] Starting Semantic Visual Alignment on {len(transcript_chunks)} chunks ---")

        for i, chunk in enumerate(transcript_chunks):
            start_time = chunk['start']
            end_time = chunk['end']
            
            # 1. Get Text Embedding (CLIP 512-dim)
            text_emb = self.get_text_embedding(chunk['text']).to(self.device)
            
            # 2. Define 5 sampling points within the chunk (every 6s approx relative to chunk)
            timestamps = np.linspace(start_time, end_time, 5)
            
            frames = []
            valid_timestamps = []
            
            for ts in timestamps:
                frame = self.get_frame_at_time(cap, ts)
                if frame:
                    frames.append(frame)
                    valid_timestamps.append(ts)
            
            if not frames:
                print(f"Chunk {i+1}: No frames extracted, skipping.")
                continue

            # 3. Extract Image Embeddings for all 5 frames (Batch)
            inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize image features to unit sphere
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            # 4. Calculate Cosine Similarity (Text vs 5 Images)
            # text_emb: (512,), image_features: (5, 512)
            similarities = F.cosine_similarity(image_features, text_emb.unsqueeze(0))
            
            # 5. Select Best
            best_idx = torch.argmax(similarities).item()
            best_frame_time = valid_timestamps[best_idx]
            best_score = similarities[best_idx].item()
            
            print(f"Chunk {i+1}: Best frame at {round(best_frame_time, 2)}s | Similarity: {best_score:.4f} | Text: '{chunk['text'][:30]}...'")
            
            # 6. Store result
            visual_metadata.append({
                "chunk_index": i,
                "timestamp": round(best_frame_time, 2),
                "embedding": image_features[best_idx].cpu().numpy().tolist(),
                "similarity_score": round(best_score, 4)
            })
            
        cap.release()
        print("--- [Vision Engine] Semantic Alignment complete. ---")
        return visual_metadata

# Singleton instance
vision_engine = SemanticVisionProcessor()

if __name__ == "__main__":
    # Test Integration
    try:
        if not os.path.exists("temp_assets/transcript_chunks.json"):
            print("Error: transcript_chunks.json not found. Run transcriber first.")
        else:
            with open("temp_assets/transcript_chunks.json", "r") as f:
                chunks = json.load(f)
                
            video_path = "temp_assets/yt_video.mp4"
            results = vision_engine.process_video_blocks(video_path, chunks)
            
            with open("temp_assets/visual_embeddings.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
                
            print("Visual pipeline complete. Data saved to temp_assets/visual_embeddings.json")
    except Exception as e:
        print(f"Vision error: {e}")