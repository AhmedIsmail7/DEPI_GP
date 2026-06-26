import cv2
import torch
import json
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class VisionProcessor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialization for Visual Processing with CLIP.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        print("Vision model loaded successfully.")

    def extract_and_embed(self, video_path: str, interval_seconds: int = 15):
        """
        Extracts frames at specific intervals and generates visual embeddings.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        
        visual_metadata = []
        frame_count = 0

        print(f"Starting visual extraction (Interval: {interval_seconds}s)...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # Convert OpenCV BGR to PIL RGB
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # CLIP Processing
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    # Normalize features (Standard practice for CLIP)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Store data
                visual_metadata.append({
                    "timestamp": round(timestamp, 2),
                    "embedding": image_features.cpu().numpy().flatten().tolist()
                })
                
                print(f"Processed frame at {round(timestamp, 2)}s")
            
            frame_count += 1
            
        cap.release()
        return visual_metadata

# Singleton instance
vision_engine = VisionProcessor()


# For only trying the module directly (not important for the main pipeline & not for import)
if __name__ == "__main__":
    # Test Integration
    try:
        video_path = "temp_assets/yt_video.mp4"
        results = vision_engine.extract_and_embed(video_path)
        
        # Save output
        with open("temp_assets/visual_embeddings.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        print("Visual pipeline complete. Data saved to temp_assets/visual_embeddings.json")
    except Exception as e:
        print(f"Vision error: {e}")