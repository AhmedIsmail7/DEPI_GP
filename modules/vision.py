# modules/vision2.py
import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import easyocr
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ----------------- Configuration & Logger -----------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("vedex.vision")

# thresholds
BLACK_FRAME_THRESHOLD = 15
BLUR_THRESHOLD = 100.0
DUPLICATE_FRAME_THRESHOLD = 0.95

# weights
SIMILARITY_WEIGHT = 0.4
OCR_WEIGHT = 0.4
QUALITY_WEIGHT = 0.2

# Load EasyOCR
try:
    use_gpu = torch.cuda.is_available()
    ocr_reader = easyocr.Reader(["en", "ar"], gpu=use_gpu)
    logger.info("EasyOCR initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    ocr_reader = None

# ----------------- Vision Processor Class -----------------
class VisionProcessor:
    def __init__(self, siglip_model="google/siglip-base-patch16-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize SigLIP natively for Embeddings
        logger.info(f"Loading SigLIP Vision & Text Encoders '{siglip_model}'...")
        self.processor = AutoProcessor.from_pretrained(siglip_model)
        self.model = AutoModel.from_pretrained(siglip_model).to(self.device).eval()
        
        self.cache_dir = Path("temp_assets/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ocr_cache_path = self.cache_dir / "ocr_cache.json"
        self.ocr_cache = self._load_ocr_cache()
        self.ocr_cache_dirty = False
        
        self.stats = {
            "total_sampled_frames": 0,
            "filtered_frames": 0,
            "ocr_cache_hits": 0,
            "selected_keyframes": 0,
        }

    # --- Embedded SigLIP Generation Methods ---
    def get_text_embedding(self, text: str) -> list:
        max_len = getattr(self.model.config, "max_position_embeddings", 64)
        inputs = self.processor(text=[text], padding="max_length", max_length=max_len, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def get_image_embedding(self, cv2_frame: np.ndarray) -> list:
        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def cosine_similarity(self, vec1: list, vec2: list) -> float:
        return float(np.dot(vec1, vec2))

    # --- OCR & Quality Methods ---
    def _load_ocr_cache(self) -> dict:
        if self.ocr_cache_path.exists():
            try:
                with open(self.ocr_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def flush_ocr_cache(self):
        if self.ocr_cache_dirty:
            with open(self.ocr_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.ocr_cache, f, ensure_ascii=False)
            self.ocr_cache_dirty = False

    def calculate_phash(self, image: np.ndarray) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_8x8 = dct[0:8, 0:8]
        median = np.median(dct_8x8)
        return "".join(["1" if b else "0" for b in (dct_8x8 > median).flatten()])

    def phash_similarity(self, hash1: str, hash2: str) -> float:
        if not hash1 or not hash2: return 0.0
        arr1 = np.frombuffer(hash1.encode('ascii'), dtype=np.uint8)
        arr2 = np.frombuffer(hash2.encode('ascii'), dtype=np.uint8)
        return 1.0 - (np.sum(arr1 != arr2) / len(hash1))

    def evaluate_quality(self, frame: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        brightness_score = max(0.0, min(1.0, 1.0 - abs(mean_brightness - 127.5) / 127.5))
        
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = sharpness / (sharpness + 100.0)
        
        return {
            "quality": 0.5 * brightness_score + 0.5 * sharpness_score,
            "brightness": mean_brightness,
            "sharpness": sharpness,
            "entropy": 0.0 
        }

    def process_ocr_for_frame(self, frame: np.ndarray) -> Tuple[str, dict]:
        h = hashlib.sha256(frame.tobytes()).hexdigest()
        if h in self.ocr_cache:
            self.stats["ocr_cache_hits"] += 1
            return h, self.ocr_cache[h]

        if not ocr_reader:
            return h, {"text": "", "confidence": 0.0}

        try:
            results = ocr_reader.readtext(frame)
            if results:
                texts = [res[1] for res in results if res[1]]
                confidences = [res[2] for res in results]
                data = {"text": " ".join(texts).strip(), "confidence": float(np.mean(confidences))}
            else:
                data = {"text": "", "confidence": 0.0}
        except Exception:
            data = {"text": "", "confidence": 0.0}

        self.ocr_cache[h] = data
        self.ocr_cache_dirty = True
        return h, data

    # --- Main Pipeline ---
    def process_video_blocks(self, video_path: str, transcript_chunks: List[Dict[str, Any]]):
        video_p = Path(video_path)
        cap = cv2.VideoCapture(str(video_p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        cap.release()

        keyframes_dir = Path("temp_assets/keyframes")
        keyframes_dir.mkdir(parents=True, exist_ok=True)

        visual_embeddings = []
        previous_keyframe_hash = None
        sample_count = 5

        logger.info(f"Starting Vision Pipeline for {len(transcript_chunks)} chunks...")

        for idx, chunk in enumerate(transcript_chunks):
            chunk_idx = chunk.get("chunk_index", idx)
            start_t = chunk.get("start_time", 0.0)
            end_t = chunk.get("end_time", duration_sec)
            transcript = chunk.get("text", "")

            timestamps = np.linspace(start_t, end_t, num=sample_count)
            candidates = []
            
            cap = cv2.VideoCapture(str(video_p))
            for ts in timestamps:
                frame_number = min(int(ts * fps), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret: continue

                self.stats["total_sampled_frames"] += 1
                
                quality_data = self.evaluate_quality(frame)
                current_hash = self.calculate_phash(frame)
                
                if quality_data["brightness"] < BLACK_FRAME_THRESHOLD or quality_data["sharpness"] < BLUR_THRESHOLD:
                    self.stats["filtered_frames"] += 1
                    continue
                    
                if previous_keyframe_hash and self.phash_similarity(current_hash, previous_keyframe_hash) > DUPLICATE_FRAME_THRESHOLD:
                    self.stats["filtered_frames"] += 1
                    continue

                candidates.append({
                    "frame": frame,
                    "frame_time": round(float(ts), 2),
                    "pHash": current_hash,
                    "quality_score": quality_data["quality"]
                })
            cap.release()

            if not candidates:
                logger.warning(f"Chunk {chunk_idx}: All frames filtered. Skipping visually.")
                continue

            for cand in candidates:
                _, ocr_data = self.process_ocr_for_frame(cand["frame"])
                cand["ocr_text"] = ocr_data.get("text", "")
                
                semantic_text = f"Transcript: {transcript} | OCR: {cand['ocr_text']}" if cand['ocr_text'] else transcript
                
                cand["text_emb"] = self.get_text_embedding(semantic_text)
                cand["img_emb"] = self.get_image_embedding(cand["frame"])
                cand["sim_score"] = self.cosine_similarity(cand["text_emb"], cand["img_emb"])
                
                ocr_bonus = 0.2 if cand["ocr_text"] else 0.0
                cand["ranking_score"] = (SIMILARITY_WEIGHT * cand["sim_score"]) + (QUALITY_WEIGHT * cand["quality_score"]) + ocr_bonus

            candidates.sort(key=lambda x: x["ranking_score"], reverse=True)
            winner = candidates[0]

            frame_path = keyframes_dir / f"chunk_{chunk_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), winner["frame"])
            previous_keyframe_hash = winner["pHash"]
            self.stats["selected_keyframes"] += 1

            visual_embeddings.append({
                "chunk_index": chunk_idx,
                "start_time": start_t,
                "end_time": end_t,
                "timestamp": winner["frame_time"],
                "frame_path": str(frame_path),
                "ocr_text": winner["ocr_text"],
                "image_embedding": winner["img_emb"],
                "text_embedding": winner["text_emb"],
                "similarity_score": round(winner["sim_score"], 4)
            })
            
            logger.info(f"[Success] Chunk {chunk_idx} -> Selected Frame at {winner['frame_time']}s (Sim: {winner['sim_score']:.3f})")

        self.flush_ocr_cache()
        
        output_file = Path("temp_assets/visual_embeddings.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(visual_embeddings, f, ensure_ascii=False, indent=4)
            
        print(f"\n--- [Complete] Vision Pipeline Finished. Keyframes saved to {keyframes_dir} ---")

# ----------------- Execution -----------------
if __name__ == "__main__":
    try:
        folder = "temp_assets"
        video_files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.mkv', '.avi'))]
        
        transcripts_file = os.path.join(folder, "siglip_text_embeddings.json")
        
        if not video_files:
            raise FileNotFoundError("No video files found in temp_assets/ directory.")
        if not os.path.exists(transcripts_file):
            raise FileNotFoundError(f"Missing embeddings file: {transcripts_file}. Run transcribe2.py first.")
            
        test_video = os.path.join(folder, video_files[0])
        print(f"Starting Vision Pipeline on video: {test_video}")
        
        with open(transcripts_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Handle both formats: plain list OR {"metadata":..., "segments":[...]}
        chunks = raw.get("segments", raw) if isinstance(raw, dict) else raw
            
        vision_engine = VisionProcessor()
        vision_engine.process_video_blocks(test_video, chunks)
        
    except Exception as e:
        print(f"\n[Vision Pipeline Error]: {str(e)}")