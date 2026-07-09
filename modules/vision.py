# modules/vision.py
"""
Vedex - Preprocessing Vision Module
====================================
This module implements the vision pipeline. It samples frames from video chunks,
filters them based on quality, computes OCR on the top candidates, generates
embeddings, and ranks them to select the best representative keyframe.
"""

import os
import json
import time
import logging
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import easyocr
import torch
from PIL import Image

from config import VISION
from modules.embeddings import embedding_manager

# Initialize logger
logger = logging.getLogger("vedex.vision")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Safe configurations from config.py or default fallbacks
BLACK_FRAME_THRESHOLD = getattr(VISION, "black_frame_threshold", 15)
BLUR_THRESHOLD = getattr(VISION, "blur_threshold", 100.0)
DUPLICATE_FRAME_THRESHOLD = getattr(VISION, "duplicate_frame_threshold", 0.95)

SIMILARITY_WEIGHT = getattr(VISION, "similarity_weight", 0.4)
OCR_WEIGHT = getattr(VISION, "ocr_weight", 0.4)
QUALITY_WEIGHT = getattr(VISION, "quality_weight", 0.2)

# Load EasyOCR Reader
try:
    # Use GPU for EasyOCR if CUDA is available
    use_gpu = torch.cuda.is_available()
    ocr_reader = easyocr.Reader(["en", "ar"], gpu=use_gpu)
    logger.info("EasyOCR initialized successfully for English and Arabic.")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    ocr_reader = None

class VisionProcessor:
    def __init__(self):
        self.cache_dir = Path("temp_assets/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ocr_cache_path = self.cache_dir / "ocr_cache.json"
        self.ocr_cache = self._load_ocr_cache()
        self.ocr_lock = threading.Lock() if "threading" in globals() else None
        self.ocr_cache_dirty = False
        
        # Statistics tracking
        self.stats = {
            "total_sampled_frames": 0,
            "filtered_frames": 0,
            "ocr_cache_hits": 0,
            "selected_keyframes": 0,
            "cosine_similarities": [],
            "quality_scores": [],
        }

    def _load_ocr_cache(self) -> dict:
        if self.ocr_cache_path.exists():
            try:
                with open(self.ocr_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load OCR cache: {e}. Starting fresh.")
        return {}

    def _save_ocr_cache(self):
        try:
            with open(self.ocr_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.ocr_cache, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write OCR cache: {e}")

    def flush_ocr_cache(self):
        """
        Flush dirty OCR cache to disk.
        """
        if self.ocr_cache_dirty:
            self._save_ocr_cache()
            self.ocr_cache_dirty = False
            logger.info("OCR cache persisted to disk.")

    def calculate_phash(self, image: np.ndarray) -> str:
        """
        Compute 64-bit Perceptual Hash (pHash) using Discrete Cosine Transform (DCT).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(resized))
            # Extract top-left 8x8 DCT coefficients (excluding high frequencies)
            dct_8x8 = dct[0:8, 0:8]
            median = np.median(dct_8x8)
            hash_bits = (dct_8x8 > median).flatten()
            return "".join(["1" if b else "0" for b in hash_bits])
        except Exception as e:
            logger.error(f"Failed to compute pHash: {e}")
            return "0" * 64

    def phash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity score [0, 1] based on Hamming distance.
        Vectorized with NumPy for performance.
        """
        if len(hash1) != len(hash2) or not hash1:
            return 0.0
        arr1 = np.frombuffer(hash1.encode('ascii'), dtype=np.uint8)
        arr2 = np.frombuffer(hash2.encode('ascii'), dtype=np.uint8)
        hamming_dist = np.sum(arr1 != arr2)
        return 1.0 - (hamming_dist / len(hash1))

    def calculate_entropy(self, gray_image: np.ndarray) -> float:
        """
        Calculate 1D Shannon Entropy of grayscale frame.
        """
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    def evaluate_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Evaluate frame quality: Brightness, Sharpness (Laplacian variance), and Entropy.
        Normalizes scores to range [0, 1].
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Brightness score (ideal peaks at middle gray 127.5)
        mean_brightness = float(np.mean(gray))
        brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
        brightness_score = max(0.0, min(1.0, brightness_score))
        
        # 2. Sharpness score (variance of Laplacian, normalized with soft saturation)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = sharpness / (sharpness + 100.0)
        
        # 3. Entropy score (normalized by max possible 8-bit entropy = 8.0)
        entropy = self.calculate_entropy(gray)
        entropy_score = max(0.0, min(1.0, entropy / 8.0))
        
        # Combined FrameQuality
        quality = 0.4 * brightness_score + 0.4 * sharpness_score + 0.2 * entropy_score
        
        return {
            "quality": quality,
            "brightness": mean_brightness,
            "sharpness": sharpness,
            "entropy": entropy
        }

    def process_ocr_for_frame(self, frame: np.ndarray) -> Tuple[str, dict]:
        """
        Performs EasyOCR on a frame with caching support.
        """
        h = hashlib.sha256(frame.tobytes()).hexdigest()
        
        # Cache lookup
        if h in self.ocr_cache:
            self.stats["ocr_cache_hits"] += 1
            return h, self.ocr_cache[h]

        if not ocr_reader:
            return h, {"text": "", "confidence": 0.0}

        try:
            results = ocr_reader.readtext(frame)
            if not results:
                data = {"text": "", "confidence": 0.0}
            else:
                texts = [res[1] for res in results if res[1]]
                confidences = [res[2] for res in results]
                ocr_text = " ".join(texts).strip()
                avg_conf = float(np.mean(confidences))
                data = {"text": ocr_text, "confidence": avg_conf}
        except Exception as e:
            logger.warning(f"OCR failed for frame hash {h}: {e}")
            data = {"text": "", "confidence": 0.0}

        # Update persistent cache
        if self.ocr_lock:
            with self.ocr_lock:
                self.ocr_cache[h] = data
        else:
            self.ocr_cache[h] = data
        self.ocr_cache_dirty = True
        return h, data

    def calculate_ocr_score(self, ocr_text: str, confidence: float) -> float:
        """
        OCR Score = 0.5 * confidence + 0.3 * normalized_word_count + 0.2 * alphabetic_ratio
        """
        if not ocr_text:
            return 0.0
        
        words = ocr_text.split()
        word_count = len(words)
        # Assume 50 words is standard maximum for one frame representation
        normalized_word_count = min(word_count / 50.0, 1.0)
        
        text_len = len(ocr_text)
        # Check alphabetic ratio (English + Arabic letters + spaces)
        alpha_count = sum(c.isalnum() or c.isspace() or '\u0600' <= c <= '\u06FF' for c in ocr_text)
        alphabetic_ratio = alpha_count / text_len if text_len > 0 else 0.0
        
        return 0.5 * confidence + 0.3 * normalized_word_count + 0.2 * alphabetic_ratio

    def process_video_blocks(self, video_path: str, transcript_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs the vision preprocessing pipeline on all chunks.
        """
        video_p = Path(video_path)
        if not video_p.exists():
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(str(video_p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        cap.release()

        if duration_sec <= 0:
            raise ValueError(f"Invalid video length or format: {video_path}")

        # Ensure keyframes directory exists
        keyframes_dir = Path("temp_assets/keyframes")
        keyframes_dir.mkdir(parents=True, exist_ok=True)

        # Configurable sampling rate (default 10 frames per chunk)
        sample_count = getattr(VISION, "keyframe_sampling_interval", 10)

        visual_embeddings = []
        previous_keyframe_hash = None

        logger.info(f"Starting frame extraction and ranking for {len(transcript_chunks)} chunks...")
        start_time = time.time()

        for idx, chunk in enumerate(transcript_chunks):
            chunk_idx = chunk.get("index", idx)
            start_t = chunk.get("start", chunk.get("start_time", 0.0))
            end_t = chunk.get("end", chunk.get("end_time", duration_sec))
            transcript = chunk.get("transcript", chunk.get("text", ""))

            # Uniform sampling
            timestamps = np.linspace(start_t, end_t, num=sample_count)
            
            candidates = []
            chunk_sampled = 0
            chunk_filtered = 0

            cap = cv2.VideoCapture(str(video_p))
            for frame_idx_in_chunk, ts in enumerate(timestamps):
                frame_number = int(ts * fps)
                if frame_number >= total_frames:
                    frame_number = total_frames - 1

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                chunk_sampled += 1
                self.stats["total_sampled_frames"] += 1

                # Frame filtering: 1. Brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                if mean_brightness < BLACK_FRAME_THRESHOLD:
                    chunk_filtered += 1
                    self.stats["filtered_frames"] += 1
                    continue

                # 2. Blur Check (variance of Laplacian)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sharpness < BLUR_THRESHOLD:
                    chunk_filtered += 1
                    self.stats["filtered_frames"] += 1
                    continue

                # 3. Duplicate check via pHash Hamming similarity
                current_hash = self.calculate_phash(frame)
                if previous_keyframe_hash is not None:
                    sim = self.phash_similarity(current_hash, previous_keyframe_hash)
                    if sim > DUPLICATE_FRAME_THRESHOLD:
                        chunk_filtered += 1
                        self.stats["filtered_frames"] += 1
                        continue

                # Evaluate quality score
                quality_data = self.evaluate_quality(frame)

                candidates.append({
                    "frame": frame,
                    "frame_index": frame_number,
                    "frame_time": round(float(ts), 2),
                    "pHash": current_hash,
                    "brightness": quality_data["brightness"],
                    "sharpness": quality_data["sharpness"],
                    "entropy": quality_data["entropy"],
                    "quality_score": quality_data["quality"]
                })
            cap.release()

            # Fallback if all frames were filtered: Keep the highest quality frame overall
            if not candidates and chunk_sampled > 0:
                logger.warning(f"Chunk {chunk_idx}: All frames filtered. Falling back to highest quality.")
                cap = cv2.VideoCapture(str(video_p))
                fallback_candidates = []
                for ts in timestamps:
                    frame_number = int(ts * fps)
                    if frame_number >= total_frames:
                        frame_number = total_frames - 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        quality_data = self.evaluate_quality(frame)
                        fallback_candidates.append({
                            "frame": frame,
                            "frame_index": frame_number,
                            "frame_time": round(float(ts), 2),
                            "pHash": self.calculate_phash(frame),
                            "brightness": quality_data["brightness"],
                            "sharpness": quality_data["sharpness"],
                            "entropy": quality_data["entropy"],
                            "quality_score": quality_data["quality"]
                        })
                cap.release()
                if fallback_candidates:
                    fallback_candidates.sort(key=lambda x: x["quality_score"], reverse=True)
                    candidates = [fallback_candidates[0]]

            if not candidates:
                logger.error(f"Chunk {chunk_idx}: Absolutely no frames could be extracted.")
                continue

            # Keep Top 5 by image quality
            candidates.sort(key=lambda x: x["quality_score"], reverse=True)
            top_candidates = candidates[:5]

            # Parallel Threaded OCR on Top 5
            frames_to_ocr = [cand["frame"] for cand in top_candidates]
            with ThreadPoolExecutor(max_workers=min(5, len(frames_to_ocr))) as executor:
                ocr_results = list(executor.map(lambda f: self.process_ocr_for_frame(f), frames_to_ocr))

            # Enrich candidates with OCR info and build semantic text
            for idx_cand, (hash_val, ocr_data) in enumerate(ocr_results):
                ocr_text = ocr_data.get("text", "")
                conf = ocr_data.get("confidence", 0.0)
                
                # Semantic text logic:
                if ocr_text:
                    semantic_text = f"Transcript:\n{transcript}\n\nVisible Text:\n{ocr_text}"
                else:
                    semantic_text = transcript

                top_candidates[idx_cand]["ocr_text"] = ocr_text
                top_candidates[idx_cand]["ocr_confidence"] = conf
                top_candidates[idx_cand]["semantic_text"] = semantic_text
                top_candidates[idx_cand]["ocr_score"] = self.calculate_ocr_score(ocr_text, conf)

            # Batched GPU SigLIP2 Embedding generation for Top 5
            texts_to_embed = [cand["semantic_text"] for cand in top_candidates]
            images_to_embed = [cand["frame"] for cand in top_candidates]

            try:
                text_embs = embedding_manager.batch_text_embeddings(texts_to_embed)
                image_embs = embedding_manager.batch_image_embeddings(images_to_embed)
            except Exception as e:
                logger.error(f"SigLIP2 batched embedding failed for Chunk {chunk_idx}: {e}. Skipping.")
                continue

            # Rank the Top 5 candidates
            ranked_candidates = []
            for i_cand, cand in enumerate(top_candidates):
                text_emb = text_embs[i_cand]
                img_emb = image_embs[i_cand]
                
                sim = embedding_manager.cosine_similarity(text_emb, img_emb)
                
                # Multi-criteria scoring
                final_score = (
                    SIMILARITY_WEIGHT * sim +
                    OCR_WEIGHT * cand["ocr_score"] +
                    QUALITY_WEIGHT * cand["quality_score"]
                )

                cand["similarity_score"] = sim
                cand["ranking_score"] = final_score
                cand["text_embedding"] = text_emb
                cand["image_embedding"] = img_emb
                ranked_candidates.append(cand)

            # Choose the absolute best frame
            ranked_candidates.sort(key=lambda x: x["ranking_score"], reverse=True)
            winner = ranked_candidates[0]

            # Save Keyframe: zero-padded numbering (5 digits)
            frame_path = keyframes_dir / f"chunk_{chunk_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), winner["frame"])

            # Keep previous hash updated
            previous_keyframe_hash = winner["pHash"]

            # Save stats
            self.stats["selected_keyframes"] += 1
            self.stats["cosine_similarities"].append(winner["similarity_score"])
            self.stats["quality_scores"].append(winner["quality_score"])

            # Store result metadata
            visual_embeddings.append({
                "chunk_index": chunk_idx,
                "start": start_t,
                "end": end_t,
                "timestamp": winner["frame_time"],
                "frame_path": str(frame_path),
                "ocr_text": winner["ocr_text"],
                "semantic_text": winner["semantic_text"],
                "image_embedding": winner["image_embedding"].tolist(),
                "text_embedding": winner["text_embedding"].tolist(),
                "similarity_score": round(winner["similarity_score"], 4),
                "ocr_score": round(winner["ocr_score"], 4),
                "quality_score": round(winner["quality_score"], 4),
                "ranking_score": round(winner["ranking_score"], 4),
                "brightness": round(winner["brightness"], 2),
                "sharpness": round(winner["sharpness"], 2),
                "entropy": round(winner["entropy"], 4),
                "frame_index": winner["frame_index"],
                "frame_time": winner["frame_time"]
            })

            logger.info(
                f"Chunk {chunk_idx:04d} | Winner Frame Time: {winner['frame_time']}s | "
                f"Similarity: {winner['similarity_score']:.4f} | "
                f"OCR Score: {winner['ocr_score']:.4f} | "
                f"Quality: {winner['quality_score']:.4f} | "
                f"Final Score: {winner['ranking_score']:.4f} | "
                f"Filtered: {chunk_filtered}/{chunk_sampled}"
            )

        total_time = time.time() - start_time
        
        # Summary printing
        avg_sim = np.mean(self.stats["cosine_similarities"]) if self.stats["cosine_similarities"] else 0.0
        avg_qual = np.mean(self.stats["quality_scores"]) if self.stats["quality_scores"] else 0.0

        print("\n" + "=" * 60)
        print("VEDEX PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total Sampled Frames : {self.stats['total_sampled_frames']}")
        print(f"Filtered Frames       : {self.stats['filtered_frames']}")
        print(f"OCR Cache Hits       : {self.stats['ocr_cache_hits']}")
        print(f"Selected Keyframes   : {self.stats['selected_keyframes']}")
        print(f"Average Similarity   : {avg_sim:.4f}")
        print(f"Average Quality Score: {avg_qual:.4f}")
        print(f"Total Processing Time: {total_time:.2f} seconds")
        print("=" * 60 + "\n")

        # Flush dirty caches to disk
        self.flush_ocr_cache()
        embedding_manager.flush_caches()

        # Write results to visual_embeddings.json
        output_file = Path("temp_assets/visual_embeddings.json")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(visual_embeddings, f, ensure_ascii=False, indent=4)
            logger.info(f"Visual embeddings saved successfully to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to write visual embeddings JSON: {e}")

        return visual_embeddings

# Global instance
vision_engine = VisionProcessor()

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--transcripts", type=str, default="temp_assets/transcript_chunks.json", help="Path to transcripts JSON file")
    args = parser.parse_args()

    # Configure root logger for standalone run
    logging.basicConfig(level=logging.INFO)

    try:
        with open(args.transcripts, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} transcript chunks from {args.transcripts}")
        vision_engine.process_video_blocks(args.video, chunks)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)