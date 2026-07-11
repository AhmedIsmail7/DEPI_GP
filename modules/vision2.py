# modules/vision2.py
"""
Vedex - Vision Pipeline
========================
For each 30-second audio chunk produced by transcribe2.py:
  1. Sample 5 frames evenly from the chunk's time window.
  2. Filter out black / blurry / duplicate frames.
  3. Use SigLIP to embed the chunk text AND all candidate frames,
     then pick the frame with the highest cosine similarity to the text.
  4. Run EasyOCR on the winning frame.
  5. Save keyframe image + full metadata to visual_embeddings.json.

No external project dependencies - fully self-contained.
"""

import os
import json
import time
import logging
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import easyocr
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ─────────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger("vedex.vision2")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(_h)

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
FRAMES_PER_CHUNK        = 5     # number of frames sampled per chunk
BLACK_FRAME_THRESHOLD   = 15    # min mean brightness to keep frame
BLUR_THRESHOLD          = 100.0 # min Laplacian variance to keep frame
DUPLICATE_THRESHOLD     = 0.95  # pHash similarity above this → duplicate
SIGLIP_MODEL            = "google/siglip-base-patch16-224"

# ─────────────────────────────────────────────────────────────
#  EasyOCR  (GPU if available)
# ─────────────────────────────────────────────────────────────
_ocr_reader: easyocr.Reader | None = None
_ocr_lock = threading.Lock()

def _get_ocr_reader() -> easyocr.Reader | None:
    global _ocr_reader
    if _ocr_reader is None:
        try:
            use_gpu = torch.cuda.is_available()
            _ocr_reader = easyocr.Reader(["en", "ar"], gpu=use_gpu)
            logger.info("EasyOCR loaded (en + ar).")
        except Exception as e:
            logger.error(f"EasyOCR failed to load: {e}")
    return _ocr_reader

# ─────────────────────────────────────────────────────────────
#  SigLIP Embedding Manager
# ─────────────────────────────────────────────────────────────
class SigLIPManager:
    """Wraps SigLIP for batch text & image embedding."""

    def __init__(self, model_name: str = SIGLIP_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SigLIP '{model_name}' on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_text_len = getattr(self.model.config, "max_position_embeddings", 64)
        logger.info("SigLIP loaded.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        inputs = self.processor(
            text=texts,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            f = self.model.get_text_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().tolist()

    def embed_images(self, cv2_frames: List[np.ndarray]) -> List[List[float]]:
        pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in cv2_frames]
        inputs = self.processor(images=pil_imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            f = self.model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().tolist()

    @staticmethod
    def cosine_sim(a: List[float], b: List[float]) -> float:
        return float(np.dot(a, b))


# ─────────────────────────────────────────────────────────────
#  Vision Processor
# ─────────────────────────────────────────────────────────────
class VisionProcessor:

    def __init__(self, siglip_model: str = SIGLIP_MODEL):
        self.siglip = SigLIPManager(siglip_model)

        # OCR cache (hash → result) persisted to disk
        self.cache_dir = Path("temp_assets/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_cache_path = self.cache_dir / "ocr_cache.json"
        self.ocr_cache: dict = self._load_ocr_cache()
        self.ocr_cache_dirty = False

        self.stats = {
            "total_sampled": 0,
            "filtered": 0,
            "ocr_cache_hits": 0,
            "selected": 0,
        }

    # ── OCR cache ──────────────────────────────────────────
    def _load_ocr_cache(self) -> dict:
        if self.ocr_cache_path.exists():
            try:
                return json.loads(self.ocr_cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _flush_ocr_cache(self):
        if self.ocr_cache_dirty:
            self.ocr_cache_path.write_text(
                json.dumps(self.ocr_cache, ensure_ascii=False), encoding="utf-8"
            )
            self.ocr_cache_dirty = False

    # ── Frame quality helpers ───────────────────────────────
    def _phash(self, frame: np.ndarray) -> str:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct     = cv2.dct(np.float32(resized))[0:8, 0:8]
        median  = np.median(dct)
        return "".join("1" if b else "0" for b in (dct > median).flatten())

    def _phash_sim(self, h1: str, h2: str) -> float:
        if len(h1) != len(h2) or not h1:
            return 0.0
        a1 = np.frombuffer(h1.encode(), dtype=np.uint8)
        a2 = np.frombuffer(h2.encode(), dtype=np.uint8)
        return 1.0 - np.sum(a1 != a2) / len(h1)

    def _quality(self, frame: np.ndarray) -> Dict[str, float]:
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        b_score    = max(0.0, min(1.0, 1.0 - abs(brightness - 127.5) / 127.5))
        s_score    = sharpness / (sharpness + 100.0)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy       = float(-np.sum(hist * np.log2(hist)))
        entropy_score = max(0.0, min(1.0, entropy / 8.0))

        return {
            "quality":    0.4 * b_score + 0.4 * s_score + 0.2 * entropy_score,
            "brightness": brightness,
            "sharpness":  sharpness,
            "entropy":    entropy,
        }

    # ── OCR ─────────────────────────────────────────────────
    def _run_ocr(self, frame: np.ndarray) -> Tuple[str, float]:
        """Returns (text, confidence) for a frame, with cache."""
        key = hashlib.sha256(frame.tobytes()).hexdigest()

        with _ocr_lock:
            if key in self.ocr_cache:
                self.stats["ocr_cache_hits"] += 1
                cached = self.ocr_cache[key]
                return cached["text"], cached["confidence"]

        reader = _get_ocr_reader()
        text, conf = "", 0.0
        if reader:
            try:
                results = reader.readtext(frame)
                if results:
                    text = " ".join(r[1] for r in results if r[1]).strip()
                    conf = float(np.mean([r[2] for r in results]))
            except Exception as e:
                logger.warning(f"OCR error: {e}")

        with _ocr_lock:
            self.ocr_cache[key] = {"text": text, "confidence": conf}
            self.ocr_cache_dirty = True

        return text, conf

    # ── Main pipeline ────────────────────────────────────────
    def process_video_blocks(
        self,
        video_path: str,
        transcript_chunks: List[Dict[str, Any]],
        output_dir: str = "temp_assets",
    ) -> str:
        """
        Parameters
        ----------
        video_path        : path to the downloaded video file
        transcript_chunks : list of chunks from siglip_text_embeddings.json
                            (segments list, each has start_time / end_time / text)
        output_dir        : where to save visual_embeddings.json and keyframes/

        Returns
        -------
        Path to the saved visual_embeddings.json
        """
        video_p = Path(video_path)
        if not video_p.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        keyframes_dir = Path(output_dir) / "keyframes"
        keyframes_dir.mkdir(parents=True, exist_ok=True)

        # Open the video file ONCE
        cap = cv2.VideoCapture(str(video_p))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps if fps > 0 else 0.0
        logger.info(f"Video: {video_p.name}  |  {duration:.1f}s  |  {fps:.1f} fps")

        visual_embeddings: List[Dict] = []
        prev_hash:         str | None = None
        t_start = time.time()

        logger.info(f"Processing {len(transcript_chunks)} chunks ...")

        for idx, chunk in enumerate(transcript_chunks):
            chunk_idx = chunk.get("chunk_index", idx)
            start_t   = float(chunk.get("start_time", 0.0))
            end_t     = float(chunk.get("end_time",   duration))
            text      = chunk.get("text", "").strip()

            # ── 1. Sample FRAMES_PER_CHUNK frames ──────────────
            timestamps = np.linspace(start_t, end_t, num=FRAMES_PER_CHUNK)
            candidates = []

            for ts in timestamps:
                fn = min(int(ts * fps), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                self.stats["total_sampled"] += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Black frame check
                if float(np.mean(gray)) < BLACK_FRAME_THRESHOLD:
                    self.stats["filtered"] += 1
                    continue

                # Blur check
                if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < BLUR_THRESHOLD:
                    self.stats["filtered"] += 1
                    continue

                # Duplicate check against previous keyframe
                ph = self._phash(frame)
                if prev_hash and self._phash_sim(ph, prev_hash) > DUPLICATE_THRESHOLD:
                    self.stats["filtered"] += 1
                    continue

                q = self._quality(frame)
                candidates.append({
                    "frame":         frame,
                    "frame_time":    round(float(ts), 2),
                    "frame_index":   fn,
                    "phash":         ph,
                    **q,
                })

            # Fallback: if all filtered, use the sharpest frame ignoring thresholds
            if not candidates:
                logger.warning(f"Chunk {chunk_idx}: all frames filtered → fallback to best quality.")
                for ts in timestamps:
                    fn = min(int(ts * fps), total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        q = self._quality(frame)
                        candidates.append({"frame": frame, "frame_time": round(float(ts), 2),
                                           "frame_index": fn, "phash": self._phash(frame), **q})
                if not candidates:
                    logger.error(f"Chunk {chunk_idx}: no frames could be read. Skipping.")
                    continue

            # ── 2. Keep top 5 by visual quality ────────────────
            candidates.sort(key=lambda c: c["quality"], reverse=True)
            top5 = candidates[:5]

            # ── 3. SigLIP: embed text + all top-5 images ───────
            chunk_text_label = f"Transcript: {text}" if text else "video frame"
            try:
                text_embs  = self.siglip.embed_texts([chunk_text_label])
                image_embs = self.siglip.embed_images([c["frame"] for c in top5])
            except Exception as e:
                logger.error(f"Chunk {chunk_idx}: SigLIP embedding failed ({e}). Skipping.")
                continue

            text_emb = text_embs[0]

            # ── 4. Rank by cosine similarity to chunk text ──────
            for i, cand in enumerate(top5):
                cand["image_embedding"] = image_embs[i]
                cand["sim_score"]       = self.siglip.cosine_sim(text_emb, image_embs[i])

            top5.sort(key=lambda c: c["sim_score"], reverse=True)
            winner = top5[0]

            # ── 5. OCR on the winning frame ─────────────────────
            ocr_text, ocr_conf = self._run_ocr(winner["frame"])

            # ── 6. Save keyframe image ──────────────────────────
            frame_path = keyframes_dir / f"chunk_{chunk_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), winner["frame"])
            prev_hash = winner["phash"]
            self.stats["selected"] += 1

            # ── 7. Collect result ───────────────────────────────
            visual_embeddings.append({
                "chunk_index":     chunk_idx,
                "start_time":      start_t,
                "end_time":        end_t,
                "timestamp":       winner["frame_time"],
                "frame_path":      str(frame_path),
                "ocr_text":        ocr_text,
                "ocr_confidence":  round(ocr_conf, 4),
                "image_embedding": winner["image_embedding"],
                "text_embedding":  text_emb,
                "similarity_score": round(winner["sim_score"], 4),
                "quality_score":   round(winner["quality"], 4),
                "brightness":      round(winner["brightness"], 2),
                "sharpness":       round(winner["sharpness"],  2),
                "entropy":         round(winner["entropy"],     4),
                "frame_index":     winner["frame_index"],
            })

            logger.info(
                # f"[Chunk {chunk_idx:04d}] "
                f"{start_t:.1f}s–{end_t:.1f}s | "
                # f"winner @ {winner['frame_time']}s | "
                f"sim={winner['sim_score']:.3f} | "
                # f"ocr={len(ocr_text)} chars"
            )

        cap.release()
        self._flush_ocr_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t_start

        # ── Summary ─────────────────────────────────────────
        print("\n" + "=" * 55)
        print("  VEDEX VISION PIPELINE — SUMMARY")
        print("=" * 55)
        print(f"  Chunks processed  : {len(visual_embeddings)}")
        print(f"  Frames sampled    : {self.stats['total_sampled']}")
        print(f"  Frames filtered   : {self.stats['filtered']}")
        print(f"  OCR cache hits    : {self.stats['ocr_cache_hits']}")
        print(f"  Keyframes saved   : {self.stats['selected']}")
        if visual_embeddings:
            avg_sim = np.mean([v["similarity_score"] for v in visual_embeddings])
            print(f"  Avg similarity    : {avg_sim:.4f}")
        print(f"  Total time        : {elapsed:.1f}s")
        print("=" * 55 + "\n")

        # ── Write output JSON ────────────────────────────────
        out_path = Path(output_dir) / "visual_embeddings.json"
        out_path.write_text(
            json.dumps(visual_embeddings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved {len(visual_embeddings)} entries → {out_path}")
        return str(out_path)


# ─────────────────────────────────────────────────────────────
#  Lazy singleton
# ─────────────────────────────────────────────────────────────
_vision_instance: VisionProcessor | None = None

def get_vision_processor() -> VisionProcessor:
    global _vision_instance
    if _vision_instance is None:
        _vision_instance = VisionProcessor()
    return _vision_instance


# ─────────────────────────────────────────────────────────────
#  Standalone CLI
#  Usage: python modules/vision2.py --video temp_assets/yt_video.mp4
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys
    import traceback

    parser = argparse.ArgumentParser(description="Vedex Vision Pipeline")
    parser.add_argument("--video",       required=True,  help="Path to the video file")
    parser.add_argument("--transcripts", default="temp_assets/siglip_text_embeddings.json",
                        help="Path to siglip_text_embeddings.json from transcribe2.py")
    parser.add_argument("--output",      default="temp_assets",
                        help="Output directory for visual_embeddings.json and keyframes/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        with open(args.transcripts, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Support both plain list and {"metadata":..., "segments":[...]}
        chunks = raw.get("segments", raw) if isinstance(raw, dict) else raw

        logger.info(f"Loaded {len(chunks)} chunks from {args.transcripts}")
        get_vision_processor().process_video_blocks(args.video, chunks, args.output)

    except Exception as e:
        logger.error(f"Fatal: {e}")
        traceback.print_exc()
        sys.exit(1)