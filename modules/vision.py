"""
grabs frames from video, throws out bad ones (black/blurry/duplicate), 
and gets the best frame for each chunk using siglip.
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    TEMP_ASSETS_DIR, FRAME_SAMPLE_COUNT,
    BLACK_FRAME_THRESHOLD, BLUR_THRESHOLD, DUPLICATE_FRAME_THRESHOLD,
)
from schemas import TranscriptChunk, VisualChunk
from modules.embeddings import embedding_manager


class SemanticVisionProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- frame helpers ----

    def _calculate_phash(self, frame: np.ndarray) -> str:
        """hash frame to find duplicates"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_8x8 = dct[0:8, 0:8]
        median = np.median(dct_8x8)
        bits = (dct_8x8 > median).flatten()
        return "".join("1" if b else "0" for b in bits)

    def _phash_similarity(self, hash1: str, hash2: str) -> float:
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 0.0
        arr1 = np.frombuffer(hash1.encode("ascii"), dtype=np.uint8)
        arr2 = np.frombuffer(hash2.encode("ascii"), dtype=np.uint8)
        hamming = np.sum(arr1 != arr2)
        return 1.0 - (hamming / len(hash1))

    def _passes_quality_filter(self, frame: np.ndarray, previous_hash: str | None) -> tuple[bool, str | None, str | None]:
        """check if frame is good. returns true/false and reason."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(np.mean(gray))
        if brightness < BLACK_FRAME_THRESHOLD:
            return False, None, "black_frame"

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < BLUR_THRESHOLD:
            return False, None, "blurry"

        current_hash = self._calculate_phash(frame)
        if previous_hash is not None:
            similarity = self._phash_similarity(current_hash, previous_hash)
            if similarity > DUPLICATE_FRAME_THRESHOLD:
                return False, current_hash, "duplicate"

        return True, current_hash, None

    def get_raw_frame_at_time(self, cap, timestamp_sec):
        """get the raw frame from opencv"""
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ret, frame = cap.read()
        return frame if ret else None

    def process_video_blocks(
        self, video_path: str, video_id: str, transcript_chunks: list[TranscriptChunk],
    ) -> list[VisualChunk]:
        """
        loop over transcript chunks and get the best frame for each
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        results: list[VisualChunk] = []
        previous_keyframe_hash: str | None = None

        print(f"--- [Vision Engine] Aligning {len(transcript_chunks)} chunks for video_id={video_id} ---")

        for chunk in transcript_chunks:
            timestamps = np.linspace(chunk.start, chunk.end, FRAME_SAMPLE_COUNT)

            candidates = []  # list of (raw_frame, timestamp, phash)
            for ts in timestamps:
                frame = self.get_raw_frame_at_time(cap, ts)
                if frame is None:
                    continue
                passes, phash, rejection_reason = self._passes_quality_filter(frame, previous_keyframe_hash)
                if passes:
                    candidates.append((frame, ts, phash))
                    # update hash so we compare with the last frame, allows multiple frames per chunk
                    previous_keyframe_hash = phash
                else:
                    # 2. Print it so Modal catches it in the background logs stream
                    print(f"   [Quality Filter] Frame at {round(ts, 2)}s rejected: {rejection_reason}")

            # if all frames are bad, just use the first one so we don't have nothing
            if not candidates:
                for ts in timestamps:
                    frame = self.get_raw_frame_at_time(cap, ts)
                    if frame is not None:
                        candidates.append((frame, ts, None))
                        break

            if not candidates:
                print(f"Chunk {chunk.index}: No frames extracted at all, skipping.")
                continue

            frames_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f, _, _ in candidates]
            image_features = torch.tensor(
                [embedding_manager.get_image_embedding(f) for f in frames_pil]
            )

            # keep all good frames
            for i in range(len(candidates)):
                frame, ts, phash = candidates[i]
                results.append(VisualChunk(
                    video_id=video_id,
                    chunk_index=chunk.index,
                    timestamp=round(float(ts), 2),
                    embedding=image_features[i].cpu().numpy().tolist(),
                    similarity_score=1.0,  # we don't use this score anymore
                ))

            print(f"Chunk {chunk.index}: kept {len(candidates)} distinct frame(s) out of {FRAME_SAMPLE_COUNT}")

        cap.release()
        embedding_manager.flush_caches()
        print("--- [Vision Engine] Alignment complete. ---")
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