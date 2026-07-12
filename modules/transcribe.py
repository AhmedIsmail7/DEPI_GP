import os
import json
import torch
from pydub import AudioSegment
import whisper
from modules.embeddings import embedding_manager

from config import (
    WHISPER_MODEL_SIZE,
    CHUNK_DURATION_MS, OVERLAP_MS, TEMP_ASSETS_DIR, CHUNKS_SUBDIR,
)
from schemas import TranscriptChunk


def compute_chunk_boundaries(total_ms: int, chunk_ms: int, overlap_ms: int) -> list[tuple[int, int]]:
    """
    Pure sliding-window math, extracted from process_audio_with_overlap
    so it's testable without loading audio or running Whisper.
    Returns a list of (start_ms, end_ms) tuples.
    """
    step_ms = chunk_ms - overlap_ms
    return [(start, start + chunk_ms) for start in range(0, total_ms, step_ms)]


class Transcriber:
    def __init__(self, model_size=WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    @property
    def model(self):
        """Lazy-loaded: keeps this class importable on a laptop without
        pulling in Whisper weights until actually needed."""
        if self._model is None:
            print(f"Loading Whisper model '{self.model_size}' on {self.device}...")
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def process_audio_with_overlap(
        self, video_path: str, video_id: str,
        chunk_ms: int = CHUNK_DURATION_MS, overlap_ms: int = OVERLAP_MS,
    ) -> list[TranscriptChunk]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        audio = AudioSegment.from_file(video_path)
        boundaries = compute_chunk_boundaries(len(audio), chunk_ms, overlap_ms)
        chunks: list[TranscriptChunk] = []

        chunks_dir = os.path.join(TEMP_ASSETS_DIR, CHUNKS_SUBDIR)
        os.makedirs(chunks_dir, exist_ok=True)

        print(f"Starting sliding window transcription: {len(audio)}ms total.")

        for i, (start_ms, end_ms) in enumerate(boundaries):
            chunk_audio = audio[start_ms:end_ms]

            chunk_path = os.path.join(chunks_dir, f"chunk_{i}.mp3")
            chunk_audio.export(chunk_path, format="mp3")

            result = self.model.transcribe(chunk_path, fp16=(self.device == "cuda"))
            text = result["text"].strip()
            vector = embedding_manager.get_text_embedding(text)

            chunks.append(TranscriptChunk(
                video_id=video_id,
                index=i,
                start=start_ms / 1000,
                end=end_ms / 1000,
                text=text,
                embedding=vector,
            ))

            os.remove(chunk_path)
            print(f"Processed chunk {i}: {start_ms/1000}s - {end_ms/1000}s")

        os.rmdir(chunks_dir)
        return chunks


# Singleton — model loading is now lazy, so this is cheap on import
transcriber_engine = Transcriber()

if __name__ == "__main__":
    try:
        folder = TEMP_ASSETS_DIR
        files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
        if not files:
            raise FileNotFoundError("No video files found in temp_assets/")

        test_file = os.path.join(folder, files[0])
        test_video_id = os.path.splitext(files[0])[0]
        print(f"Testing with file: {test_file} | video_id: {test_video_id}")

        results = transcriber_engine.process_audio_with_overlap(test_file, test_video_id)

        out_path = os.path.join(TEMP_ASSETS_DIR, "transcript_chunks.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in results], f, indent=4)

        print(f"Pipeline complete. Data saved to {out_path}")
        print(f"Total chunks processed: {len(results)}")

    except Exception as e:
        print(f"Pipeline error: {e}")