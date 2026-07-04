import os

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    import whisper
except Exception:  # pragma: no cover - optional dependency
    whisper = None

try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover - optional dependency
    AudioSegment = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from config import (
    WHISPER_MODEL,
    TEXT_EMBEDDING_MODEL,
    CHUNK_DURATION_MS,
    OVERLAP_DURATION_MS,
    CHUNKS_DIR,
    TEMP_DIR,
    TRANSCRIPT_OUTPUT,
)

from modules.utils import save_json


class Transcriber:
    def __init__(self, model_size: str = WHISPER_MODEL):
        """
        Initializes Whisper and the text embedding model lazily.
        """

        self.model_size = model_size
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.model = None
        self.embedding_model = None

        print(f"Whisper model '{model_size}' and embeddings will load on first use on {self.device}.")

    def _ensure_models_loaded(self):
        if self.model is not None and self.embedding_model is not None:
            return

        if whisper is None or SentenceTransformer is None or torch is None:
            raise ImportError("Whisper and sentence-transformers are required for transcription")

        print(f"Loading Whisper model '{self.model_size}' on {self.device}...")
        self.model = whisper.load_model(self.model_size, device=self.device)

        print(f"Loading Embedding model '{TEXT_EMBEDDING_MODEL}'...")
        self.embedding_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)

        print("Models loaded successfully.")

    def process_audio_with_overlap(
        self,
        video_path: str,
        chunk_ms: int = CHUNK_DURATION_MS,
        overlap_ms: int = OVERLAP_DURATION_MS,
    ):
        """
        Sliding-window transcription + embedding generation.
        """

        self._ensure_models_loaded()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if AudioSegment is None:
            raise ImportError("pydub is required for transcription")

        audio = AudioSegment.from_file(video_path)

        step_ms = chunk_ms - overlap_ms

        chunks_metadata = []

        os.makedirs(CHUNKS_DIR, exist_ok=True)

        print(f"Starting transcription ({len(audio) / 1000:.2f} seconds)...")

        for i, start_ms in enumerate(range(0, len(audio), step_ms)):

            end_ms = min(start_ms + chunk_ms, len(audio))

            chunk = audio[start_ms:end_ms]

            chunk_path = os.path.join(
                CHUNKS_DIR,
                f"chunk_{i}.mp3"
            )

            chunk.export(chunk_path, format="mp3")

            result = self.model.transcribe(
                chunk_path,
                fp16=(self.device == "cuda")
            )

            text = result["text"].strip()
            embedding = self.embedding_model.encode(text).tolist()

            chunks_metadata.append(
                {
                    "index": i,
                    "start": start_ms / 1000,
                    "end": end_ms / 1000,
                    "text": text,
                    "embedding": embedding,
                }
            )

            os.remove(chunk_path)

            print(
                f"Chunk {i} | "
                f"{start_ms/1000:.2f}s -> {end_ms/1000:.2f}s"
            )

        if os.path.exists(CHUNKS_DIR) and not os.listdir(CHUNKS_DIR):
            os.rmdir(CHUNKS_DIR)

        return chunks_metadata


transcriber_engine = Transcriber()


if __name__ == "__main__":

    try:

        videos = [
            file
            for file in os.listdir(TEMP_DIR)
            if file.endswith(".mp4")
        ]

        if not videos:
            raise FileNotFoundError(
                f"No video found inside '{TEMP_DIR}'."
            )

        video_path = os.path.join(TEMP_DIR, videos[0])

        print(f"Processing: {video_path}")

        transcript = transcriber_engine.process_audio_with_overlap(
            video_path
        )

        save_json(
            transcript,
            TRANSCRIPT_OUTPUT
        )

        print(f"Transcript saved to: {TRANSCRIPT_OUTPUT}")
        print(f"Total chunks: {len(transcript)}")

    except Exception as e:
        print(f"Pipeline Error: {e}")