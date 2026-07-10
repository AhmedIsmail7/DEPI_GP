import os
import json
from pydub import AudioSegment
from faster_whisper import WhisperModel

DEFAULT_WHISPER_MODEL = os.getenv("TRANSCRIPTION_WHISPER_MODEL", "large-v3")

class DualEmbeddingTranscriber:
    def __init__(
        self, 
        whisper_model=DEFAULT_WHISPER_MODEL,
        rag_embedding_model=None, # kept for backward compatibility
        siglip_model=None # kept for backward compatibility
    ):
        # Local torch import to avoid memory fragmentation before Whisper loads
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        print(f"--- [Init] Pipeline initializing on device: {self.device} (compute_type: {self.compute_type}) ---")
        
        # Load Faster-Whisper Model
        print(f"Loading Faster-Whisper model '{whisper_model}'...")
        cpu_threads = 1 if self.device == "cpu" else 0
        self.transcriber = WhisperModel(
            whisper_model, 
            device=self.device, 
            compute_type=self.compute_type,
            cpu_threads=cpu_threads
        )
        print("--- [Init] Whisper Model Loaded Successfully ---")

    def process_video_pipeline(self, video_path: str, output_dir: str = "temp_assets", chunk_ms: int = 30000, overlap_ms: int = 5000):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        chunks_temp_dir = os.path.join(output_dir, "temp_audio_chunks")
        os.makedirs(chunks_temp_dir, exist_ok=True)

        audio = AudioSegment.from_file(video_path)
        step_ms = chunk_ms - overlap_ms
        
        raw_chunks = []
        current_video_language = None

        print(f"--- [Processing] Starting sliding window transcription ({len(audio)/1000:.2f}s total duration) ---")

        for i, start_ms in enumerate(range(0, len(audio), step_ms)):
            end_ms = min(start_ms + chunk_ms, len(audio))
            chunk = audio[start_ms:end_ms]
            
            chunk_path = os.path.join(chunks_temp_dir, f"chunk_{i}.mp3")
            chunk.export(chunk_path, format="mp3")

            segments, info = self.transcriber.transcribe(
                chunk_path, 
                beam_size=5,
                language=current_video_language
            )
            segments = list(segments)
            text = " ".join([seg.text for seg in segments]).strip()
            
            if current_video_language is None and info is not None:
                current_video_language = info.language
                print(f"--- [Language Detection] Locked Language: '{current_video_language}' (Confidence: {info.language_probability:.2f}) ---")
            
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

            if not text or len(text) < 3:
                print(f"[Skip] Chunk {i} ({start_ms/1000:.1f}s - {end_ms/1000:.1f}s): No speech detected.")
                continue

            start_sec = round(start_ms / 1000, 2)
            end_sec = round(end_ms / 1000, 2)

            raw_chunks.append({
                "chunk_index": i,
                "start_time": start_sec,
                "end_time": end_sec,
                "text": text
            })

            print(f"[Success] Processed Chunk {i} ({start_sec}s - {end_sec}s) | Text length: {len(text)} chars")

        if os.path.exists(chunks_temp_dir):
            try:
                os.rmdir(chunks_temp_dir)
            except Exception:
                pass

        # Unload Whisper model to free memory before loading SigLIP
        print("Unloading Whisper model to free memory...")
        del self.transcriber
        self.transcriber = None
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Generating SigLIP text embeddings for all chunks...")
        # Local import of embedding_manager to prevent memory fragmentation during Whisper loading phase
        from modules.embeddings import embedding_manager
        
        siglip_data_output = []
        for chunk in raw_chunks:
            # Encode with SigLIP
            siglip_vector = embedding_manager.get_text_embedding(chunk["text"]).tolist()
            siglip_data_output.append({
                "chunk_index": chunk["chunk_index"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "text": chunk["text"],
                "embedding": siglip_vector
            })

        # Write to both output paths for compatibility
        rag_json_path = os.path.join(output_dir, "rag_text_embeddings.json")
        siglip_json_path = os.path.join(output_dir, "siglip_text_embeddings.json")

        with open(rag_json_path, "w", encoding="utf-8") as f:
            json.dump(siglip_data_output, f, ensure_ascii=False, indent=4)

        with open(siglip_json_path, "w", encoding="utf-8") as f:
            json.dump(siglip_data_output, f, ensure_ascii=False, indent=4)

        print("--- [Complete] Transcription & SigLIP Embedding Pipeline Finished Successfully ---")
        print(f"SigLIP Text Embeddings saved to: {siglip_json_path}")
        
        return rag_json_path, siglip_json_path

dual_transcriber = DualEmbeddingTranscriber()
