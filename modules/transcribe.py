import os
import json
import subprocess
import sys
from pydub import AudioSegment

DEFAULT_WHISPER_MODEL = os.getenv("TRANSCRIPTION_WHISPER_MODEL", "large-v3")

class DualEmbeddingTranscriber:
    def __init__(
        self, 
        whisper_model=DEFAULT_WHISPER_MODEL,
        rag_embedding_model=None, # kept for backward compatibility
        siglip_model=None # kept for backward compatibility
    ):
        self.device = "cuda" if os.environ.get("CUDA_AVAILABLE") == "True" else "cpu"
        self.whisper_model = whisper_model
        print(f"--- [Init] Pipeline initializing on device: {self.device} ---")

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

            # Execute transcription of this chunk in a separate clean process to prevent MKL memory allocation errors
            cmd = [
                sys.executable,
                "modules/transcribe_chunk.py",
                "--model", self.whisper_model,
                "--audio", chunk_path,
            ]
            if current_video_language:
                cmd.extend(["--language", current_video_language])
                
            res = subprocess.run(cmd, capture_output=True, text=True)
            text = ""
            
            if res.returncode != 0:
                print(f"[Error] Chunk {i} transcription failed: {res.stderr}")
            else:
                try:
                    data = json.loads(res.stdout.strip())
                    text = data.get("text", "")
                    if current_video_language is None and data.get("language"):
                        current_video_language = data["language"]
                        print(f"--- [Language Detection] Locked Language: '{current_video_language}' (Confidence: {data.get('language_probability', 0.0):.2f}) ---")
                except Exception as je:
                    print(f"[Error] Chunk {i} output parsing failed: {je}")

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
