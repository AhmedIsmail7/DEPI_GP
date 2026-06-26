import whisper
import os
import torch
import json
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer

class Transcriber:
    def __init__(self, model_size="base"):
        """
        Initialization with GPU support and Embedding model readiness.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Whisper
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        
        # Load Embedding Model
        print("Loading Embedding model 'all-MiniLM-L6-v2'...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Models loaded successfully.")

    def process_audio_with_overlap(self, video_path: str, chunk_ms: int = 30000, overlap_ms: int = 5000):
        """
        Sliding window transcription + Embedding generation pipeline.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        audio = AudioSegment.from_file(video_path)
        step_ms = chunk_ms - overlap_ms
        chunks_metadata = []

        print(f"Starting sliding window transcription: {len(audio)}ms total.")

        # Create temp dir for chunks
        os.makedirs("temp_assets/chunks", exist_ok=True)

        for i, start_ms in enumerate(range(0, len(audio), step_ms)):
            end_ms = start_ms + chunk_ms
            chunk = audio[start_ms:end_ms]
            
            # Save temporary chunk
            chunk_path = f"temp_assets/chunks/chunk_{i}.mp3"
            chunk.export(chunk_path, format="mp3")

            # Transcription
            result = self.model.transcribe(
                chunk_path, 
                fp16=(self.device == "cuda")
            )
            
            text = result["text"].strip()
            
            # Generate Embeddings (384-d vector)
            vector = self.embedding_model.encode(text).tolist()
            
            # Store metadata + vector
            chunks_metadata.append({
                "index": i,
                "start": start_ms / 1000,
                "end": end_ms / 1000,
                "text": text,
                "embedding": vector
            })
            
            # Cleanup temp chunk
            os.remove(chunk_path)
            print(f"Processed chunk {i}: {start_ms/1000}s - {end_ms/1000}s")

        # Clean up temp directory
        os.rmdir("temp_assets/chunks")
        return chunks_metadata

# Singleton instance
transcriber_engine = Transcriber(model_size="base")


# For only trying the module directly (not important for the main pipeline & not for import)
if __name__ == "__main__":
    try:
        # Dynamic file selection
        folder = "temp_assets"
        files = [f for f in os.listdir(folder) if f.endswith('.mp4')]
        if not files:
            raise FileNotFoundError("No video files found in temp_assets/")
        
        test_file = os.path.join(folder, files[0]) 
        print(f"Testing with file: {test_file}")
        
        # Run pipeline
        results = transcriber_engine.process_audio_with_overlap(test_file)
        
        # Save output
        with open("temp_assets/transcript_chunks.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        print(f"Pipeline complete. Data saved to temp_assets/transcript_chunks.json")
        print(f"Total chunks processed: {len(results)}")
        
    except Exception as e:
        print(f"Pipeline error: {e}")