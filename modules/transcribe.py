import os
import json
import torch
import whisper
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

DEFAULT_WHISPER_MODEL = os.getenv("TRANSCRIPTION_WHISPER_MODEL", "base")


class DualEmbeddingTranscriber:
    def __init__(
        self, 
        whisper_model=DEFAULT_WHISPER_MODEL,
        rag_embedding_model="intfloat/multilingual-e5-small", 
        siglip_model="google/siglip-base-patch16-224"
    ):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- [Init] Pipeline initializing on device: {self.device} ---")
        
        # 1. Load the configured Whisper model
        print(f"Loading Whisper model '{whisper_model}'...")
        self.transcriber = whisper.load_model(whisper_model, device=self.device)
        
        # 2. Load Multilingual SentenceTransformer
        print(f"Loading RAG Embedding model '{rag_embedding_model}'...")
        self.rag_encoder = SentenceTransformer(rag_embedding_model, device=self.device)
        
        # 3. Load SigLIP Text Encoder
        print(f"Loading SigLIP Text Encoder '{siglip_model}'...")
        self.siglip_tokenizer = AutoTokenizer.from_pretrained(siglip_model)
        self.siglip_model = AutoModel.from_pretrained(siglip_model).to(self.device)
        self.siglip_model.eval() 
        
        print("--- [Init] All Models Loaded Successfully ---")

    def detect_video_language(self, audio_chunk_path: str) -> str:
      
        audio = whisper.load_audio(audio_chunk_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.transcriber.dims.n_mels).to(self.device)
        
        _, probs = self.transcriber.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        print(f"--- [Language Detection] Locked Language: '{detected_lang}' (Confidence: {probs[detected_lang]:.2f}) ---")
        return detected_lang

    def get_siglip_text_embedding(self, text: str) -> list:
        max_len = getattr(self.siglip_model.config, "max_position_embeddings", 64)
        
        inputs = self.siglip_tokenizer(
            [text], 
            padding="max_length", 
            max_length=max_len, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.siglip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features[0].cpu().tolist()

    def process_video_pipeline(self, video_path: str, output_dir: str = "temp_assets", chunk_ms: int = 30000, overlap_ms: int = 5000):
       
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        chunks_temp_dir = os.path.join(output_dir, "temp_audio_chunks")
        os.makedirs(chunks_temp_dir, exist_ok=True)

        audio = AudioSegment.from_file(video_path)
        step_ms = chunk_ms - overlap_ms
        
        rag_data_output = []
        siglip_data_output = []
        
        current_video_language = None

        print(f"--- [Processing] Starting sliding window transcription ({len(audio)/1000:.2f}s total duration) ---")

        for i, start_ms in enumerate(range(0, len(audio), step_ms)):
            end_ms = min(start_ms + chunk_ms, len(audio))
            chunk = audio[start_ms:end_ms]
            
            chunk_path = os.path.join(chunks_temp_dir, f"chunk_{i}.mp3")
            chunk.export(chunk_path, format="mp3")

            if current_video_language is None:
                current_video_language = self.detect_video_language(chunk_path)

            result = self.transcriber.transcribe(
                chunk_path, 
                fp16=(self.device == "cuda"),
                language=current_video_language
            )
            
            raw_text = result.get("text")
            text = raw_text.strip() if raw_text is not None else ""
            
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

            if not text or len(text) < 3:
                print(f"[Skip] Chunk {i} ({start_ms/1000:.1f}s - {end_ms/1000:.1f}s): No speech detected.")
                continue

            rag_formatted_text = f"passage: {text}"
            rag_vector = self.rag_encoder.encode(rag_formatted_text).tolist()

            siglip_vector = self.get_siglip_text_embedding(text)

            start_sec = round(start_ms / 1000, 2)
            end_sec = round(end_ms / 1000, 2)

            rag_data_output.append({
                "chunk_index": i,
                "start_time": start_sec,
                "end_time": end_sec,
                "text": text,
                "embedding": rag_vector
            })

            siglip_data_output.append({
                "chunk_index": i,
                "start_time": start_sec,
                "end_time": end_sec,
                "text": text,
                "embedding": siglip_vector
            })

            print(f"[Success] Processed Chunk {i} ({start_sec}s - {end_sec}s) | Text length: {len(text)} chars")

        if os.path.exists(chunks_temp_dir):
            os.rmdir(chunks_temp_dir)

        rag_json_path = os.path.join(output_dir, "rag_text_embeddings.json")
        siglip_json_path = os.path.join(output_dir, "siglip_text_embeddings.json")

        with open(rag_json_path, "w", encoding="utf-8") as f:
            json.dump(rag_data_output, f, ensure_ascii=False, indent=4)

        with open(siglip_json_path, "w", encoding="utf-8") as f:
            json.dump(siglip_data_output, f, ensure_ascii=False, indent=4)

        print("--- [Complete] Transcription & Dual Embedding Pipeline Finished Successfully ---")
        print(f"1. RAG Embeddings saved to: {rag_json_path}")
        print(f"2. SigLIP Text Embeddings saved to: {siglip_json_path}")
        
        return rag_json_path, siglip_json_path

dual_transcriber = DualEmbeddingTranscriber()

if __name__ == "__main__":
    try:
        folder = "temp_assets"
        os.makedirs(folder, exist_ok=True)
        files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.mkv', '.avi', '.mp3', '.wav'))]
        
        if not files:
            print(f"[Info] Please place a test video file inside '{folder}/' directory to run standalone test.")
        else:
            test_video = os.path.join(folder, files[0])
            print(f"Starting pipeline on test file: {test_video}")
            dual_transcriber.process_video_pipeline(test_video)
            
    except Exception as e:
        print(f"\n[Pipeline Error]: {str(e)}")
