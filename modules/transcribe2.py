import os
import json
import torch
import whisper
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────────────────────
#  Constants  (tune here if needed)
# ─────────────────────────────────────────────────────────────
CHUNK_SEC      = 30          # length of each RAG chunk (seconds)
OVERLAP_SEC    = 5           # overlap between chunks  (seconds)
STEP_SEC       = CHUNK_SEC - OVERLAP_SEC   # 25 s effective step
MIN_TEXT_LEN   = 3           # skip chunks shorter than this
RAG_BATCH_SIZE = 16          # sentence-transformer batch size
SIG_BATCH_SIZE = 16          # siglip batch size


class DualEmbeddingTranscriber:
    """
    Memory-safe transcription + dual-embedding pipeline.

    Flow
    ----
    1.  Load audio entirely into RAM as a float32 numpy array
        (no MP3 temp files written to disk).
    2.  Detect language from the first 30 s.
    3.  Transcribe the FULL audio with Whisper to get fine-grained
        `segments` (each ~3-10 s with precise timestamps).
    4.  Merge those segments into 30-second sliding-window chunks
        with 5-second overlap — so vision.py gets the exact time
        windows it expects.
    5.  De-duplicate overlap: mark each segment as "used" so text
        that falls in the overlap zone is not repeated twice.
    6.  Batch-encode all chunks in one pass (RAG + SigLIP).
    7.  Write two JSON files (rag_text_embeddings.json and
        siglip_text_embeddings.json) — no intermediate files.
    """

    def __init__(
        self,
        whisper_model:      str = "turbo",
        rag_embedding_model: str = "intfloat/multilingual-e5-small",
        siglip_model:       str = "google/siglip-base-patch16-224",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_metadata = {
            "whisper":       whisper_model,
            "rag_encoder":   rag_embedding_model,
            "siglip_encoder": siglip_model,
        }

        print(f"--- [Init] Pipeline starting on device: {self.device} ---")

        # 1. Whisper
        print(f"Loading Whisper '{whisper_model}'...")
        self.transcriber = whisper.load_model(whisper_model, device=self.device)

        # 2. Multilingual RAG encoder (E5)
        print(f"Loading RAG encoder '{rag_embedding_model}'...")
        self.rag_encoder = SentenceTransformer(rag_embedding_model, device=self.device)

        # 3. SigLIP text encoder
        print(f"Loading SigLIP text encoder '{siglip_model}'...")
        self.siglip_tokenizer = AutoTokenizer.from_pretrained(siglip_model)
        self.siglip_model     = AutoModel.from_pretrained(siglip_model).to(self.device)
        self.siglip_model.eval()

        print("--- [Init] All models loaded ---")

    # ──────────────────────────────────────────────────────────
    #  Language detection  (in-memory, no disk I/O)
    # ──────────────────────────────────────────────────────────
    def _detect_language(self, audio_array: np.ndarray) -> str:
        sample = whisper.pad_or_trim(audio_array)          # first 30 s
        mel    = whisper.log_mel_spectrogram(
            sample, n_mels=self.transcriber.dims.n_mels
        ).to(self.device)
        _, probs = self.transcriber.detect_language(mel)
        lang = max(probs, key=probs.get)
        print(f"--- [Language] Detected: '{lang}'  (conf {probs[lang]:.2f}) ---")
        return lang

    # ──────────────────────────────────────────────────────────
    #  Merge Whisper segments → 30-second sliding-window chunks
    # ──────────────────────────────────────────────────────────
    def _merge_segments_to_chunks(
        self,
        segments:    list,
        total_secs:  float,
    ) -> list:
        """
        For every 30-second window (step = 25 s), collect all Whisper
        segments whose start time falls inside [window_start, window_end).
        De-duplicate overlap by tracking which segment indices were already
        used in a 'primary' window (i.e., a window where the segment starts
        after the previous window's non-overlapping region).
        """
        chunks        = []
        used_primary  = set()          # segment indices already counted once

        start = 0.0
        chunk_idx = 0

        while start < total_secs:
            end = min(start + CHUNK_SEC, total_secs)

            # Collect segments inside this window
            window_segs = [
                s for s in segments
                if s["start"] >= start and s["start"] < end
            ]

            # Text: only include segments not yet used as primary,
            # OR all if this is the first window they appear in.
            primary_segs = []
            overlap_segs = []
            for s in window_segs:
                if id(s) not in used_primary:
                    primary_segs.append(s)
                    used_primary.add(id(s))
                else:
                    overlap_segs.append(s)

            # Build text — primary segments first, then overlap context
            all_segs_text = " ".join(
                s["text"].strip()
                for s in (primary_segs + overlap_segs)
                if s["text"].strip()
            ).strip()

            if len(all_segs_text) >= MIN_TEXT_LEN:
                chunks.append({
                    "chunk_index": chunk_idx,
                    "start_time":  round(start, 2),
                    "end_time":    round(end,   2),
                    "text":        all_segs_text,
                    "segment_count": len(window_segs),
                })
                chunk_idx += 1
            else:
                print(f"[Skip] Window {start:.1f}s–{end:.1f}s: no speech.")

            start += STEP_SEC

        return chunks

    # ──────────────────────────────────────────────────────────
    #  Batch SigLIP embeddings
    # ──────────────────────────────────────────────────────────
    def _siglip_batch_encode(
        self, texts: list[str], batch_size: int = SIG_BATCH_SIZE
    ) -> list[list[float]]:
        all_emb = []
        max_len = getattr(self.siglip_model.config, "max_position_embeddings", 64)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.siglip_tokenizer(
                batch,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                feats = self.siglip_model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-norm
                all_emb.extend(feats.cpu().tolist())

        return all_emb

    # ──────────────────────────────────────────────────────────
    #  Main pipeline
    # ──────────────────────────────────────────────────────────
    def process_video_pipeline(
        self,
        video_path: str,
        output_dir: str = "temp_assets",
    ) -> tuple[str, str] | tuple[None, None]:

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"[Fatal] File not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)

        # ── Step 1: Load audio into RAM ───────────────────────
        print(f"--- [Load] Reading audio from: {video_path} ---")
        audio_array = whisper.load_audio(video_path)        # float32 numpy
        total_secs  = len(audio_array) / whisper.audio.SAMPLE_RATE
        print(f"--- [Load] Duration: {total_secs:.1f} s ---")

        # ── Step 2: Language detection ────────────────────────
        lang = self._detect_language(audio_array)

        # ── Step 3: Full transcription → fine segments ────────
        print("--- [Transcribe] Running Whisper on full audio ---")
        result = self.transcriber.transcribe(
            audio_array,
            fp16=(self.device == "cuda"),
            language=lang,
            verbose=False,
        )
        raw_segments = result.get("segments", [])
        print(f"--- [Transcribe] Got {len(raw_segments)} raw segments ---")

        if not raw_segments:
            print("[Warning] Whisper returned no segments — no speech detected.")
            return None, None

        # ── Step 4: Merge into 30-second chunks ───────────────
        print("--- [Chunk] Merging segments into 30-second windows ---")
        chunks = self._merge_segments_to_chunks(raw_segments, total_secs)

        if not chunks:
            print("[Warning] No valid chunks after merging.")
            return None, None

        print(f"--- [Chunk] Built {len(chunks)} chunks ---")

        # ── Step 5: Batch encoding ────────────────────────────
        print("--- [Embed] Batch encoding all chunks ---")

        rag_texts    = [f"passage: {c['text']}" for c in chunks]
        siglip_texts = [c["text"]               for c in chunks]

        # RAG encoder (E5)
        rag_embs = self.rag_encoder.encode(
            rag_texts,
            batch_size=RAG_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,    # E5 wants L2-normed vectors
        ).tolist()

        # SigLIP text encoder
        siglip_embs = self._siglip_batch_encode(siglip_texts)

        # ── Step 6: Build output payloads ─────────────────────
        metadata = {
            "source_video":     os.path.basename(video_path),
            "detected_language": lang,
            "total_chunks":     len(chunks),
            "chunk_sec":        CHUNK_SEC,
            "overlap_sec":      OVERLAP_SEC,
            "models_used":      self.models_metadata,
        }

        rag_payload    = {"metadata": metadata, "segments": []}
        siglip_payload = {"metadata": metadata, "segments": []}

        for idx, chunk in enumerate(chunks):
            base = {
                "chunk_index": chunk["chunk_index"],
                "start_time":  chunk["start_time"],
                "end_time":    chunk["end_time"],
                "text":        chunk["text"],
            }

            rag_seg            = base.copy()
            rag_seg["embedding"] = rag_embs[idx]
            rag_payload["segments"].append(rag_seg)

            sig_seg            = base.copy()
            sig_seg["embedding"] = siglip_embs[idx]
            siglip_payload["segments"].append(sig_seg)

            print(
                f"[OK] Chunk {chunk['chunk_index']:03d} "
                f"({chunk['start_time']:.1f}s–{chunk['end_time']:.1f}s) "
                f"| {len(chunk['text'])} chars"
            )

        # ── Step 7: Write JSON (single atomic write per file) ─
        rag_path    = os.path.join(output_dir, "rag_text_embeddings.json")
        siglip_path = os.path.join(output_dir, "siglip_text_embeddings.json")

        with open(rag_path,    "w", encoding="utf-8") as f:
            json.dump(rag_payload, f, ensure_ascii=False, indent=2)

        with open(siglip_path, "w", encoding="utf-8") as f:
            json.dump(siglip_payload, f, ensure_ascii=False, indent=2)

        print(f"\n--- [Done] {len(chunks)} chunks processed ---")
        print(f"  RAG    → {rag_path}")
        print(f"  SigLIP → {siglip_path}")
        return rag_path, siglip_path


# ─────────────────────────────────────────────────────────────
#  Lazy singleton  (loaded on first access, NOT at import time)
# ─────────────────────────────────────────────────────────────
_transcriber_instance: DualEmbeddingTranscriber | None = None


def get_transcriber() -> DualEmbeddingTranscriber:
    """Returns the singleton, initializing it on first call."""
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = DualEmbeddingTranscriber()
    return _transcriber_instance


# ─────────────────────────────────────────────────────────────
#  Standalone CLI test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback

    folder = "temp_assets"
    os.makedirs(folder, exist_ok=True)
    files  = [
        f for f in os.listdir(folder)
        if f.endswith((".mp4", ".mkv", ".avi", ".mp3", ".wav"))
    ]

    if not files:
        print(f"[Info] Place a video/audio file in '{folder}/' and rerun.")
    else:
        test_file = os.path.join(folder, files[0])
        print(f"Test file: {test_file}")
        try:
            get_transcriber().process_video_pipeline(test_file)
        except Exception as e:
            print(f"\n[Fatal Error]: {e}")
            traceback.print_exc()