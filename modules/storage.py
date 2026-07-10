# modules/storage.py
"""
Vedex - Storage & Indexing Module
===================================
Persists preprocessing outputs (transcripts and visual embeddings) into
separate Qdrant vector collections for multimodal retrieval.

Pipeline:
    transcript_chunks.json  →  text_chunks  collection  (768-dim, Jina CLIP v2)
    visual_embeddings.json  →  video_frames collection  (768-dim, Jina CLIP v2)

This module does NOT implement retrieval, search, reranking, or LLM logic.
"""

from __future__ import annotations

import json
import logging
import time
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import DATABASE

# ─── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("vedex.storage")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
    logger.addHandler(_h)


# ─── Constants (read from config, safe fallbacks) ─────────────────────────────
TEXT_COLLECTION:   str = getattr(DATABASE, "text_collection",   "text_chunks")
VISION_COLLECTION: str = getattr(DATABASE, "vision_collection", "video_frames")
TEXT_DIM:    int = getattr(DATABASE, "text_vector_dim",   768)   # Jina CLIP v2
VISION_DIM:  int = getattr(DATABASE, "vision_vector_dim", 768)   # Jina CLIP v2
BATCH_SIZE:  int = getattr(DATABASE, "batch_size",        100)
MAX_RETRIES: int = 3
RETRY_BASE:  float = 2.0   # exponential base (seconds)
SCHEMA_VERSION: str = "1.0"


# ─── Exceptions ───────────────────────────────────────────────────────────────

class ConfigurationError(ValueError):
    """Raised when embedding dimensions in data do not match config.py values."""


class SchemaError(ValueError):
    """Raised when a required JSON file fails schema validation."""


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class CollectionStats:
    """Per-collection indexing statistics."""
    name: str
    uploaded: int = 0
    skipped: int = 0
    failed: int = 0
    duration_seconds: float = 0.0


@dataclass
class IndexingResult:
    """
    Structured result object returned by StorageManager.index_preprocessing_outputs().
    """
    success: bool
    video_id: str
    text_stats: CollectionStats = field(default_factory=lambda: CollectionStats(TEXT_COLLECTION))
    image_stats: CollectionStats = field(default_factory=lambda: CollectionStats(VISION_COLLECTION))
    total_duration_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "video_id": self.video_id,
            "collections": {
                self.text_stats.name: {
                    "uploaded": self.text_stats.uploaded,
                    "skipped": self.text_stats.skipped,
                    "failed": self.text_stats.failed,
                    "duration_seconds": round(self.text_stats.duration_seconds, 2),
                },
                self.image_stats.name: {
                    "uploaded": self.image_stats.uploaded,
                    "skipped": self.image_stats.skipped,
                    "failed": self.image_stats.failed,
                    "duration_seconds": round(self.image_stats.duration_seconds, 2),
                },
            },
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "warnings": self.warnings,
            "errors": self.errors,
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "schema_version": SCHEMA_VERSION,
        }


# ─── Schema validators ────────────────────────────────────────────────────────

class SchemaError(ValueError):
    """Raised when a required JSON file fails schema validation."""


def _validate_transcript_chunks(data: Any, path: Path) -> List[Dict]:
    if not isinstance(data, list):
        raise SchemaError(f"{path.name}: root must be a JSON array.")
    if len(data) == 0:
        raise SchemaError(f"{path.name}: array is empty.")
    required = {"text", "embedding"}
    for idx, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise SchemaError(f"{path.name}[{idx}]: missing required keys {missing}.")
        if not isinstance(item["embedding"], list) or len(item["embedding"]) == 0:
            raise SchemaError(f"{path.name}[{idx}]: 'embedding' must be a non-empty list.")
    return data


def _validate_visual_embeddings(data: Any, path: Path) -> List[Dict]:
    if not isinstance(data, list):
        raise SchemaError(f"{path.name}: root must be a JSON array.")
    if len(data) == 0:
        raise SchemaError(f"{path.name}: array is empty.")
    for idx, item in enumerate(data):
        if "chunk_index" not in item:
            raise SchemaError(f"{path.name}[{idx}]: missing required key 'chunk_index'.")
        emb = item.get("image_embedding", item.get("embedding"))
        if emb is None:
            raise SchemaError(f"{path.name}[{idx}]: missing required keys 'image_embedding' or 'embedding'.")
        if not isinstance(emb, list) or len(emb) == 0:
            raise SchemaError(f"{path.name}[{idx}]: embedding must be a non-empty list.")
    return data


def _validate_report(data: Any, path: Path) -> Dict:
    if not isinstance(data, dict):
        raise SchemaError(f"{path.name}: root must be a JSON object.")
    return data


def _load_and_validate(path: Path, validator) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    if path.stat().st_size == 0:
        raise SchemaError(f"{path.name} is empty (0 bytes).")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return validator(data, path)


# ─── ID generation ────────────────────────────────────────────────────────────

def _deterministic_id(video_id: str, collection: str, chunk_index: int) -> str:
    """
    Generate a stable UUID-format point ID from (video_id, collection, chunk_index).
    Idempotent: same inputs always produce the same UUID, preventing duplicates.
    """
    seed = f"{video_id}::{collection}::{chunk_index}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


# ─── StorageManager ───────────────────────────────────────────────────────────

class StorageManager:
    """
    Persists Vedex preprocessing outputs into Qdrant vector collections.

    Responsibilities:
    - Validate transcript and visual embedding JSON files.
    - Auto-create or verify Qdrant collections.
    - Batch-upsert points with exponential-backoff retry.
    - Generate deterministic, idempotent point IDs.
    - Write a structured indexing report.

    Not responsible for retrieval, search, or LLM logic.
    """

    def __init__(self) -> None:
        self._client: Optional[QdrantClient] = None
        self.report_dir = Path("temp_assets")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ── Qdrant client (lazy init) ─────────────────────────────────────────────

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            url = getattr(DATABASE, "qdrant_url",     "http://localhost:6333")
            key = getattr(DATABASE, "qdrant_api_key", "") or None
            logger.info(f"Connecting to Qdrant at {url} ...")
            self._client = QdrantClient(url=url, api_key=key, timeout=30)
        return self._client

    # ── Collection management ─────────────────────────────────────────────────

    def _ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        """
        Create a Qdrant collection if it does not exist, or verify its vector
        dimension matches the expected size.

        Raises:
            ValueError: If the collection exists but uses a different vector size.
        """
        try:
            exists = self.client.collection_exists(name)
        except Exception as e:
            logger.warning(f"Could not query collection existence for '{name}': {e}. Attempting creation.")
            exists = False

        if exists:
            info = self.client.get_collection(name)
            actual_size = info.config.params.vectors.size  # type: ignore[union-attr]
            if actual_size != vector_size:
                raise ValueError(
                    f"Collection '{name}' already exists with vector size {actual_size}, "
                    f"expected {vector_size}. Delete it manually or set force_rebuild=True."
                )
            logger.info(f"Collection '{name}' already exists (dim={vector_size}). Skipping creation.")
        else:
            logger.info(f"Creating collection '{name}' (dim={vector_size}, distance={distance.name}) ...")
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance),
            )
            logger.info(f"Collection '{name}' created successfully.")

    def init_collections(self, text_dim: int = TEXT_DIM, vision_dim: int = VISION_DIM) -> None:
        """Initialize both text and vision Qdrant collections."""
        self._ensure_collection(TEXT_COLLECTION,   text_dim)
        self._ensure_collection(VISION_COLLECTION, vision_dim)

    # ── Batch upload with retry ───────────────────────────────────────────────

    def _upsert_batch_with_retry(
        self,
        collection: str,
        points: List[models.PointStruct],
        stats: CollectionStats,
    ) -> None:
        """
        Upsert a batch of points with exponential backoff retry (up to MAX_RETRIES).
        Updates stats.uploaded / stats.failed in place.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.client.upsert(collection_name=collection, points=points, wait=True)
                stats.uploaded += len(points)
                return
            except (UnexpectedResponse, Exception) as exc:
                if attempt == MAX_RETRIES:
                    logger.error(f"Batch upload to '{collection}' failed after {MAX_RETRIES} retries: {exc}")
                    stats.failed += len(points)
                    return
                wait = RETRY_BASE ** attempt
                logger.warning(f"Batch upload attempt {attempt} failed ({exc}). Retrying in {wait:.1f}s …")
                time.sleep(wait)

    # ── Payload builders ──────────────────────────────────────────────────────

    @staticmethod
    def _build_text_payload(
        chunk: Dict,
        chunk_idx: int,
        video_id: str,
        report_meta: Dict,
    ) -> Dict[str, Any]:
        return {
            "video_id":    video_id,
            "chunk_id":    chunk_idx,
            "transcript":  chunk.get("text", ""),
            "start_time":  chunk.get("start_time", chunk.get("start", 0.0)),
            "end_time":    chunk.get("end_time",   chunk.get("end",   0.0)),
            "metadata":    report_meta,
            "schema_version": SCHEMA_VERSION,
            "indexed_at":  datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @staticmethod
    def _build_vision_payload(
        vis: Dict,
        video_id: str,
        report_meta: Dict,
    ) -> Dict[str, Any]:
        return {
            "video_id":        video_id,
            "chunk_id":        vis.get("chunk_index", 0),
            "frame_time":      vis.get("frame_time",  vis.get("timestamp", 0.0)),
            "frame_index":     vis.get("frame_index", 0),
            "keyframe_path":   vis.get("keyframe_path", ""),
            "ocr_text":        vis.get("semantic_text", vis.get("ocr_text", "")),
            "quality_score":   vis.get("quality_score",   0.0),
            "similarity_score":vis.get("similarity_score", 0.0),
            "ocr_score":       vis.get("ocr_score",        0.0),
            "start_time":      vis.get("start_time", 0.0),
            "end_time":        vis.get("end_time",   0.0),
            "metadata":        report_meta,
            "schema_version":  SCHEMA_VERSION,
            "indexed_at":      datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    # ── Existence check helpers ───────────────────────────────────────────────

    def _point_exists(self, collection: str, point_id: str) -> bool:
        """Check whether a point with the given ID already exists in the collection."""
        try:
            result = self.client.retrieve(collection_name=collection, ids=[point_id], with_payload=False)
            return len(result) > 0
        except Exception:
            return False

    # ── Core indexing methods ─────────────────────────────────────────────────

    def _index_text(
        self,
        chunks: List[Dict],
        video_id: str,
        report_meta: Dict,
        stats: CollectionStats,
    ) -> None:
        """Upload text chunks to the text_chunks collection."""
        t0 = time.time()
        actual_text_dim = len(chunks[0].get("embedding", []))

        batch: List[models.PointStruct] = []

        with tqdm(total=len(chunks), desc=f"Indexing {TEXT_COLLECTION}", unit="chunk") as pbar:
            for idx, chunk in enumerate(chunks):
                emb = chunk.get("embedding")
                if not emb or len(emb) != actual_text_dim:
                    logger.warning(f"Chunk {idx}: invalid or missing text embedding. Skipping.")
                    stats.skipped += 1
                    pbar.update(1)
                    continue

                point_id = _deterministic_id(video_id, TEXT_COLLECTION, idx)

                if self._point_exists(TEXT_COLLECTION, point_id):
                    logger.debug(f"Chunk {idx}: already indexed ({point_id}). Skipping.")
                    stats.skipped += 1
                    pbar.update(1)
                    continue

                batch.append(
                    models.PointStruct(
                        id=point_id,
                        vector=emb,
                        payload=self._build_text_payload(chunk, idx, video_id, report_meta),
                    )
                )

                if len(batch) >= BATCH_SIZE:
                    self._upsert_batch_with_retry(TEXT_COLLECTION, batch, stats)
                    batch.clear()

                pbar.update(1)

            # Flush remaining
            if batch:
                self._upsert_batch_with_retry(TEXT_COLLECTION, batch, stats)

        stats.duration_seconds = time.time() - t0
        logger.info(
            f"[{TEXT_COLLECTION}] Uploaded: {stats.uploaded} | "
            f"Skipped: {stats.skipped} | Failed: {stats.failed} | "
            f"Time: {stats.duration_seconds:.2f}s"
        )

    def _index_vision(
        self,
        visual_embeddings: List[Dict],
        video_id: str,
        report_meta: Dict,
        stats: CollectionStats,
    ) -> None:
        """Upload vision frames to the video_frames collection (Jina CLIP v2 768-dim)."""
        t0 = time.time()
        actual_vision_dim = len(visual_embeddings[0].get("image_embedding", visual_embeddings[0].get("embedding", [])))

        batch: List[models.PointStruct] = []

        with tqdm(total=len(visual_embeddings), desc=f"Indexing {VISION_COLLECTION}", unit="frame") as pbar:
            for vis in visual_embeddings:
                chunk_idx = vis.get("chunk_index", 0)
                emb = vis.get("image_embedding", vis.get("embedding"))
                if not emb or len(emb) != actual_vision_dim:
                    logger.warning(f"Frame chunk {chunk_idx}: invalid embedding. Skipping.")
                    stats.skipped += 1
                    pbar.update(1)
                    continue

                point_id = _deterministic_id(video_id, VISION_COLLECTION, chunk_idx)

                if self._point_exists(VISION_COLLECTION, point_id):
                    logger.debug(f"Frame {chunk_idx}: already indexed ({point_id}). Skipping.")
                    stats.skipped += 1
                    pbar.update(1)
                    continue

                batch.append(
                    models.PointStruct(
                        id=point_id,
                        vector=emb,
                        payload=self._build_vision_payload(vis, video_id, report_meta),
                    )
                )

                if len(batch) >= BATCH_SIZE:
                    self._upsert_batch_with_retry(VISION_COLLECTION, batch, stats)
                    batch.clear()

                pbar.update(1)

            # Flush remaining
            if batch:
                self._upsert_batch_with_retry(VISION_COLLECTION, batch, stats)

        stats.duration_seconds = time.time() - t0
        logger.info(
            f"[{VISION_COLLECTION}] Uploaded: {stats.uploaded} | "
            f"Skipped: {stats.skipped} | Failed: {stats.failed} | "
            f"Time: {stats.duration_seconds:.2f}s"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_inputs(
        self,
        transcript_path: Path,
        visual_path: Path,
        report_path: Path,
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Load and validate all required preprocessing output files.

        Returns:
            Tuple of (transcript_chunks, visual_embeddings, report_meta).

        Raises:
            FileNotFoundError: If any required file is missing.
            SchemaError: If any file fails schema validation.
        """
        logger.info("Validating input files …")
        chunks   = _load_and_validate(transcript_path, _validate_transcript_chunks)
        visuals  = _load_and_validate(visual_path,     _validate_visual_embeddings)
        report   = _load_and_validate(report_path,     _validate_report)
        logger.info(
            f"Inputs validated: {len(chunks)} text chunks, {len(visuals)} visual frames."
        )
        return chunks, visuals, report

    def index_preprocessing_outputs(
        self,
        transcript_path: Path = Path("temp_assets/transcript_chunks.json"),
        visual_path:     Path = Path("temp_assets/visual_embeddings.json"),
        report_path:     Path = Path("temp_assets/preprocessing_report.json"),
        video_id:        Optional[str] = None,
    ) -> IndexingResult:
        """
        Full indexing pipeline: validate → init collections → upload → report.

        Args:
            transcript_path: Path to transcript_chunks.json.
            visual_path:     Path to visual_embeddings.json.
            report_path:     Path to preprocessing_report.json.
            video_id:        Optional explicit video identifier.
                             Falls back to stem of video path in report.

        Returns:
            IndexingResult with detailed statistics.
        """
        pipeline_start = time.time()
        result = IndexingResult(success=False, video_id=video_id or "unknown")

        # ── 1. Validate inputs ────────────────────────────────────────────────
        try:
            chunks, visuals, report = self.validate_inputs(
                transcript_path, visual_path, report_path
            )
        except (FileNotFoundError, SchemaError, json.JSONDecodeError) as exc:
            msg = f"Input validation failed: {exc}"
            logger.error(msg)
            result.errors.append(msg)
            self._write_report(result)
            return result

        # ── 2. Resolve video_id ───────────────────────────────────────────────
        if not video_id:
            meta = report.get("video_metadata", {})
            raw_path = meta.get("path", "")
            video_id = Path(raw_path).stem if raw_path else "unknown_video"
        result.video_id = video_id

        # Extract compact metadata for payloads (avoid huge nested objects)
        report_meta: Dict[str, Any] = {
            "video_path":    report.get("video_metadata", {}).get("path", ""),
            "video_source":  report.get("video_metadata", {}).get("source", ""),
            "gpu_detected":  report.get("video_metadata", {}).get("gpu_detected", False),
        }

        # ── 3. Validate embedding dimensions against config ───────────────────
        actual_text_dim   = len(chunks[0].get("embedding", []))
        actual_vision_dim = len(visuals[0].get("image_embedding", visuals[0].get("embedding", [])))

        if actual_text_dim == 0:
            msg = "Text embedding dimension is 0. Cannot index."
            result.errors.append(msg)
            logger.error(msg)
            self._write_report(result)
            return result

        if actual_vision_dim == 0:
            msg = "Vision embedding dimension is 0. Cannot index."
            result.errors.append(msg)
            logger.error(msg)
            self._write_report(result)
            return result

        if actual_text_dim != TEXT_DIM:
            msg = (
                f"ConfigurationError: text embedding dimension in data ({actual_text_dim}) "
                f"does not match config TEXT_DIM ({TEXT_DIM}). "
                f"Update config.py or re-run transcription."
            )
            result.errors.append(msg)
            logger.error(msg)
            self._write_report(result)
            return result

        if actual_vision_dim != VISION_DIM:
            msg = (
                f"ConfigurationError: vision embedding dimension in data ({actual_vision_dim}) "
                f"does not match config VISION_DIM ({VISION_DIM}). "
                f"Ensure vision.py uses Jina CLIP v2 (768-dim) and update config.py if needed."
            )
            result.errors.append(msg)
            logger.error(msg)
            self._write_report(result)
            return result

        # ── 4. Ensure collections exist ───────────────────────────────────────
        try:
            self.init_collections(
                text_dim=actual_text_dim,
                vision_dim=actual_vision_dim,
            )
        except ValueError as exc:
            msg = f"Collection dimension mismatch: {exc}"
            result.errors.append(msg)
            logger.error(msg)
            self._write_report(result)
            return result
        except Exception as exc:
            msg = f"Failed to connect to Qdrant or create collections: {exc}"
            result.errors.append(msg)
            logger.error(msg)
            self._write_report(result)
            return result

        # ── 5. Index text chunks ──────────────────────────────────────────────
        logger.info(f"=== Indexing text chunks into '{TEXT_COLLECTION}' ===")
        self._index_text(chunks, video_id, report_meta, result.text_stats)

        # ── 6. Index visual embeddings ────────────────────────────────────────
        logger.info(f"=== Indexing visual frames into '{VISION_COLLECTION}' ===")
        self._index_vision(visuals, video_id, report_meta, result.image_stats)

        # ── 7. Finalize ───────────────────────────────────────────────────────
        result.total_duration_seconds = time.time() - pipeline_start
        result.success = result.text_stats.failed == 0 and result.image_stats.failed == 0

        self._write_report(result)
        self._print_summary(result)

        return result

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _write_report(self, result: IndexingResult) -> None:
        """Persist structured indexing report to temp_assets/indexing_report.json."""
        out = self.report_dir / "indexing_report.json"
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=4)
            logger.info(f"Indexing report saved to {out}")
        except Exception as exc:
            logger.error(f"Failed to write indexing report: {exc}")

    @staticmethod
    def _print_summary(result: IndexingResult) -> None:
        ts = result.text_stats
        vs = result.image_stats
        print("\n" + "=" * 65)
        print("VEDEX STORAGE & INDEXING SUMMARY")
        print("=" * 65)
        print(f"{'Collection':<22} | {'Uploaded':>8} | {'Skipped':>7} | {'Failed':>6} | {'Time(s)':>8}")
        print("-" * 65)
        print(f"{ts.name:<22} | {ts.uploaded:>8} | {ts.skipped:>7} | {ts.failed:>6} | {ts.duration_seconds:>8.2f}")
        print(f"{vs.name:<22} | {vs.uploaded:>8} | {vs.skipped:>7} | {vs.failed:>6} | {vs.duration_seconds:>8.2f}")
        print("-" * 65)
        print(f"{'TOTAL':<22} | {ts.uploaded + vs.uploaded:>8} | {ts.skipped + vs.skipped:>7} | "
              f"{ts.failed + vs.failed:>6} | {result.total_duration_seconds:>8.2f}")
        status = "SUCCESS" if result.success else "COMPLETED WITH ERRORS"
        print(f"\nStatus: {status}")
        if result.warnings:
            print(f"Warnings ({len(result.warnings)}): " + "; ".join(result.warnings))
        if result.errors:
            print(f"Errors  ({len(result.errors)}): " + "; ".join(result.errors))
        print("=" * 65 + "\n")


# ─── Global singleton ──────────────────────────────────────────────────────────
storage_manager = StorageManager()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vedex Storage & Indexing")
    parser.add_argument("--transcript",  default="temp_assets/transcript_chunks.json")
    parser.add_argument("--visual",      default="temp_assets/visual_embeddings.json")
    parser.add_argument("--report",      default="temp_assets/preprocessing_report.json")
    parser.add_argument("--video-id",    default=None)
    args = parser.parse_args()

    result = storage_manager.index_preprocessing_outputs(
        transcript_path=Path(args.transcript),
        visual_path=Path(args.visual),
        report_path=Path(args.report),
        video_id=args.video_id,
    )
    raise SystemExit(0 if result.success else 1)
