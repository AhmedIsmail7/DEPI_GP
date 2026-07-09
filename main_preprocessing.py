# main_preprocessing.py
"""
Vedex Preprocessing Orchestrator
=================================
Coordination script that runs the entire Vedex preprocessing pipeline:
1. Video Ingestion (ingest.py)
2. Audio Transcription (transcribe.py)
3. Vision Alignment (vision.py)

Supports resumable execution, stage skip checks, and generates a structured report.
"""

import os
import sys
import time
import json
import shutil
import argparse
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

import re
import torch

# Configure logger
logger = logging.getLogger("vedex.orchestrator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PipelineState:
    """
    Data class to track state, durations, outputs, and errors of the preprocessing pipeline.
    """
    def __init__(self):
        self.completed_stages: List[str] = []
        self.skipped_stages: List[str] = []
        self.stage_durations: Dict[str, float] = {}
        self.generated_files: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.total_duration: float = 0.0
        self.gpu_available: bool = False
        self.video_path: Optional[str] = None
        self.video_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_metadata": {
                "path": self.video_path,
                "source": self.video_source,
                "gpu_detected": self.gpu_available
            },
            "stages": {
                "executed": self.completed_stages,
                "skipped": self.skipped_stages
            },
            "durations": {
                **self.stage_durations,
                "total": round(self.total_duration, 2)
            },
            "outputs": self.generated_files,
            "warnings": self.warnings,
            "errors": self.errors
        }

class PreprocessingOrchestrator:
    def __init__(self, force: bool = False):
        self.force = force
        self.state = PipelineState()
        self.temp_dir = Path("temp_assets")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.temp_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Central GPU detection
        self.state.gpu_available = torch.cuda.is_available()
        logger.info(f"GPU Detection: CUDA is {'available' if self.state.gpu_available else 'UNAVAILABLE'}")

    def clean_temporary_files(self):
        """
        Cleanup temporary processing artifacts (e.g. temp audio directories),
        while strictly preserving cache directory contents.
        """
        logger.info("Cleaning up temporary directories...")
        temp_audio = self.temp_dir / "temp_audio_chunks"
        if temp_audio.exists():
            try:
                shutil.rmtree(temp_audio)
                logger.info("Cleaned temporary audio chunk directory.")
            except Exception as e:
                self.state.warnings.append(f"Failed to clean temporary audio dir: {e}")
                logger.warning(f"Failed to clean temporary audio dir: {e}")

    def validate_json_file(self, path: Path, expected_type: type = list) -> bool:
        """
        Helper to validate a JSON file exists, is non-empty, and has correct base type.
        """
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return isinstance(data, expected_type)
        except Exception:
            return False

    def run_ingestion(self, url: Optional[str], video_path: Optional[str]) -> Optional[str]:
        """
        Stage 1: Video Ingestion
        """
        logger.info("=== Stage 1: Video Ingestion ===")
        start_time = time.time()
        
        # Check if local video path was specified directly
        if video_path:
            p = Path(video_path)
            if not p.exists():
                raise FileNotFoundError(f"Specified local video file does not exist: {video_path}")
            self.state.video_path = str(p)
            self.state.video_source = "local"
            self.state.skipped_stages.append("ingestion")
            logger.info(f"Ingestion skipped: Using local video {video_path}")
            self.state.stage_durations["ingestion"] = 0.0
            return self.state.video_path

        # URL provided
        if not url:
            raise ValueError("Either a local video path or a remote URL must be provided.")

        self.state.video_source = url
        # Determine expected video ID to check for existing file
        video_id = None
        if "youtube.com" in url or "youtu.be" in url:
            match = re.search(r'(?:v=|\/|si=)([a-zA-Z0-9-_]{11})', url)
            if match:
                video_id = match.group(1)
        elif "drive.google.com" in url:
            try:
                from modules.ingest import SecureVideoIngestion
                temp_ingest = SecureVideoIngestion()
                file_id = temp_ingest._extract_gdrive_file_id(url)
                video_id = f"gdrive_{file_id}"
            except Exception:
                pass

        from config import INGESTION
        if video_id:
            expected_output = Path(INGESTION.output_dir) / f"{video_id}.mp4"
        else:
            expected_output = Path(INGESTION.output_dir) / "yt_video.mp4"

        # Resumability check
        if not self.force and expected_output.exists() and expected_output.stat().st_size > 0:
            self.state.skipped_stages.append("ingestion")
            self.state.video_path = str(expected_output)
            logger.info(f"Ingestion skipped: {expected_output.name} already exists.")
            self.state.stage_durations["ingestion"] = 0.0
            return self.state.video_path

        # Run ingest module
        try:
            from config import INGESTION
            from modules.ingest import SecureVideoIngestion
            
            ingestion = SecureVideoIngestion(
                output_dir=INGESTION.output_dir,
                metadata_dir=INGESTION.metadata_dir
            )
            result = ingestion.ingest_video(url)
            
            if not result.get("success"):
                raise RuntimeError(f"Ingestion failed: {result.get('error')}")
                
            downloaded = result.get("video_path")
            if not downloaded or not os.path.exists(downloaded):
                raise FileNotFoundError("Ingestion finished but output video path is missing.")
            
            self.state.video_path = downloaded
            self.state.completed_stages.append("ingestion")
            self.state.generated_files.append(downloaded)
            
            duration = time.time() - start_time
            self.state.stage_durations["ingestion"] = round(duration, 2)
            logger.info(f"Ingestion complete: {downloaded} ({duration:.2f}s)")
            return downloaded
        except Exception as e:
            self.state.errors.append(f"Ingestion stage failed: {e}")
            raise e

    def run_transcription(self, video_path: str) -> Path:
        """
        Stage 2: Audio Transcription
        """
        logger.info("=== Stage 2: Audio Transcription ===")
        start_time = time.time()
        
        expected_output = self.temp_dir / "transcript_chunks.json"
        
        # Resumability check
        if not self.force and self.validate_json_file(expected_output):
            self.state.skipped_stages.append("transcription")
            logger.info("Transcription skipped: transcript_chunks.json is already present and valid.")
            self.state.stage_durations["transcription"] = 0.0
            return expected_output

        # Validate input video file exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Input video file for transcription missing: {video_path}")

        try:
            from modules.transcribe import dual_transcriber
            
            logger.info("Starting transcription pipeline (Whisper + SentenceTransformer)...")
            # Override transcribe device dynamically using startup gpu detection
            dual_transcriber.device = "cuda" if self.state.gpu_available else "cpu"
            
            # Run the transcription pipeline
            rag_path, siglip_path = dual_transcriber.process_video_pipeline(video_path)
            
            # Map rag_text_embeddings.json to unified transcript_chunks.json
            shutil.copyfile(rag_path, expected_output)
            
            # Verify outputs
            if not self.validate_json_file(expected_output):
                raise ValueError("Transcription failed to produce a valid transcript_chunks.json schema file.")

            self.state.completed_stages.append("transcription")
            self.state.generated_files.append(str(expected_output))
            
            duration = time.time() - start_time
            self.state.stage_durations["transcription"] = round(duration, 2)
            logger.info(f"Transcription complete: transcript_chunks.json generated ({duration:.2f}s)")
            return expected_output
        except Exception as e:
            self.state.errors.append(f"Transcription stage failed: {e}")
            raise e

    def run_vision_processing(self, video_path: str, transcript_path: Path) -> Path:
        """
        Stage 3: Vision Processing
        """
        logger.info("=== Stage 3: Vision Processing ===")
        start_time = time.time()
        
        expected_output = self.temp_dir / "visual_embeddings.json"

        # Resumability check
        if not self.force and self.validate_json_file(expected_output):
            self.state.skipped_stages.append("vision")
            logger.info("Vision processing skipped: visual_embeddings.json is already present and valid.")
            self.state.stage_durations["vision"] = 0.0
            return expected_output

        # Validate input files
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Input video file for vision processing missing: {video_path}")
        if not self.validate_json_file(transcript_path):
            raise FileNotFoundError(f"Valid transcript_chunks.json is missing or corrupted: {transcript_path}")

        try:
            from modules.vision import vision_engine
            
            # Load the transcript chunks
            with open(transcript_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            logger.info("Starting vision processing (sampling, filtering, OCR, ranking)...")
            vision_engine.process_video_blocks(video_path, chunks)
            
            # Verify outputs
            if not self.validate_json_file(expected_output):
                raise ValueError("Vision processing failed to generate a valid visual_embeddings.json output.")

            self.state.completed_stages.append("vision")
            self.state.generated_files.append(str(expected_output))
            
            duration = time.time() - start_time
            self.state.stage_durations["vision"] = round(duration, 2)
            logger.info(f"Vision processing complete: visual_embeddings.json generated ({duration:.2f}s)")
            return expected_output
        except Exception as e:
            self.state.errors.append(f"Vision stage failed: {e}")
            raise e

    def save_pipeline_report(self) -> Path:
        """
        Saves a structured run report to temp_assets/preprocessing_report.json
        """
        report_path = self.temp_dir / "preprocessing_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=4)
            logger.info(f"Structured pipeline report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save pipeline report: {e}")
        return report_path

    def print_summary(self):
        """
        Prints a formatted execution summary table.
        """
        print("\n" + "=" * 60)
        print("VEDEX PREPROCESSING PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"{'Stage':<18} | {'Status':<10} | {'Duration (s)':<12}")
        print("-" * 60)
        
        stages = ["ingestion", "transcription", "vision"]
        for stage in stages:
            status = "SKIPPED" if stage in self.state.skipped_stages else "EXECUTED" if stage in self.state.completed_stages else "FAILED"
            dur = self.state.stage_durations.get(stage, 0.0)
            dur_str = f"{dur:.2f}" if dur > 0.0 else "0.00"
            print(f"{stage.capitalize():<18} | {status:<10} | {dur_str:<12}")
            
        print("-" * 60)
        print(f"Total Duration : {self.state.total_duration:.2f} seconds")
        print(f"GPU Detected   : {'Yes' if self.state.gpu_available else 'No'}")
        print(f"Outputs        : {', '.join([os.path.basename(o) for o in self.state.generated_files])}")
        print("=" * 60 + "\n")

    def execute(self, url: Optional[str] = None, video_path: Optional[str] = None):
        """
        Orchestration loop: Ingest -> Transcribe -> Vision
        """
        pipeline_start = time.time()
        
        try:
            # 1. Ingestion
            video_file = self.run_ingestion(url, video_path)
            
            # 2. Transcription
            transcript_file = self.run_transcription(video_file)
            
            # 3. Vision Alignment
            self.run_vision_processing(video_file, transcript_file)
            
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user via KeyboardInterrupt. Cleaning up...")
            self.state.warnings.append("Pipeline execution interrupted by KeyboardInterrupt.")
            self.clean_temporary_files()
            self.save_pipeline_report()
            sys.exit(130)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.clean_temporary_files()
            self.save_pipeline_report()
            sys.exit(1)
            
        # Success completion path
        self.state.total_duration = time.time() - pipeline_start
        self.clean_temporary_files()
        report_file = self.save_pipeline_report()
        self.state.generated_files.append(str(report_file))
        
        self.print_summary()



def _load_links_file(path: str) -> List[str]:
    """
    Load a list of video URLs from a JSON links file (videos.json).

    Expected format:
        { "videos": [ {"url": "...", "label": "..."}, ... ] }

    Returns a flat list of URL strings.
    """
    p = Path(path)
    if not p.exists():
        print(f"ERROR: Links file not found: {path}")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("videos", [])
    urls = []
    for entry in entries:
        if isinstance(entry, str):
            urls.append(entry)
        elif isinstance(entry, dict) and entry.get("url"):
            urls.append(entry["url"])
    if not urls:
        print(f"ERROR: No valid URLs found in {path}")
        sys.exit(1)
    return urls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vedex Preprocessing Pipeline Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main_preprocessing.py --url  https://youtu.be/xxxx\n"
            "  python main_preprocessing.py --video temp_assets/yt_video.mp4\n"
            "  python main_preprocessing.py --links-file videos.json\n"
            "  python main_preprocessing.py --links-file videos.json --force\n"
        )
    )
    parser.add_argument("--url",        type=str, help="URL of a single YouTube or Google Drive video")
    parser.add_argument("--video",      type=str, help="Path to a local video file (skips ingestion)")
    parser.add_argument("--links-file", type=str, metavar="FILE",
                        help="JSON file containing a list of video URLs (default: videos.json)")
    parser.add_argument("--force",      action="store_true",
                        help="Force re-run all stages, ignoring existing outputs")
    args = parser.parse_args()

    # ── Resolve URLs to process ───────────────────────────────────────────────
    urls_to_process: List[Optional[str]] = []
    local_video: Optional[str] = None

    if args.video:
        # Local file — single run, no URL
        local_video = args.video
        urls_to_process = [None]          # sentinel: no URL needed
    elif args.url:
        urls_to_process = [args.url]
    elif args.links_file:
        urls_to_process = _load_links_file(args.links_file)
    else:
        # Default: try videos.json if it exists
        default_links = Path("videos.json")
        if default_links.exists():
            print(f"No input specified — loading URLs from {default_links}")
            urls_to_process = _load_links_file(str(default_links))
        else:
            parser.print_help()
            print("\nERROR: Specify --url, --video, or --links-file (or create videos.json).")
            sys.exit(1)

    # ── Run pipeline for each URL ─────────────────────────────────────────────
    total = len(urls_to_process)
    for idx, url in enumerate(urls_to_process, start=1):
        if total > 1:
            print(f"\n{'='*60}")
            print(f"Processing video {idx}/{total}: {url or local_video}")
            print(f"{'='*60}")
        orchestrator = PreprocessingOrchestrator(force=args.force)
        orchestrator.execute(url=url, video_path=local_video if url is None else None)

