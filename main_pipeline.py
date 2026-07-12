"""
VidEx main ingestion pipeline.
Orchestrates: URL -> download -> transcribe -> visual align -> Qdrant upload.

Intended to run headlessly on Kaggle's remote GPU via the Kaggle CLI,
but works identically on a local machine with a GPU or CPU fallback.

Video URL source (in priority order):
    1. CLI argument: python main_pipeline.py <video_url>   (local runs)
    2. Kaggle dataset config file: /kaggle/input/<slug>/job_config.json
"""




import sys
import subprocess
import os
import json
import time
import traceback

SOURCE_DIR = "/kaggle/input/videx-source"

req_file = os.path.join(SOURCE_DIR, "requirements.txt")
if os.path.exists(req_file):
    print("[Pipeline] Installing dependencies from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])

sys.path.insert(0, SOURCE_DIR)

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for key in ("QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"):
        try:
            os.environ[key] = secrets.get_secret(key)
            print(f"[Secrets] Loaded {key}")
        except Exception as e:
            print(f"[Secrets] Failed to load {key}: {e}")
except ImportError as e:
    print(f"[Secrets] kaggle_secrets not available: {e}")


from config import validate_env
from modules.ingest import download_video
from modules.transcribe import transcriber_engine
from modules.vision import vision_engine
from modules.database import db_manager

KAGGLE_JOB_CONFIG_PATH = "/kaggle/input/datasets/ahmedismail775/videx-job-config/job_config.json"

def resolve_video_url() -> str:
    """
    Determines the target video URL depending on execution context.
    Local runs pass it as a CLI arg; Kaggle runs read it from the
    mounted dataset config file.

    We can't just check len(sys.argv) >= 2 here — Kaggle's notebook/script
    execution environment injects its own internal arguments into sys.argv
    (e.g. kernel connection paths), so that check is true on Kaggle too,
    just for the wrong reason. Instead, only trust argv[1] if it actually
    looks like a URL.
    """
    if len(sys.argv) >= 2 and sys.argv[1].startswith(("http://", "https://")):
        return sys.argv[1]

    if os.path.exists(KAGGLE_JOB_CONFIG_PATH):
        with open(KAGGLE_JOB_CONFIG_PATH, "r") as f:
            job_config = json.load(f)
        url = job_config.get("video_url")
        if not url:
            raise ValueError(f"'video_url' key missing in {KAGGLE_JOB_CONFIG_PATH}")
        return url

    raise RuntimeError(
        "No video URL provided. Pass it as a CLI argument for local runs, "
        f"or ensure the job config dataset is mounted at {KAGGLE_JOB_CONFIG_PATH} on Kaggle."
    )


def run_pipeline(url: str):
    start_time = time.time()

    print("=" * 60)
    print(f"[Pipeline] Starting ingestion for: {url}")
    print("=" * 60)

    validate_env()

    print("\n[Step 1/4] Downloading video...")
    video_path, video_id = download_video(url)
    print(f"[Step 1/4] Done. video_id={video_id} | path={video_path}")

    print("\n[Step 2/4] Transcribing audio + generating text embeddings...")
    transcript_chunks = transcriber_engine.process_audio_with_overlap(video_path, video_id)
    print(f"[Step 2/4] Done. {len(transcript_chunks)} transcript chunks produced.")

    if not transcript_chunks:
        raise RuntimeError(
            "Transcription produced zero chunks — video may be silent, "
            "corrupted, or too short. Aborting before visual/DB steps."
        )

    print("\n[Step 3/4] Aligning visual frames to transcript chunks...")
    visual_chunks = vision_engine.process_video_blocks(video_path, video_id, transcript_chunks)
    print(f"[Step 3/4] Done. {len(visual_chunks)} visual chunks produced.")

    if len(visual_chunks) < len(transcript_chunks):
        skipped = len(transcript_chunks) - len(visual_chunks)
        print(f"[Step 3/4] Warning: {skipped} transcript chunk(s) had no matching "
              f"frame extracted and will be dropped during database upload.")

    print("\n[Step 4/4] Uploading to Qdrant...")
    db_manager.init_collection()
    db_manager.upsert_data(transcript_chunks, visual_chunks)
    print("[Step 4/4] Done.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[Pipeline] Completed successfully in {elapsed:.1f}s")
    print(f"[Pipeline] video_id: {video_id}")
    print("=" * 60)

    return video_id


if __name__ == "__main__":
    try:
        url = resolve_video_url()
        run_pipeline(url)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[Pipeline] FAILED: {e}")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)