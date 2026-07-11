"""
VidEx ingestion pipeline on Modal.
Replaces the entire Kaggle CLI/Secrets/dataset-sync workflow — this
deploys as a real HTTP endpoint with working secret injection, callable
directly from app.py or a production frontend.

Ingestion paths:
  - /upload  : direct file upload (primary, fully reliable)
  - /trigger : URL-based (Google Drive reliable, YouTube best-effort —
               subject to platform bot-detection/region/membership locks)
"""

import modal
from fastapi import UploadFile, File


app = modal.App("videx-ingestion")

# Shared persistent storage between the lightweight upload endpoint and
# the GPU processing function — Modal Functions don't share a local
# filesystem by default, so a Volume is how the uploaded bytes cross
# between them.
uploads_volume = modal.Volume.from_name("videx-uploads", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.0", "openai-whisper==20250625", "transformers==4.44.0",
        "yt-dlp", "gdown==5.2.0",
        "opencv-python-headless==4.10.0.84", "qdrant-client==1.10.1",
        "pillow==10.4.0", "pydub==0.25.1", "numpy>=1.26.0,<2.0.0",
        "fastapi[standard]", "python-multipart",
        "sentencepiece",  # required by SiglipTokenizer
    )
    .add_local_python_source("modules", "config", "schemas")
)


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("videx-secrets")],
    volumes={"/uploads": uploads_volume},
    timeout=900,
)
def run_ingestion_from_path(video_path: str, video_id: str) -> str:
    """Processes an already-saved file — no download step at all."""
    from config import validate_env
    from modules.transcribe import transcriber_engine
    from modules.vision import vision_engine
    from modules.database import db_manager

    validate_env()
    print(f"[Ingestion] Processing uploaded file: {video_id}")

    transcript_chunks = transcriber_engine.process_audio_with_overlap(video_path, video_id)
    if not transcript_chunks:
        raise RuntimeError("Transcription produced zero chunks.")

    visual_chunks = vision_engine.process_video_blocks(video_path, video_id, transcript_chunks)

    db_manager.init_collection()
    db_manager.upsert_data(transcript_chunks, visual_chunks)
    print(f"[Ingestion] Complete: {video_id}")
    return video_id


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("videx-secrets")],
    timeout=900,
)
def run_ingestion_from_url(video_url: str) -> str:
    """Google Drive: fully reliable. YouTube: best-effort — may fail
    due to platform bot-checks or region/membership restrictions.
    Uses player_client spoofing only (see ingest.py), no cookies."""
    from config import validate_env
    from modules.ingest import download_video
    from modules.transcribe import transcriber_engine
    from modules.vision import vision_engine
    from modules.database import db_manager

    validate_env()
    video_path, video_id = download_video(video_url)

    transcript_chunks = transcriber_engine.process_audio_with_overlap(video_path, video_id)
    if not transcript_chunks:
        raise RuntimeError("Transcription produced zero chunks.")

    visual_chunks = vision_engine.process_video_blocks(video_path, video_id, transcript_chunks)

    db_manager.init_collection()
    db_manager.upsert_data(transcript_chunks, visual_chunks)
    return video_id


@app.function(image=image, volumes={"/uploads": uploads_volume})
@modal.fastapi_endpoint(method="POST")
async def upload(file: UploadFile = File(...)):
    """Primary ingestion path: direct file upload, no scraping involved."""
    from modules.ingest import save_uploaded_file

    file_bytes = await file.read()
    video_path, video_id = save_uploaded_file(file_bytes, file.filename, output_dir="/uploads")
    uploads_volume.commit()  # flush the write so other containers can see it

    call = run_ingestion_from_path.spawn(video_path, video_id)
    return {"call_id": call.object_id, "video_id": video_id, "status": "started"}


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def trigger(payload: dict):
    """Secondary path: Google Drive (reliable) or YouTube (best-effort)."""
    call = run_ingestion_from_url.spawn(payload["video_url"])
    return {"call_id": call.object_id, "status": "started"}


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def status(call_id: str):
    function_call = modal.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
        return {"status": "complete", "video_id": result}
    except modal.exception.OutputExpiredError:
        return {"status": "expired"}
    except TimeoutError:
        return {"status": "running"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}