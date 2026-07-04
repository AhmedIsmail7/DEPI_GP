import argparse

from modules.ingest import download_video
from modules.transcribe import transcriber_engine
from modules.vision import vision_engine

from modules.utils import save_json
from modules.database import export_results_to_json

from config import (
    TRANSCRIPT_OUTPUT,
    VISUAL_OUTPUT,
    VIDEO_URL,
)


def run_pipeline(video_url: str):

    print("=" * 60)
    print("STEP 1 : Video Ingestion")
    print("=" * 60)

    video_path = download_video(video_url)

    print(f"Video Saved At : {video_path}")
    print()

    print("=" * 60)
    print("STEP 2 : Audio Transcription")
    print("=" * 60)

    transcript_chunks = transcriber_engine.process_audio_with_overlap(
        video_path
    )

    save_json(
        transcript_chunks,
        TRANSCRIPT_OUTPUT
    )

    print(f"Transcript Saved : {TRANSCRIPT_OUTPUT}")
    print()

    print("=" * 60)
    print("STEP 3 : Semantic Vision (CLIP)")
    print("=" * 60)

    visual_embeddings = vision_engine.process_video_blocks(
        video_path,
        transcript_chunks
    )

    save_json(
        visual_embeddings,
        VISUAL_OUTPUT
    )

    print(f"Visual Embeddings Saved : {VISUAL_OUTPUT}")
    print()

    export_results_to_json(TRANSCRIPT_OUTPUT, VISUAL_OUTPUT)

    print("=" * 60)
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 60)


def main():

    parser = argparse.ArgumentParser(
        description="Video Processing Pipeline"
    )

    parser.add_argument(
        "--url",
        type=str,
        default=VIDEO_URL,
        help="YouTube or Google Drive URL"
    )

    args = parser.parse_args()

    if not args.url:
        raise ValueError("No video URL provided.")

    try:
        run_pipeline(args.url)

    except Exception as e:
        print(f"\nPipeline Error: {e}")


if __name__ == "__main__":
    main()