import os
import re
import uuid
import yt_dlp
import gdown

from config import TEMP_ASSETS_DIR, MAX_VIDEO_DURATION_SECONDS


# Tried in order by yt-dlp; if one client is blocked/rate-limited, it
# falls back to the next. Kept consistent between check_duration() and
# download_youtube() so both stages spoof the same client set.
YOUTUBE_PLAYER_CLIENTS = ["android", "ios", "web_embedded"]


def detect_source(url: str) -> str:
    """Validator: Detects if the URL is YouTube or Google Drive."""
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    elif "drive.google.com" in url:
        return "gdrive"
    else:
        return "unknown"


def save_uploaded_file(file_bytes: bytes, original_filename: str, output_dir: str = TEMP_ASSETS_DIR) -> tuple[str, str]:
    """
    Saves an already-uploaded video file (no download/scraping involved,
    so no YouTube bot-check or ToS exposure here). Generates a video_id
    since uploads have no natural platform-assigned ID.

    output_dir defaults to TEMP_ASSETS_DIR for local/CLI use, but Modal's
    upload endpoint must pass the mounted Volume path ("/uploads") so the
    file is visible to the separate container that processes it.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_id = uuid.uuid4().hex
    ext = os.path.splitext(original_filename)[1] or ".mp4"
    output_path = os.path.join(output_dir, f"{video_id}{ext}")
    with open(output_path, "wb") as f:
        f.write(file_bytes)
    return output_path, video_id


def _extract_gdrive_file_id(url: str) -> str | None:
    """
    Pulls the file ID out of a Google Drive share URL.
    Used both to detect obviously-malformed links early and to build a
    stable, deterministic video_id (so re-ingesting the same file doesn't
    generate a new random ID every time).
    """
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(r"id=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def check_duration(url: str, limit_seconds=MAX_VIDEO_DURATION_SECONDS) -> dict:
    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": YOUTUBE_PLAYER_CLIENTS}},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get("duration", 0)
        if duration > limit_seconds:
            raise ValueError(f"Video too long: {duration}s. Limit is {limit_seconds}s.")
        return info


def download_youtube(url: str) -> tuple[str, str]:
    """YouTube ingestion — best-effort. May fail due to region locks,
    membership restrictions, or platform bot-detection outside our control.
    Google Drive or direct upload are the reliable primary paths."""
    info = check_duration(url)
    video_id = info.get("id") or str(uuid.uuid4())

    output_path = os.path.join(TEMP_ASSETS_DIR, f"{video_id}.mp4")
    ydl_opts = {
        "outtmpl": output_path,
        "format": "best",
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": YOUTUBE_PLAYER_CLIENTS}},
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path, video_id


def download_gdrive(url: str, limit_seconds=MAX_VIDEO_DURATION_SECONDS) -> tuple[str, str]:
    """Handles G-Drive ingestion with safety checks.
    Returns (file_path, video_id)."""
    file_id = _extract_gdrive_file_id(url)
    if file_id is None:
        raise ValueError(
            "Could not extract a file ID from the Google Drive URL. "
            "Check that the link is a valid shareable file link."
        )

    video_id = file_id
    output_path = os.path.join(TEMP_ASSETS_DIR, f"{video_id}.mp4")

    try:
        # fuzzy=True: handles Drive's virus-scan confirmation page for
        # medium/large files. Without it, gdown can silently save that
        # HTML warning page as the output file instead of the real video.
        result = gdown.download(url, output_path, quiet=False, fuzzy=True)
    except Exception as e:
        raise Exception(f"G-Drive Error: {e}")

    if result is None or not os.path.exists(output_path):
        raise Exception(
            "Failed to download from G-Drive. The link may be private "
            "(requires OAuth) or invalid."
        )

    # Sanity check: a valid warning-page save is typically only a few KB;
    # real videos are almost always well over 100KB. Catches the exact
    # failure mode above even if fuzzy=True somehow still slips through.
    if os.path.getsize(output_path) < 100_000:
        os.remove(output_path)
        raise Exception(
            "Downloaded file is suspiciously small — likely Google Drive's "
            "virus-scan warning page rather than the actual video. "
            "Check the file's sharing permissions."
        )

    from pydub.utils import mediainfo
    try:
        duration = float(mediainfo(output_path).get("duration", 0))
        if duration > limit_seconds:
            os.remove(output_path)
            raise ValueError(f"Video too long: {duration:.0f}s. Limit is {limit_seconds}s.")
    except (ValueError, TypeError):
        pass

    return output_path, video_id


def download_video(url: str) -> tuple[str, str]:
    """Main routing logic. Returns (file_path, video_id)."""
    os.makedirs(TEMP_ASSETS_DIR, exist_ok=True)
    source_type = detect_source(url)

    if source_type == "youtube":
        return download_youtube(url)
    elif source_type == "gdrive":
        return download_gdrive(url)
    else:
        raise ValueError("Unsupported source. Please provide a valid YouTube or Google Drive URL.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python modules/ingest.py <URL>")
        sys.exit(1)

    try:
        url = sys.argv[1]
        path, video_id = download_video(url)
        print(f"Ingestion successful. File saved to: {path} | video_id: {video_id}")
    except Exception as e:
        print(f"Pipeline error: {e}")