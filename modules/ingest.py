"""
Ingestion module — downloads videos from YouTube or Google Drive, or handles direct uploads.
Includes some basic safety checks so we don't accidentally download massive 10-hour videos
or follow sketchy redirect links.
"""

import os
import re
import uuid
from urllib.parse import urlparse

import requests
import yt_dlp
import gdown

from config import TEMP_ASSETS_DIR, MAX_VIDEO_DURATION_SECONDS


# Try different YouTube clients in case one gets blocked or rate-limited.
YOUTUBE_PLAYER_CLIENTS = ["android", "ios", "web_embedded"]

ALLOWED_DOMAINS = {"youtube.com", "youtu.be", "drive.google.com"}

# Characters that have absolutely no business being in a video URL.
# Rejecting these just to be safe.
DANGEROUS_CHARS = set(";|`$<>\n\r")

MAX_URL_LENGTH = 2000


def _normalize_domain(netloc: str) -> str:
    return netloc.lower().removeprefix("www.")


def sanitize_url(url: str) -> None:
    """
    Blocks obvious junk URLs before we even try to hit the network.
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")

    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"URL is unreasonably long ({len(url)} chars) — rejected.")

    if any(ch in url for ch in DANGEROUS_CHARS):
        raise ValueError("URL contains characters that are never valid in a video link.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must use http:// or https:// — got scheme '{parsed.scheme}'.")

    domain = _normalize_domain(parsed.netloc)
    if not any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS):
        raise ValueError(
            f"Domain '{parsed.netloc}' is not supported. "
            f"Only YouTube and Google Drive links are accepted."
        )


def detect_source(url: str) -> str:
    """Figures out if the URL is from YouTube or Google Drive."""
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    elif "drive.google.com" in url:
        return "gdrive"
    else:
        return "unknown"


def save_uploaded_file(file_bytes: bytes, original_filename: str, output_dir: str = TEMP_ASSETS_DIR) -> tuple[str, str]:
    """
    Saves an uploaded video file to our temp folder and gives it a UUID
    so we can track it.
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
    Pulls the actual file ID out of a Google Drive link.
    Helps us detect bad links early and gives us a stable ID so we don't 
    end up re-ingesting the exact same video multiple times.
    """
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(r"id=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def _is_google_drive_public(url: str) -> bool:
    """
    Checks if a Google Drive link is public before we try to download it.
    If it redirects to a Google login page, we fail early instead of downloading
    the login HTML page and getting confused later.
    """
    try:
        response = requests.get(url, allow_redirects=True, stream=True, timeout=10)
        final_domain = _normalize_domain(urlparse(response.url).netloc)
        response.close()
        return "accounts.google.com" not in final_domain
    except requests.RequestException:
        # If the network just blips, don't block the download attempt.
        # Let gdown figure it out.
        return True


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
    """
    YouTube ingestion — best-effort. Might fail if the video is region locked,
    members-only, or if YouTube decides we look like a bot.
    """
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
    """Downloads from Google Drive and runs some safety checks."""
    file_id = _extract_gdrive_file_id(url)
    if file_id is None:
        raise ValueError(
            "Could not extract a file ID from the Google Drive URL. "
            "Check that the link is a valid shareable file link."
        )

    if not _is_google_drive_public(url):
        raise ValueError(
            "This Google Drive link requires sign-in (it redirects to a "
            "Google login page), meaning it isn't shared publicly. Set "
            "sharing to 'Anyone with the link' and try again."
        )

    video_id = file_id  # deterministic: same file always maps to the same ID
    output_path = os.path.join(TEMP_ASSETS_DIR, f"{video_id}.mp4")

    try:
        # fuzzy=True handles Drive's virus-scan warning page for big files
        # so we don't accidentally save the HTML warning as our video file.
        result = gdown.download(url, output_path, quiet=False, fuzzy=True)
    except Exception as e:
        raise Exception(f"G-Drive Error: {e}")

    if result is None or not os.path.exists(output_path):
        raise Exception(
            "Failed to download from G-Drive. The link may be private "
            "(requires OAuth) or invalid."
        )

    # Sanity check: if the file is tiny, it's probably the Google Drive warning page,
    # not the actual video.
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
    """Main routing logic. Figures out what kind of link it is and downloads it."""
    sanitize_url(url)

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