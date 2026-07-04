import os
import sys

try:
    import gdown
except Exception:  # pragma: no cover - optional dependency
    gdown = None

try:
    import yt_dlp
except Exception:  # pragma: no cover - optional dependency
    yt_dlp = None


def detect_source(url: str) -> str:
    """Validator: Detects if the URL is YouTube or Google Drive."""
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    if "drive.google.com" in url:
        return "gdrive"
    return "unknown"

def check_duration(url: str, limit_seconds: int = 3600):
    """Checks video duration before downloading to prevent storage exhaustion."""
    if yt_dlp is None:
        raise ImportError("yt_dlp is required for video ingestion")

    ydl_opts = {"quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get("duration", 0)
        if duration > limit_seconds:
            raise ValueError(f"Video too long: {duration}s. Limit is {limit_seconds}s.")
        return True

def download_youtube(url: str):
    """Handles YouTube ingestion with specific constraints."""
    check_duration(url) # Validate duration first
    output_path = "temp_assets/yt_video.mp4"
    
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'best',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'javascript': 'node', 
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def download_gdrive(url: str):
    """Handles G-Drive ingestion with safety checks."""
    if gdown is None:
        raise ImportError("gdown is required for Google Drive ingestion")

    output_path = "temp_assets/drive_video.mp4"
    try:
        gdown.download(url, output_path, quiet=False)
        if not os.path.exists(output_path):
            raise Exception("Failed to download from G-Drive. Link might be private.")
        return output_path
    except Exception as e:
        raise Exception(f"G-Drive Error: {e}")

def download_video(url: str):
    """Main routing logic."""
    os.makedirs("temp_assets", exist_ok=True)
    source_type = detect_source(url)

    if source_type == "youtube":
        return download_youtube(url)
    if source_type == "gdrive":
        return download_gdrive(url)
    raise ValueError("Unsupported source. Please provide a valid YouTube or Google Drive URL.")


# For only trying the module directly (not important for the main pipeline & not for import)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modules/ingest.py <URL>")
        sys.exit(1)
    
    try:
        url = sys.argv[1]
        path = download_video(url)
        print(f"Ingestion successful. File saved to: {path}")
    except Exception as e:
        print(f"Pipeline error: {e}")