import yt_dlp
import gdown
import os
import sys

def detect_source(url: str) -> str:
    """Validator: Detects if the URL is YouTube or Google Drive."""
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    elif "drive.google.com" in url:
        return "gdrive"
    else:
        return "unknown"

def check_duration(url: str, limit_seconds: int = 3600) -> bool:
    """
    Checks video duration before downloading to prevent storage & VRAM exhaustion.
    Uses silent extraction to avoid terminal log pollution.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get('duration', 0)
        if duration > limit_seconds:
            raise ValueError(f"[Security Risk] Video duration ({duration}s) exceeds production limit ({limit_seconds}s).")
    return True

def download_youtube(url: str) -> str:
    """Handles YouTube ingestion with strict cleanup & fast AI-optimized formats."""
    check_duration(url) # Validate duration first
    output_path = "temp_assets/yt_video.mp4"
    
    # [P1 Fix]: Explicitly delete existing file to prevent state contamination
    if os.path.exists(output_path):
        os.remove(output_path)
    
    ydl_opts = {
        'outtmpl': output_path,
        # Format 18 = 360p MP4 (Audio+Video combined), best for Whisper & SigLIP 2 speed
        'format': '18/best[ext=mp4]/best',
        'no_warnings': True,
        'nocheckcertificate': True,
        'overwrites': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    }
    
    print(f"--- [YouTube Ingestion] Downloading stream to {output_path} ---")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    if not os.path.exists(output_path):
        raise FileNotFoundError("[Fatal Error] yt-dlp completed but output file is missing.")
        
    return output_path

def download_gdrive(url: str) -> str:
    """Handles Google Drive ingestion with safety checks & pre-cleaning."""
    output_path = "temp_assets/drive_video.mp4"
    
    # [P1 Fix]: Explicitly delete existing file to prevent state contamination
    if os.path.exists(output_path):
        os.remove(output_path)
        
    print(f"--- [G-Drive Ingestion] Downloading file to {output_path} ---")
    try:
        # gdown automatically handles public shared files
        gdown.download(url, output_path, quiet=False)
        if not os.path.exists(output_path):
            raise FileNotFoundError("[Fatal Error] Failed to download from G-Drive. Link might be private or broken.")
        return output_path
    except Exception as e:
        raise RuntimeError(f"[G-Drive Fatal Error]: {str(e)}")

def download_video(url: str) -> str:
    """Main routing & directory initialization logic."""
    os.makedirs("temp_assets", exist_ok=True)
    source_type = detect_source(url)
    
    print(f"--- [Ingest Router] Detected source: '{source_type}' ---")
    if source_type == "youtube":
        return download_youtube(url)
    elif source_type == "gdrive":
        return download_gdrive(url)
    else:
        raise ValueError("[Invalid Input] Unsupported source URL. Please provide a valid YouTube or Google Drive link.")


# For standalone CLI execution or testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modules/ingest.py <URL>")
        sys.exit(1)
    
    try:
        target_url = sys.argv[1]
        saved_path = download_video(target_url)
        print(f"\n[Success] Ingestion pipeline complete. Artifact saved to: {saved_path}")
    except Exception as err:
        print(f"\n[Pipeline Failure]: {str(err)}")
        sys.exit(1)