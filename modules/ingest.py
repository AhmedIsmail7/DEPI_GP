# modules/ingest.py
"""
VideoInsight V2.0 - Data Ingestion & Source Routing Module
Author: Ramy Safwat
Phase: 1 - Video Ingestion Pipeline

This module handles:
- Secure URL validation and injection prevention
- YouTube and Google Drive video download
- Video duration validation
- Metadata extraction and storage
"""

import re
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from yt_dlp import YoutubeDL
import gdown
import cv2

import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecureVideoIngestion:
    """
    Secure video ingestion system for VideoInsight V2.0
    
    Handles:
    - URL validation and security
    - Multi-source video downloads (YouTube, Google Drive)
    - Video duration validation
    - Metadata extraction and persistence
    
    Attributes:
        output_dir (Path): Directory for temporary video storage
        metadata_dir (Path): Directory for metadata files
        enable_storage_archive (bool): Archive videos after processing
    """
    
    def __init__(self, 
                 output_dir: str = "downloads",
                 metadata_dir: str = "metadata",
                 enable_storage_archive: bool = True) -> None:
        """
        Initialize the VideoIngestion system.
        
        Args:
            output_dir: Where to store downloaded videos
            metadata_dir: Where to store metadata files
            enable_storage_archive: Whether to archive videos after processing
        """
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.enable_storage_archive = enable_storage_archive
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized SecureVideoIngestion: downloads={self.output_dir}, metadata={self.metadata_dir}")
    
    # ==================== SECURITY FUNCTIONS ====================
    
    def sanitize_url(self, url: str) -> dict:
        """
        Sanitize and validate URL against injection attacks.
        
        Performs:
        1. Type and length validation
        2. Character validation (reject dangerous chars)
        3. Injection pattern detection
        4. URL parsing and domain whitelisting
        5. HTTP reachability check
        
        Args:
            url: The URL to validate
            
        Returns:
            dict: {
                "url": validated_url,
                "domain": domain_name,
                "status": "valid",
                "protocol": "https"
            }
            
        Raises:
            ValueError: If URL fails any validation check
        """
        # Step 1: Type check (must be string)
        if not isinstance(url, str):
            logger.error("URL must be a string")
            raise ValueError("URL must be a string")
            
        # Step 2: Length check (max 2048 chars)
        if len(url) > 2048:
            logger.error(f"URL exceeds maximum length of 2048 characters: {len(url)}")
            raise ValueError("URL exceeds maximum length of 2048 characters")
            
        # Step 3: Character validation (reject ;, &, |, `, $, (, ), <>, \n, \r)
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')', '<', '>', '\n', '\r']
        for char in dangerous_chars:
            if char in url:
                logger.error(f"URL contains dangerous character '{char}': {url}")
                raise ValueError(f"URL contains dangerous character '{char}'")
                
        # Step 4: Injection pattern detection
        # SQL Injection character patterns
        sql_char_patterns = ['"', "'", '--', '#']
        for pattern in sql_char_patterns:
            if pattern in url:
                logger.error(f"URL contains SQL injection pattern '{pattern}': {url}")
                raise ValueError(f"URL contains SQL injection pattern '{pattern}'")
                
        # SQL Injection keyword patterns (SELECT, DROP, UNION, INSERT, DELETE)
        sql_keywords = [r'\bselect\b', r'\bdrop\b', r'\bunion\b', r'\binsert\b', r'\bdelete\b']
        for keyword in sql_keywords:
            if re.search(keyword, url, re.IGNORECASE):
                logger.error(f"URL contains SQL injection keyword '{keyword}': {url}")
                raise ValueError("URL contains SQL injection keyword")
                
        # Template injection check
        if '${' in url:
            logger.error(f"URL contains template injection pattern '${{': {url}")
            raise ValueError("URL contains template injection pattern '${'")
            
        # Code injection check (eval, exec, system)
        code_patterns = [r'\beval\b', r'\bexec\b', r'\bsystem\b']
        for pattern in code_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                logger.error(f"URL contains code injection pattern '{pattern}': {url}")
                raise ValueError("URL contains code injection pattern")
                
        # Step 5: URL parsing using urlparse
        try:
            parsed_url = urlparse(url)
        except Exception as e:
            logger.error(f"Failed to parse URL: {e}")
            raise ValueError(f"Failed to parse URL: {e}")
            
        # Step 6: Domain whitelisting (only youtube.com, youtu.be, drive.google.com)
        netloc = parsed_url.netloc.lower()
        if ':' in netloc:
            netloc = netloc.split(':')[0]
            
        allowed_domains = ["youtube.com", "youtu.be", "drive.google.com"]
        is_allowed = False
        for domain in allowed_domains:
            if netloc == domain or netloc.endswith("." + domain):
                is_allowed = True
                break
                
        if not is_allowed:
            logger.error(f"Domain '{netloc}' is not whitelisted: {url}")
            raise ValueError(f"Domain '{netloc}' is not whitelisted")
            
        # Step 7: Protocol check (https preferred, http allowed)
        scheme = parsed_url.scheme.lower()
        if scheme not in ['http', 'https']:
            logger.error(f"Invalid URL scheme '{scheme}': {url}")
            raise ValueError(f"Invalid URL scheme '{scheme}'")
            
        # Step 8: HTTP HEAD request to verify reachability
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            # Some platforms return 405/403 for HEAD, so if HEAD fails we verify we got any response or try a quick stream GET
            response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
            if response.status_code >= 500:
                logger.error(f"URL is unreachable, server returned status {response.status_code}: {url}")
                raise ValueError(f"URL is unreachable, status {response.status_code}")
        except requests.Timeout:
            logger.error(f"Connection timeout verifying reachability of URL: {url}")
            raise ValueError("Connection timeout verifying URL reachability")
        except requests.RequestException as e:
            logger.error(f"Network error verifying reachability of URL: {url}, error: {e}")
            raise ValueError(f"Network error verifying URL reachability: {e}")
            
        # Step 9: Return validation result dict
        return {
            "url": url,
            "domain": netloc,
            "status": "valid",
            "protocol": scheme
        }
    
    def detect_source(self, url: str) -> str:
        """
        Detect the source type of the video URL.
        
        Args:
            url: The URL to analyze
            
        Returns:
            str: Either "youtube" or "google_drive"
            
        Raises:
            ValueError: If URL is invalid or source is unknown
        """
        # Step 1: Call sanitize_url first (security check)
        validation = self.sanitize_url(url)
        domain = validation["domain"]
        
        # Step 2-4: Check for YouTube / Google Drive indicators
        if "youtube.com" in domain or "youtu.be" in domain:
            return "youtube"
        elif "drive.google.com" in domain:
            return "google_drive"
        else:
            raise ValueError(f"Unknown source: {domain}")
    
    # ==================== DOWNLOAD FUNCTIONS ====================
    
    def download_youtube(self, url: str) -> dict:
        """
        Download video from YouTube using yt-dlp.
        
        Downloads the best available MP4 format and extracts metadata.
        
        Args:
            url: YouTube URL
            
        Returns:
            dict: {
                "success": True,
                "title": video_title,
                "path": local_file_path,
                "duration": duration_seconds,
                "uploader": channel_name,
                "video_id": youtube_video_id,
                "source": "youtube",
                "file_size_mb": file_size
            }
            
        Raises:
            Exception: If download fails
        """
        logger.info(f"Starting YouTube download: {url}")
        
        # Step 1: Configure yt-dlp options (format, output template)
        outtmpl_path = str(self.output_dir / '%(id)s.%(ext)s')
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': outtmpl_path,
            'quiet': False,
            'no_warnings': False,
        }
        
        # Step 2: Extract video info and download
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Step 4: Extract metadata (title, duration, uploader, etc.)
                video_id = info.get('id', 'unknown_id')
                ext = info.get('ext', 'mp4')
                local_path = info.get('_filename')
                
                if not local_path or not os.path.exists(local_path):
                    local_path = str(self.output_dir / f"{video_id}.{ext}")
                    
                local_path = str(Path(local_path).resolve())
                title = info.get('title', 'Unknown Title')
                duration = float(info.get('duration', 0.0))
                uploader = info.get('uploader', 'Unknown Uploader')
                
                file_size_bytes = os.path.getsize(local_path)
                file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
                
                # Step 5: Log progress and completion
                logger.info(f"Successfully downloaded YouTube video: {title} to {local_path}")
                
                # Step 6: Return result dict
                return {
                    "success": True,
                    "title": title,
                    "path": local_path,
                    "duration": duration,
                    "uploader": uploader,
                    "video_id": video_id,
                    "source": "youtube",
                    "file_size_mb": file_size_mb
                }
        except Exception as e:
            # Step 3: Handle exceptions
            logger.error(f"Failed to download YouTube video {url}: {e}")
            raise Exception(f"YouTube download failed: {e}")
            
    def download_google_drive(self, file_id: str) -> dict:
        """
        Download file from Google Drive.
        
        Uses gdown to download public shared files.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            dict: {
                "success": True,
                "file_id": file_id,
                "path": local_file_path,
                "source": "google_drive",
                "file_size_mb": file_size
            }
            
        Raises:
            ValueError: If file is private/requires authentication
            Exception: If download fails
        """
        logger.info(f"Starting Google Drive download for file ID: {file_id}")
        
        # Step 1: Check if file is public using _is_google_drive_public()
        if not self._is_google_drive_public(file_id):
            logger.error(f"Google Drive file {file_id} is private or requires authentication")
            raise ValueError("Google Drive file is private or requires authentication")
            
        # Step 2: Construct download URL
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            # Step 3: Use gdown.download() to fetch file
            dest_dir = str(self.output_dir)
            downloaded_file = gdown.download(url, output=dest_dir, quiet=False, fuzzy=True)
            
            if not downloaded_file or not os.path.exists(downloaded_file):
                # Fallback to direct output path if gdown returned None
                fixed_path = str(self.output_dir / f"{file_id}.mp4")
                downloaded_file = gdown.download(url, output=fixed_path, quiet=False, fuzzy=True)
                
            if not downloaded_file or not os.path.exists(downloaded_file):
                raise Exception("Download failed, output file not created")
                
            local_path = str(Path(downloaded_file).resolve())
            file_size_bytes = os.path.getsize(local_path)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            
            # Step 4: Log progress and completion
            logger.info(f"Successfully downloaded Google Drive file to {local_path} ({file_size_mb} MB)")
            
            # Step 5: Return result dict
            return {
                "success": True,
                "file_id": file_id,
                "path": local_path,
                "source": "google_drive",
                "file_size_mb": file_size_mb
            }
        except Exception as e:
            logger.error(f"Failed to download Google Drive file {file_id}: {e}")
            raise Exception(f"Google Drive download failed: {e}")
            
    def _is_google_drive_public(self, file_id: str) -> bool:
        """
        Verify that a Google Drive file is publicly accessible.
        
        Sends a HEAD request to detect if file requires authentication.
        Redirects to accounts.google.com indicate private files.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            bool: True if public, False if private or error
        """
        # Step 1: Construct Google Drive download URL
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            # Step 2: Send HEAD request with timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
            
            # Step 3: Check if response URL or history redirects to 'accounts.google.com'
            if "accounts.google.com" in response.url:
                return False
                
            for resp in response.history:
                if "accounts.google.com" in resp.url:
                    return False
                    
            if response.status_code == 403 or response.status_code == 400:
                # Double check with a stream GET request in case HEAD failed with 403/400
                get_resp = requests.get(url, headers=headers, timeout=5, stream=True)
                if "accounts.google.com" in get_resp.url:
                    return False
                if get_resp.status_code == 403:
                    return False
                    
            # Step 4: Return True if public, False if private/error
            return True
        except Exception as e:
            logger.error(f"Error checking if Google Drive file is public: {e}")
            return False
            
    def _extract_gdrive_file_id(self, url: str) -> str:
        """
        Extract file ID from Google Drive URL.
        
        Handles multiple URL formats:
        - https://drive.google.com/file/d/{FILE_ID}/view
        - https://drive.google.com/open?id={FILE_ID}
        - https://drive.google.com/uc?id={FILE_ID}
        
        Args:
            url: Google Drive URL
            
        Returns:
            str: Extracted file ID
            
        Raises:
            ValueError: If no valid file ID found
        """
        # Step 1: Define regex patterns for different URL formats
        patterns = [
            r'drive.google.com/file/d/([a-zA-Z0-9-_]+)',
            r'drive.google.com/open\?id=([a-zA-Z0-9-_]+)',
            r'drive.google.com/uc\?id=([a-zA-Z0-9-_]+)'
        ]
        
        # Step 2 & 3: Attempt to match and return first match
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                file_id = match.group(1)
                if 20 <= len(file_id) <= 44:
                    return file_id
                    
        # Step 4: Raise error if no match found
        logger.error(f"Could not extract Google Drive file ID from URL: {url}")
        raise ValueError("Could not extract Google Drive file ID from URL")
        
    # ==================== VALIDATION FUNCTIONS ====================
    
    def validate_video_duration(self, 
                                video_path: str, 
                                max_duration: int = 3600) -> bool:
        """
        Validate video duration does not exceed maximum.
        
        Uses OpenCV to read video metadata and calculate duration.
        Default max is 3600 seconds (60 minutes).
        
        Args:
            video_path: Path to video file
            max_duration: Maximum allowed duration in seconds (default 3600)
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If video exceeds max duration or is corrupted
        """
        # Step 1: Use cv2.VideoCapture to read video
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            raise ValueError(f"Video file does not exist: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Step 7: Handle corrupted/unreadable videos gracefully
            logger.error(f"Could not open video file: {video_path}")
            raise ValueError("Corrupted or unreadable video file")
            
        try:
            # Step 2: Extract frame_count and fps from video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps <= 0 or frame_count <= 0:
                logger.error(f"Invalid video metadata: fps={fps}, frames={frame_count}")
                raise ValueError("Corrupted or unreadable video file")
                
            # Step 3: Calculate duration: frame_count / fps
            duration = frame_count / fps
            logger.info(f"Validated duration: {duration:.2f} seconds (max allowed: {max_duration} seconds)")
            
            # Step 4 & 6: Compare against max_duration and raise ValueError if exceeds
            if duration > max_duration:
                duration_min = round(duration / 60, 1)
                max_min = max_duration // 60
                msg = f"Video too long: {duration_min} min (max {max_min} min)"
                logger.error(msg)
                raise ValueError(msg)
                
            # Step 5: Release video capture object
            return True
        finally:
            cap.release()
            
    # ==================== METADATA FUNCTIONS ====================
    
    def extract_and_save_metadata(self,
                                   video_path: str,
                                   source_url: str,
                                   source_type: str) -> str:
        """
        Extract video metadata and save as JSON and Markdown files.
        
        Extracts properties from video file and creates permanent records.
        Saves two formats for different use cases:
        - JSON: Machine-readable, structured format
        - Markdown: Human-readable, printable format
        
        Args:
            video_path: Path to video file
            source_url: Original source URL
            source_type: Either "youtube" or "google_drive"
            
        Returns:
            str: Path to saved JSON metadata file
        """
        # Step 1: Use OpenCV to extract video properties
        if not os.path.exists(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Corrupted or unreadable video file")
            
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps > 0:
                duration = frame_count / fps
            else:
                duration = 0.0
                
            resolution = f"{width}x{height}"
            
            # Step 2: Calculate additional properties (file size, duration formatting)
            file_size_bytes = os.path.getsize(video_path)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            
            duration_min = int(duration // 60)
            duration_sec = int(duration % 60)
            duration_formatted = f"{duration_min:02d}:{duration_sec:02d}"
            
            video_id = Path(video_path).stem
            
            # Step 3: Build metadata dictionary with proper structure
            metadata = {
                "video_info": {
                    "video_id": video_id,
                    "source_url": source_url,
                    "source_type": source_type,
                    "local_path": str(Path(video_path).resolve()),
                    "duration_seconds": round(duration, 2),
                    "duration_formatted": duration_formatted,
                    "resolution": resolution,
                    "fps": round(fps, 2),
                    "frame_count": frame_count,
                    "file_size_mb": file_size_mb
                },
                "ingestion_info": {
                    "ingested_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "status": "ready_for_processing"
                }
            }
            
            # Step 4: Save JSON file
            json_path = self.metadata_dir / f"{video_id}_metadata.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
                
            # Step 5: Save Markdown file using _save_metadata_markdown()
            md_path = self.metadata_dir / f"{video_id}_metadata.md"
            self._save_metadata_markdown(md_path, metadata)
            
            # Step 6: Return path to JSON file
            return str(json_path.resolve())
        finally:
            cap.release()
            
    def _save_metadata_markdown(self, md_path: Path, metadata: dict) -> None:
        """
        Save metadata in human-readable Markdown format.
        
        Creates formatted Markdown file with video information.
        Useful for quick reference and documentation.
        
        Args:
            md_path: Output path for Markdown file
            metadata: Metadata dictionary to save
        """
        # Step 1-4: Extract, format as Markdown and write to file
        info = metadata["video_info"]
        ingest = metadata["ingestion_info"]
        
        md_content = f"""# Video Metadata - {info['video_id']}

## 📹 Video Information
- **Source URL:** {info['source_url']}
- **Source Type:** {info['source_type']}
- **Local Path:** {info['local_path']}
- **Duration:** {info['duration_formatted']} ({info['duration_seconds']} seconds)
- **Resolution:** {info['resolution']}
- **FPS:** {info['fps']}
- **Frame Count:** {info['frame_count']}
- **File Size:** {info['file_size_mb']} MB

## 🗂️ Processing Status
- **Status:** {ingest['status']}
- **Ingested At:** {ingest['ingested_at']}
"""
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
    # ==================== MAIN PIPELINE ====================
    
    def ingest_video(self, url: str) -> dict:
        """
        Main ingestion pipeline: Validate → Download → Validate → Extract Metadata.
        
        Orchestrates complete workflow from URL input to processed video output.
        
        Args:
            url: Video source URL
            
        Returns:
            dict (Success): {
                "success": True,
                "video_path": "/path/to/video.mp4",
                "metadata_path": "/path/to/metadata.json",
                "source_type": "youtube" or "google_drive",
                "duration": duration_seconds,
                "video_id": extracted_video_id
            }
            
            dict (Failure): {
                "success": False,
                "error": "Error description",
                "step_failed": "Which step failed" (optional)
            }
        """
        # Step 1: Log pipeline start
        logger.info(f"Starting Ingestion Pipeline for URL: {url}")
        
        try:
            # Step 2: Validate URL
            sanitized = self.sanitize_url(url)
            
            # Step 3: Detect source type
            source_type = self.detect_source(url)
            
            # Step 4: Download video (branch based on source)
            if source_type == "youtube":
                download_res = self.download_youtube(url)
                video_path = download_res["path"]
                video_id = download_res["video_id"]
            elif source_type == "google_drive":
                file_id = self._extract_gdrive_file_id(url)
                download_res = self.download_google_drive(file_id)
                video_path = download_res["path"]
                video_id = f"gdrive_{file_id}"
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
                
            # Step 5: Validate video duration
            try:
                from config import INGESTION
                max_duration = INGESTION.max_duration_seconds
            except ImportError:
                max_duration = 3600
                
            self.validate_video_duration(video_path, max_duration=max_duration)
            
            # Step 6: Extract and save metadata
            metadata_path = self.extract_and_save_metadata(video_path, url, source_type)
            
            # Get final duration
            cap = cv2.VideoCapture(video_path)
            duration_seconds = 0.0
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0:
                    duration_seconds = frames / fps
                cap.release()
                
            # Step 7: Return success result
            return {
                "success": True,
                "video_path": video_path,
                "metadata_path": metadata_path,
                "source_type": source_type,
                "duration": duration_seconds,
                "video_id": video_id
            }
        except Exception as e:
            # Use try-except for graceful error handling
            error_msg = str(e)
            logger.error(f"Ingestion pipeline failed for URL {url}: {error_msg}")
            
            step_failed = "Unknown step"
            if "sanitize" in error_msg or "character" in error_msg or "injection" in error_msg or "whitelist" in error_msg:
                step_failed = "URL validation"
            elif "source" in error_msg:
                step_failed = "Source detection"
            elif "download" in error_msg:
                step_failed = "Download"
            elif "too long" in error_msg or "duration" in error_msg:
                step_failed = "Duration validation"
            elif "metadata" in error_msg:
                step_failed = "Metadata extraction"
                
            return {
                "success": False,
                "error": error_msg,
                "step_failed": step_failed
            }


if __name__ == "__main__":
    """
    Example usage of SecureVideoIngestion class.
    """
    # Initialize ingestion system
    ingestion = SecureVideoIngestion(
        output_dir="downloads",
        metadata_dir="metadata",
        enable_storage_archive=True
    )
    
    # Run a simple test check
    print("SecureVideoIngestion module loaded successfully.")
    result = ingestion.ingest_video(sys.argv[1])
    print(result)