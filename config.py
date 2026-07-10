# config.py
"""
VideoInsight V2.0 - Shared Configuration & Environment Setup

This module centralizes all configuration parameters, API endpoints,
and environment variables for the entire project.

Usage:
    from config import SETTINGS, INGESTION_CONFIG, QDRANT_CONFIG
    
    output_dir = SETTINGS.output_dir
    max_video_duration = INGESTION_CONFIG.max_duration_seconds
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


# ==================== PROJECT STRUCTURE ====================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, DOWNLOADS_DIR, METADATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)


# ==================== PYDANTIC SETTINGS ====================

class IngestionSettings(BaseSettings):
    """Configuration for VideoIngestion module (Phase 1)"""
    
    model_config = SettingsConfigDict(
        env_prefix="INGESTION_",
        case_sensitive=False,
        extra="allow"
    )
    
    # Directory settings
    output_dir: str = str(DOWNLOADS_DIR)
    metadata_dir: str = str(METADATA_DIR)
    archive_enabled: bool = True
    
    # Video validation
    max_duration_seconds: int = 3600  # 60 minutes
    max_duration_minutes: int = 60
    min_duration_seconds: int = 10  # Minimum video length
    
    # Download settings
    youtube_format: str = "best[ext=mp4]"  # yt-dlp format string
    download_timeout: int = 300  # seconds (5 minutes)
    chunk_size: int = 8192  # bytes for streaming downloads
    
    # URL validation
    max_url_length: int = 2048
    allowed_domains: list = [
        "youtube.com",
        "youtu.be",
        "www.youtube.com",
        "www.youtu.be",
        "drive.google.com",
        "www.drive.google.com"
    ]
    
    # Security
    enable_url_validation: bool = True
    reject_private_gdrive_files: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = str(LOGS_DIR / "ingestion.log")


class TranscriptionSettings(BaseSettings):
    """Configuration for Transcription module (Phase 2)"""
    
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPTION_",
        case_sensitive=False,
        extra="allow"
    )
    
    # Whisper settings
    whisper_model: str = "large-v3"  # tiny, base, small, medium, large, large-v3
    device: str = "cuda"  # or "cpu"
    
    # Text chunking
    chunk_duration_seconds: int = 30
    overlap_seconds: int = 5
    
    # Embeddings
    embedding_model: str = "google/siglip-base-patch16-224"  # SigLIP
    embedding_dimension: int = 768
    
    # Logging
    log_level: str = "INFO"


class VisionSettings(BaseSettings):
    """Configuration for Vision module (Phase 3)"""
    
    model_config = SettingsConfigDict(
        env_prefix="VISION_",
        case_sensitive=False,
        extra="allow"
    )
    
    # Frame extraction
    keyframe_sampling_interval: int = 15  # seconds
    histogram_similarity_threshold: float = 0.95
    
    # CLIP settings (legacy — replaced by SigLIP in vision pipeline)
    clip_model: str = "openai/clip-vit-base-patch32"
    vision_embedding_dimension: int = 768  # SigLIP2 output dimension
    device: str = "cuda"  # or "cpu"
    
    # Logging
    log_level: str = "INFO"


class DatabaseSettings(BaseSettings):
    """Configuration for Qdrant Database (Phase 4)"""
    
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        case_sensitive=False,
        extra="allow"
    )
    
    # Qdrant Cloud settings
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # Collections
    text_collection: str = "text_chunks"
    vision_collection: str = "video_frames"
    
    # Vector dimensions
    text_vector_dim: int = 768   # SigLIP
    vision_vector_dim: int = 768  # SigLIP2 (google/siglip2-base-patch16-224)
    
    # Batch upload
    batch_size: int = 100
    upload_timeout: int = 300
    
    # Logging
    log_level: str = "INFO"
    
    @field_validator("qdrant_api_key")
    @classmethod
    def validate_api_key(cls, v):
        return v


class RetrievalSettings(BaseSettings):
    """Configuration for Retrieval & LLM module (Phase 5)"""
    
    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        case_sensitive=False,
        extra="allow"
    )
    
    # Gemini API settings
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = "gemini-2.0-flash"
    
    # Search settings
    search_limit: int = 5  # Top K results
    similarity_threshold: float = 0.6
    
    # LLM settings
    temperature: float = 0.7
    max_output_tokens: int = 512
    
    # Rate limiting & caching
    rate_limit_calls: int = 60
    rate_limit_period: int = 60  # seconds
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    lru_cache_size: int = 1000
    
    # Retry strategy
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    
    # Logging
    log_level: str = "INFO"
    
    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key(cls, v):
        return v


class StreamlitSettings(BaseSettings):
    """Configuration for Streamlit UI"""
    
    model_config = SettingsConfigDict(
        env_prefix="STREAMLIT_",
        case_sensitive=False,
        extra="allow"
    )
    
    # App settings
    app_title: str = "VideoInsight V2.0"
    app_icon: str = "🎥"
    theme: str = "light"  # light or dark
    
    # Page settings
    page_width: str = "wide"
    
    # Video player
    video_player_height: int = 600
    
    # Logging
    log_level: str = "INFO"


# ==================== INSTANTIATE SETTINGS ====================

INGESTION = IngestionSettings()
TRANSCRIPTION = TranscriptionSettings()
VISION = VisionSettings()
DATABASE = None
RETRIEVAL = None

STREAMLIT_CFG = StreamlitSettings()
def get_database_settings():
    global DATABASE
    if DATABASE is None:
        DATABASE = DatabaseSettings()
    return DATABASE


def get_retrieval_settings():
    global RETRIEVAL
    if RETRIEVAL is None:
        RETRIEVAL = RetrievalSettings()
    return RETRIEVAL

# ==================== GLOBAL SETTINGS ====================

class GlobalSettings(BaseSettings):
    """Global project settings"""
    
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="allow"
    )
    
    # Project info
    project_name: str = "VideoInsight V2.0"
    version: str = "2.0.0"
    
    # Directories
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    downloads_dir: Path = DOWNLOADS_DIR
    metadata_dir: Path = METADATA_DIR
    logs_dir: Path = LOGS_DIR
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Debug mode
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"


SETTINGS = GlobalSettings()


# ==================== CONVENIENCE SHORTCUTS ====================

# Ingestion
INGESTION_CONFIG = {
    "output_dir": INGESTION.output_dir,
    "metadata_dir": INGESTION.metadata_dir,
    "max_duration": INGESTION.max_duration_seconds,
    "timeout": INGESTION.download_timeout,
}

# Database
_db = get_database_settings()

QDRANT_CONFIG = {
    "api_key": _db.qdrant_api_key,
    "url": _db.qdrant_url,
    "text_collection": _db.text_collection,
    "vision_collection": _db.vision_collection,
}


# Retrieval
_ret = get_retrieval_settings()

GEMINI_CONFIG = {
    "api_key": _ret.gemini_api_key,
    "model": _ret.gemini_model,
    "temperature": _ret.temperature,
    "max_tokens": _ret.max_output_tokens,
}

# ==================== UTILITY FUNCTIONS ====================

def get_log_config() -> dict:
    """Get logging configuration dictionary"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": SETTINGS.log_format
            }
        },
        "handlers": {
            "default": {
                "level": SETTINGS.log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard"
            },
            "file": {
                "level": SETTINGS.log_level,
                "class": "logging.FileHandler",
                "filename": INGESTION.log_file,
                "formatter": "standard"
            }
        },
        "loggers": {
            "": {
                "handlers": ["default", "file"] if INGESTION.log_to_file else ["default"],
                "level": SETTINGS.log_level,
                "propagate": True
            }
        }
    }


def validate_configuration() -> bool:
    """
    Validate all configuration settings.
    
    Returns:
        bool: True if all validations pass
        
    Raises:
        ValueError: If any critical configuration is missing
    """
    checks = [
    (INGESTION.output_dir, "Ingestion output directory"),
    (INGESTION.metadata_dir, "Metadata directory"),
]
    for config_value, config_name in checks:
        if not config_value:
            raise ValueError(f"Missing required configuration: {config_name}")
    
    return True


def print_config_summary() -> None:
    """Print a summary of current configuration (for debugging)"""
    print("\n" + "=" * 60)
    print("VideoInsight V2.0 - Configuration Summary")
    print("=" * 60)
    print(f"Project Name: {SETTINGS.project_name}")
    print(f"Version: {SETTINGS.version}")
    print(f"Debug Mode: {SETTINGS.debug}")
    print(f"\nDirectories:")
    print(f"  Project Root: {SETTINGS.project_root}")
    print(f"  Downloads: {INGESTION.output_dir}")
    print(f"  Metadata: {INGESTION.metadata_dir}")
    print(f"\nIngestion:")
    print(f"  Max Duration: {INGESTION.max_duration_minutes} minutes")
    print(f"  YouTube Format: {INGESTION.youtube_format}")
    print(f"\nDatabase:")
    print(f"  Qdrant URL: {DATABASE.qdrant_url}")
    print(f"  Text Collection: {DATABASE.text_collection}")
    print(f"  Vision Collection: {DATABASE.vision_collection}")
    print(f"\nRetrieval:")
    print(f"  Gemini Model: {RETRIEVAL.gemini_model}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    """
    Example usage of configuration module.
    """
    # Print configuration summary
    print_config_summary()
    
    # Access individual settings
    print("Example: Accessing settings")
    print(f"Max video duration: {INGESTION.max_duration_minutes} minutes")
    print(f"Downloads directory: {INGESTION.output_dir}")
    print(f"Qdrant API URL: {DATABASE.qdrant_url}")
    
    # Validate configuration
    try:
        validate_configuration()
        print("[OK] All configuration checks passed!")
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
