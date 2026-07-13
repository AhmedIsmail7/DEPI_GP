"""
Application Configuration.
Loads environment variables and defines global constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "video_insight_collection"

# --- LLM ---
# The SDK resolves this automatically, but we declare it here for explicit validation.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite"

# --- Models ---
WHISPER_MODEL_SIZE = "base"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"  # Unified vision-language embedding model
EMBEDDING_DIM = 768  # Embedding dimension for both text and image modalities

# --- Chunking ---
CHUNK_DURATION_MS = 30_000
OVERLAP_MS = 5_000
MAX_VIDEO_DURATION_SECONDS = 3600  # Maximum permitted video duration

# --- Vision Frame Selection Defaults ---
FRAME_SAMPLE_COUNT = 8              # Frames analyzed per chunk
BLACK_FRAME_THRESHOLD = 15          # Threshold for filtering dark/black frames
BLUR_THRESHOLD = 100.0              # Threshold for filtering blurry frames
DUPLICATE_FRAME_THRESHOLD = 0.95    # Similarity threshold to drop duplicate frames

# --- Paths ---
TEMP_ASSETS_DIR = "temp_assets"
CHUNKS_SUBDIR = "chunks"

# --- Database upload reliability ---
UPSERT_MAX_RETRIES = 4
UPSERT_BASE_DELAY_SECONDS = 1.0

# --- Retrieval ---
DEFAULT_TOP_K = 3

# --- LLM Chat Memory ---
CHAT_HISTORY_TOKEN_LIMIT = 500

# --- Startup validation ---
def validate_env():
    """Validates the presence of required environment variables."""
    missing = [
        name for name, val in [
            ("QDRANT_URL", QDRANT_URL),
            ("QDRANT_API_KEY", QDRANT_API_KEY),
            ("GEMINI_API_KEY", GEMINI_API_KEY),
        ] if not val
    ]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")