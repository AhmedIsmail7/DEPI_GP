"""
Centralized config file. 
Keep all your keys and global settings here instead of hardcoding them all over the place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "video_insight_collection"

# --- LLM ---
# google-genai automatically picks up GEMINI_API_KEY from the env,
# but it's good to declare it here so we can fail early if it's missing.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite"

# --- Models ---
WHISPER_MODEL_SIZE = "base"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"  # unified text+image embeddings
EMBEDDING_DIM = 768  # both modalities share this dimension now

# --- Chunking ---
CHUNK_DURATION_MS = 30_000
OVERLAP_MS = 5_000
MAX_VIDEO_DURATION_SECONDS = 3600  # don't ingest videos longer than an hour

# --- Vision frame selection ---
FRAME_SAMPLE_COUNT = 8              # how many frames we check per chunk
BLACK_FRAME_THRESHOLD = 15          # skip frames that are too dark
BLUR_THRESHOLD = 100.0              # skip blurry frames
DUPLICATE_FRAME_THRESHOLD = 0.95    # skip consecutive frames that look identical

# --- Paths ---
TEMP_ASSETS_DIR = "temp_assets"
CHUNKS_SUBDIR = "chunks"

# --- Database upload reliability ---
UPSERT_MAX_RETRIES = 4
UPSERT_BASE_DELAY_SECONDS = 1.0

# --- Retrieval ---
DEFAULT_TOP_K = 3

# --- Startup validation ---
def validate_env():
    """Run this at startup to crash immediately if you forgot to set your API keys."""
    missing = [
        name for name, val in [
            ("QDRANT_URL", QDRANT_URL),
            ("QDRANT_API_KEY", QDRANT_API_KEY),
            ("GEMINI_API_KEY", GEMINI_API_KEY),
        ] if not val
    ]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")