"""
all the config stuff and environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- db ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "video_insight_collection"

# --- ai ---
# we declare it here to make sure it's set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite"

# --- Models ---
WHISPER_MODEL_SIZE = "base"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"  # model that works for both text and images
EMBEDDING_DIM = 768  # size of vectors

# --- Chunking ---
CHUNK_DURATION_MS = 30_000
OVERLAP_MS = 5_000
MAX_VIDEO_DURATION_SECONDS = 3600  # max video length

# --- Vision Frame Selection Defaults ---
FRAME_SAMPLE_COUNT = 8              # how many frames to check per chunk
BLACK_FRAME_THRESHOLD = 15          # filter out dark frames
BLUR_THRESHOLD = 100.0              # filter out blurry frames
DUPLICATE_FRAME_THRESHOLD = 0.95    # skip frames that look identical

# --- Paths ---
TEMP_ASSETS_DIR = "temp_assets"
CHUNKS_SUBDIR = "chunks"

# --- retry settings ---
UPSERT_MAX_RETRIES = 4
UPSERT_BASE_DELAY_SECONDS = 1.0

# --- Retrieval ---
DEFAULT_TOP_K = 3

# --- LLM Chat Memory ---
CHAT_HISTORY_TOKEN_LIMIT = 500

# --- Startup validation ---
def validate_env():
    """make sure we didn't forget any api keys in .env"""
    missing = [
        name for name, val in [
            ("QDRANT_URL", QDRANT_URL),
            ("QDRANT_API_KEY", QDRANT_API_KEY),
            ("GEMINI_API_KEY", GEMINI_API_KEY),
        ] if not val
    ]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")