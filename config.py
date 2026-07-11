"""
Centralized environment and pipeline configuration.
Every module should import constants from here instead of calling
os.getenv() or hardcoding values independently.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "vedex_knowledge"

# --- LLM (Gemini via google-genai) ---
# google-genai's Client() auto-reads GEMINI_API_KEY from env,
# but we expose it here too for explicit use / early validation.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# --- Models ---
WHISPER_MODEL_SIZE = "base"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"  # unified text+image embedding space
EMBEDDING_DIM = 768  # both text and image embeddings share this dimension now

# --- Chunking ---
CHUNK_DURATION_MS = 30_000
OVERLAP_MS = 5_000
MAX_VIDEO_DURATION_SECONDS = 3600  # 60 min ingestion cap

# --- Vision frame selection ---
FRAME_SAMPLE_COUNT = 8              # candidate frames per chunk (was fixed at 5)
BLACK_FRAME_THRESHOLD = 15          # mean brightness below this = rejected as black/blank
BLUR_THRESHOLD = 100.0              # Laplacian variance below this = rejected as too blurry
DUPLICATE_FRAME_THRESHOLD = 0.95    # pHash similarity above this vs. the previous keyframe = rejected as duplicate

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
    """Call this once at pipeline startup to fail fast on missing config."""
    missing = [
        name for name, val in [
            ("QDRANT_URL", QDRANT_URL),
            ("QDRANT_API_KEY", QDRANT_API_KEY),
            ("GEMINI_API_KEY", GEMINI_API_KEY),
        ] if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )