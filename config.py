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
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# --- Chunking ---
CHUNK_DURATION_MS = 30_000
OVERLAP_MS = 5_000
MAX_VIDEO_DURATION_SECONDS = 3600  # 60 min ingestion cap

# --- Paths ---
TEMP_ASSETS_DIR = "temp_assets"
CHUNKS_SUBDIR = "chunks"

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