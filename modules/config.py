import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# Directories
# ==========================================

TEMP_DIR = "temp_assets"
CHUNKS_DIR = os.path.join(TEMP_DIR, "chunks")

os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# Whisper
# ==========================================

WHISPER_MODEL = "base"      # tiny | base | small | medium | large

CHUNK_DURATION_MS = 30_000
OVERLAP_DURATION_MS = 5_000

# ==========================================
# CLIP
# ==========================================

CLIP_MODEL = "openai/clip-vit-base-patch32"

NUM_FRAMES_PER_CHUNK = 5

# ==========================================
# Embedding
# ==========================================

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TEXT_VECTOR_SIZE = 384
IMAGE_VECTOR_SIZE = 512

# ==========================================
# Output Files
# ==========================================

TRANSCRIPT_OUTPUT = os.path.join(
    TEMP_DIR,
    "transcript_chunks.json"
)

VISUAL_OUTPUT = os.path.join(
    TEMP_DIR,
    "visual_embeddings.json"
)

# ==========================================
# Qdrant
# ==========================================

COLLECTION_NAME = "video_knowledge"

TOP_K_RESULTS = 3

# ==========================================
# Cohere
# ==========================================

COHERE_MODEL = "command-a-03-2025"

# ==========================================
# Video Source
# ==========================================

VIDEO_URL = os.getenv(
    "VIDEO_URL",
    "https://youtu.be/PUT_YOUR_VIDEO_HERE"
)

# ==========================================
# Environment Variables
# ==========================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")