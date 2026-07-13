<p align="center">
  <img src="frontend/assets/logo.png" alt="VidEx Logo" width="320">
</p>

<h1 align="center">VidEx — Multimodal RAG Video Intelligence</h1>

<p align="center">
  <em>Ask questions about any video. Get answers grounded in what was actually said and shown.</em>
</p>

<p align="center">
  <a href="https://drive.google.com/file/d/17F0gAQnYbacdbQGkqcD_v05M8KUsOFca/view?usp=sharing">
    <img src="https://img.shields.io/badge/🎬_Watch_Demo-Video_Showcase-FF0000?style=for-the-badge&logo=googledrive&logoColor=white" alt="Watch Demo">
  </a>
</p>

<p align="center">
  <a href="#-features"><img src="https://img.shields.io/badge/Features-blue?style=for-the-badge" alt="Features"></a>
  <a href="#-tech-stack"><img src="https://img.shields.io/badge/Tech_Stack-blueviolet?style=for-the-badge" alt="Tech Stack"></a>
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-green?style=for-the-badge" alt="Quick Start"></a>
  <a href="#-api-reference"><img src="https://img.shields.io/badge/API_Docs-orange?style=for-the-badge" alt="API Docs"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Gemini-8E75B2?logo=googlegemini&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/Qdrant-DC382D?logo=qdrant&logoColor=white" alt="Qdrant">
  <img src="https://img.shields.io/badge/Modal-000000?logo=modal&logoColor=white" alt="Modal">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

---

## 📖 Overview

**VidEx** is a multimodal Retrieval-Augmented Generation (RAG) system that transforms educational videos into interactive, queryable knowledge bases. Upload a lecture, tutorial, or any educational video — VidEx transcribes the audio, extracts visually meaningful frames, embeds everything into a unified vector space, and lets you chat with the content using natural language.

> 💡 **Think of it as ChatGPT for your video lectures** — but every answer is grounded in what the professor actually said and showed on screen, complete with clickable timestamp citations.

### How It Works

```
┌──────────────┐     ┌───────────────────────────────────────────────────────┐
│  User Upload │────▶│                  Ingestion Pipeline (Modal GPU)        │
│  or URL      │     │                                                       │
└──────────────┘     │  ┌─────────┐   ┌──────────┐   ┌──────────────────┐  │
                     │  │ Whisper │──▶│ SigLIP   │──▶│ Qdrant Vector DB │  │
                     │  │ (Audio) │   │ (Vision) │   │ (Dual Indexing)  │  │
                     │  └─────────┘   └──────────┘   └──────────────────┘  │
                     └───────────────────────────────────────────────────────┘

┌──────────────┐     ┌───────────────────────────────────────────────────────┐
│  User Query  │────▶│                    Query Pipeline                     │
│  + Frame 📸  │     │                                                       │
└──────────────┘     │  ┌───────────┐   ┌──────────┐   ┌────────────────┐  │
                     │  │ Dual-Mode │──▶│ RRF      │──▶│ Gemini LLM     │  │
                     │  │ Retrieval │   │ Fusion   │   │ (Multimodal)   │  │
                     │  └───────────┘   └──────────┘   └────────────────┘  │
                     └───────────────────────────────────────────────────────┘
```

---

## 📸 Gallery

<div align="center">
  <img src="images/Screenshot%202026-07-13%20143913.png" width="49%">
  <img src="images/Screenshot%202026-07-13%20143938.png" width="49%">
  <img src="images/Screenshot%202026-07-13%20143943.png" width="49%">
  <img src="images/Screenshot%202026-07-13%20143952.png" width="49%">
  <img src="images/Screenshot%202026-07-13%20144011.png" width="49%">
  <img src="images/Screenshot%202026-07-13%20144046.png" width="49%">
  <img src="images/Screenshot%202026-07-13%20144116.png" width="49%">
  <img src="images/Screenshot%202026-07-13%20144204.png" width="49%">
</div>

---

## ✨ Features

### 🎯 Core Capabilities
- **Multimodal RAG** — Searches across both transcript text and visual frames simultaneously using a unified SigLIP embedding space
- **Reciprocal Rank Fusion** — Combines text and visual retrieval scores for more accurate context selection
- **Gemini-Powered Answers** — Generates conversational, professor-style explanations with Gemini 3.1 Flash Lite
- **Clickable Timestamp Citations** — Every answer links back to exact moments in the video
- **Screenshot Analysis** — Upload a frame screenshot and ask "what's happening here?" for visual Q&A

### 📹 Video Ingestion
- **Direct Upload** — Drag & drop any `.mp4`, `.mov`, or `.mkv` file
- **YouTube URLs** — Paste a link (best-effort; subject to platform restrictions)
- **Google Drive Links** — Reliable ingestion from publicly shared files
- **GPU-Accelerated Processing** — Ingestion runs on Modal T4 instances for speed

### 🧠 Intelligent Processing
- **Whisper Transcription** — Sliding-window audio processing with configurable overlap
- **Smart Frame Selection** — Perceptual hashing + blur detection + black frame filtering to extract only meaningful keyframes
- **Rolling Chat Memory** — Automatic summarization keeps conversation context within token limits
- **LRU Response Caching** — Identical queries return instant cached responses
- **Disk-Based Embedding Cache** — Content-hash deduplication prevents redundant model inference

### 🖥️ Dual Frontend
- **Web App (FastAPI + Vanilla JS)** — Modern dark-themed SPA with glassmorphism design, authentication, dashboard, and real-time chat
- **Streamlit App** — Lightweight alternative for quick prototyping and demos

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Gemini 3.1 Flash Lite | Answer generation with multimodal context |
| **Embeddings** | SigLIP (`google/siglip-base-patch16-224`) | Unified 768-dim text + image embeddings |
| **Transcription** | OpenAI Whisper | Audio-to-text with sliding window chunking |
| **Vector DB** | Qdrant Cloud | Dual-index (text + image) vector storage |
| **Backend** | FastAPI + Uvicorn | REST API server with CORS and static file serving |
| **Frontend** | Vanilla JS + Tailwind CSS | Dark-themed SPA with Material Design icons |
| **Compute** | Modal (T4 GPU) | Serverless GPU inference for ingestion |
| **Containerization** | Docker | Production deployment on Hugging Face Spaces |
| **Vision** | OpenCV | Frame extraction, quality filtering, pHash |
| **Data Validation** | Pydantic v2 | Strict schema enforcement across the pipeline |
| **Testing** | Pytest | Unit tests for schemas, chunking, and ingestion |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 – 3.12
- [FFmpeg](https://ffmpeg.org/download.html) installed and on `PATH`
- API keys for [Gemini](https://aistudio.google.com/apikey), [Qdrant Cloud](https://cloud.qdrant.io/), and optionally [Modal](https://modal.com/)

### 1. Clone & Install

```bash
git clone https://github.com/AhmedIsmail7/DEPI_GP.git
cd DEPI_GP
```

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
QDRANT_URL="https://your-cluster.cloud.qdrant.io:6333"
QDRANT_API_KEY="your-qdrant-api-key"
GEMINI_API_KEY="your-gemini-api-key"
```

### 3. Run the Application

**Option A — FastAPI Web App (recommended):**

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

**Option B — Streamlit App:**

```bash
streamlit run app.py
```

### 4. Deploy Ingestion Pipeline (Optional)

The video ingestion pipeline runs on [Modal](https://modal.com/) for GPU-accelerated processing:

```bash
# Set up Modal secrets (one-time)
modal secret create videx-secrets \
  QDRANT_URL="..." \
  QDRANT_API_KEY="..." \
  GEMINI_API_KEY="..."

# Deploy the ingestion endpoints
modal deploy modal_app.py
```

---

## 🐳 Docker

```bash
docker build -t videx .
docker run -p 7860:7860 --env-file .env videx
```

The containerized version exposes port `7860` and is optimized for deployment on Hugging Face Spaces.

---

## 📡 API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/signup` | Create a new account |
| `POST` | `/api/auth/login` | Authenticate and get user info |

### Videos

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/videos` | List all ingested video IDs |
| `DELETE` | `/api/videos/{video_id}` | Remove a video from the index |

### Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ingest/upload` | Upload a video file for processing |
| `POST` | `/api/ingest/url` | Submit a YouTube/GDrive URL for processing |
| `GET` | `/api/ingest/status/{call_id}` | Poll ingestion job progress |

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Query a video with multimodal RAG |

<details>
<summary><b>Example: Chat Request</b></summary>

```bash
curl -X POST http://localhost:8000/api/chat \
  -F "video_id=my_lecture" \
  -F "query=What is the time complexity of binary search?" \
  -F "frame=@screenshot.png"
```

**Response:**
```json
{
  "answer": "The professor explains that binary search has a time complexity of O(log n) because...",
  "sources": [95.0, 120.0, 145.0]
}
```
</details>

---

## 📂 Project Structure

```
DEPI_GP/
├── server.py              # FastAPI application entry point & REST API
├── app.py                 # Streamlit UI (alternative frontend)
├── modal_app.py           # Modal GPU deployment for ingestion pipeline
├── config.py              # Environment variables & global constants
├── schemas.py             # Pydantic data contracts (shared across all modules)
│
├── modules/
│   ├── ingest.py          # Video download (YouTube, GDrive) & upload handling
│   ├── transcribe.py      # Whisper transcription with sliding window overlap
│   ├── vision.py          # Frame extraction, quality filtering & keyframe selection
│   ├── embeddings.py      # SigLIP unified text/image embedding with disk cache
│   ├── database.py        # Qdrant vector DB interface with retry logic
│   ├── retrieval.py       # Dual-mode search + Reciprocal Rank Fusion
│   └── llm_handler.py     # Gemini prompt construction & rolling summarization
│
├── frontend/
│   ├── index.html         # SPA shell with dark glassmorphism theme
│   ├── app.js             # Client-side router & authentication logic
│   ├── pages/
│   │   ├── login.js       # Login page component
│   │   ├── signup.js      # Signup page component
│   │   ├── dashboard.js   # Video management dashboard
│   │   ├── chat.js        # Real-time multimodal chat interface
│   │   └── pricing.js     # Subscription tiers page
│   └── assets/
│       ├── logo.png       # VidEx wordmark
│       └── logo_icon.png  # VidEx favicon
│
├── tests/
│   ├── conftest.py        # Pytest configuration & shared fixtures
│   ├── test_schemas.py    # Data contract validation tests
│   ├── test_chunking.py   # Sliding window boundary tests
│   └── test_ingest.py     # URL sanitization & download handler tests
│
├── Dockerfile             # Production container (Python 3.10 + FFmpeg)
├── requirements.txt       # Pinned dependencies (CPU PyTorch for local dev)
├── pyproject.toml         # Project metadata & dependency groups
├── .env.example           # Template for required environment variables
└── .gitignore             # Standard Python + media exclusions
```

---

## ⚙️ Configuration

All tunable parameters live in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | Whisper model variant (`tiny`, `base`, `small`, `medium`, `large`) |
| `CHUNK_DURATION_MS` | `30000` | Audio chunk window size in milliseconds |
| `OVERLAP_MS` | `5000` | Overlap between adjacent chunks |
| `MAX_VIDEO_DURATION_SECONDS` | `3600` | Maximum allowed video length (1 hour) |
| `FRAME_SAMPLE_COUNT` | `8` | Candidate frames sampled per chunk |
| `BLACK_FRAME_THRESHOLD` | `15` | Brightness cutoff for black frame rejection |
| `BLUR_THRESHOLD` | `100.0` | Laplacian variance cutoff for blur detection |
| `DUPLICATE_FRAME_THRESHOLD` | `0.95` | pHash similarity cutoff for duplicate frames |
| `DEFAULT_TOP_K` | `3` | Number of retrieval results per query |
| `CHAT_HISTORY_TOKEN_LIMIT` | `500` | Token budget for recent chat history |

---

## 🧪 Testing

```bash
pytest tests/ -v
```

Tests cover:
- ✅ Pydantic schema validation (embedding dimensions, time ranges, score bounds)
- ✅ Sliding window chunk boundary computation
- ✅ URL sanitization and domain allowlisting
- ✅ Video download handler routing

---

## 🏗️ Architecture Decisions

| Decision | Rationale |
|---|---|
| **SigLIP over CLIP + MiniLM** | Single model encodes both text and images into the same 768-dim space, eliminating the need for ad-hoc weighted-average fusion between unrelated embedding spaces |
| **Reciprocal Rank Fusion** | More robust than simple score averaging; naturally handles cases where a chunk ranks well on text but poorly on vision (or vice versa) |
| **Modal for ingestion** | GPU-intensive Whisper + SigLIP inference runs on serverless T4 instances, keeping the query-serving backend lightweight and CPU-only |
| **Rolling summarization** | Instead of truncating old messages, overflow chat history is compressed into a rolling summary via Gemini Flash Lite, preserving long-term context |
| **Deterministic point IDs** | UUID5 from `(video_id, chunk_index, timestamp)` ensures re-ingesting a video overwrites old data rather than creating duplicates |
| **Lazy model loading** | Whisper, SigLIP, and Qdrant connections initialize on first use, not on import — keeps startup fast and tests importable |

---

## 👥 Team

This project was built as a graduation project for the **Digital Egypt Pioneers Initiative (DEPI)** program.

---

## 📄 License

This project is developed for educational purposes as part of the DEPI graduation program.
