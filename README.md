# Vedex: Real-time Educational Avatar System

Vedex is a production-grade, AI-driven educational platform designed to provide real-time interaction through an intelligent, RAG-enabled avatar. The system leverages multi-agent orchestration to process video, transcribe audio, and provide context-aware responses.

## 🏗 Architecture Overview

Vedex follows a modular micro-service inspired architecture:

- **Ingestion Engine:** Handles video source detection (YouTube/Drive) and stream preprocessing.
- **Transcription Pipeline:** Audio extraction and segmentation using OpenAI Whisper.
- **Reasoning Engine:** Multi-agent system using LangGraph and Cohere for RAG/Retrieval.
- **Visual Interaction:** React Three Fiber / Three.js Avatar integration for real-time engagement.

## 🚀 Tech Stack

- **Backend:** FastAPI, Python 3.12
- **Orchestration:** LangGraph (Multi-Agent RAG)
- **Database:** SQLAlchemy (PostgreSQL/SQLite), Qdrant (Vector Store)
- **Transcription:** OpenAI Whisper
- **Infrastructure:** Docker, uv (Dependency Management)

## 🛠 System Prerequisites

Ensure your development environment meets these requirements:

- **Python 3.12+**
- **uv** (Package manager)
- **Node.js** (Required for `yt-dlp` JS runtime)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Vedex
   Initialize the environment:
   ```

Bash
chmod +x setup.sh
./setup.sh

📂 Project Structure
Plaintext
Vedex/
├── modules/
│ ├── ingest.py # Data ingestion & Validation
│ ├── transcribe.py # Audio-to-Text & Segmentation
│ ├── retrieval.py # RAG Orchestration
│ └── database.py # Schema definitions
├── temp_assets/ # Processed media storage
├── app.py # FastAPI Entry Point
├── pyproject.toml # Dependency definitions
└── setup.sh # OS-level initialization
💡 Usage
To test the ingestion module:

Bash
python modules/ingest.py "<YOUTUBE_URL>"

🛡 Status

## 🛡 Status

- **Phase 1 (Ingestion):** Completed & Verified.
- **Phase 2 (Transcription & Embedding):** Completed & Verified.
- **Phase 3 (Vision & Embedding):** Completed & Verified.
- **Phase 4 (RAG/Agent):** Planned.

Developed for Graduation Project [2026]
