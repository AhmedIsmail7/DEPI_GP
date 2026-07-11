# VidEx — Multimodal RAG Educational Video Assistant
 
VidEx lets you ask natural-language questions about a video's content and get
back a cited, timestamped answer. It ingests a video (via direct upload,
Google Drive, or YouTube), transcribes and visually indexes it, stores the
result in a vector database, and answers questions using retrieval-augmented
generation with Gemini.
 
## Architecture
 
```
[ Streamlit UI (app.py) ]
        │
        ├── Ingest ──► [ Modal: GPU functions ] ──► [ Qdrant Cloud ]
        │                (Whisper + CLIP)             (vector storage)
        │
        └── Ask ──────► [ Qdrant Cloud ] ──► [ Gemini API ] ──► Answer
                          (retrieval)          (generation)
```
 
- **Ingestion** (download/transcribe/visually-index/upload) runs on
  [Modal](https://modal.com) — a serverless GPU platform. This is the only
  part of the pipeline that needs a GPU.
- **Querying** (retrieval + answer generation) runs locally in the Streamlit
  app — no GPU needed for this part.
- **Qdrant Cloud** is the shared vector database both sides talk to.
## Prerequisites
 
You'll need free accounts on three services. None require a credit card to
get started.
 
| Service | What it's for | Where to sign up |
|---|---|---|
| **Qdrant Cloud** | Vector database storage | https://cloud.qdrant.io |
| **Google AI Studio** | Gemini API key (answer generation) | https://aistudio.google.com |
| **Modal** | Serverless GPU (ingestion pipeline) | https://modal.com |
 
You'll also need:
- Python 3.11 or 3.12 (**not 3.13** — some ML dependencies here don't yet
  have mature support for it)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `git`
---
 
## 1. Clone the repo
 
```bash
git clone <your-fork-or-repo-url>
cd videx
```
 
## 2. Install dependencies
 
```bash
uv python pin 3.12
uv sync
uv sync --group dev   # optional, only needed to run the test suite
```
 
## 3. Set up your credentials
 
### 3a. Qdrant Cloud
 
1. Create a free cluster at https://cloud.qdrant.io
2. From your cluster's dashboard, copy the **Cluster URL** and create an
   **API Key**.
### 3b. Gemini (Google AI Studio)
 
1. Go to https://aistudio.google.com
2. Click **Get API key** → **Create API key**
3. Copy the key.
### 3c. Create your local `.env` file
 
Create a file named `.env` in the project root:
 
```bash
QDRANT_URL=https://your-cluster-url.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```
 
This file is used by the **local** app (querying) and is already listed in
`.gitignore` — never commit it.
 
## 4. Set up Modal (for ingestion)
 
Modal runs the GPU-heavy ingestion pipeline (download, Whisper
transcription, CLIP visual alignment) and needs its own copy of your Qdrant
credentials, stored as a **Modal Secret** (separate from your local `.env`
— Modal's servers can't read your local files).
 
```bash
uv run modal setup      # opens a browser to link your (free) Modal account
uv run modal secret create videx-secrets QDRANT_URL=<your-qdrant-url> QDRANT_API_KEY=<your-qdrant-key>
```
 
Then deploy the ingestion pipeline:
 
```bash
uv run modal deploy modal_app.py
```
 
This prints several public HTTPS URLs — one each for `upload`, `trigger`,
and `status`. Copy them.
 
### Update `app.py` with your Modal URLs
 
Open `app.py` and replace these three constants near the top with the URLs
Modal just gave you:
 
```python
MODAL_UPLOAD_URL = "https://your-username--videx-ingestion-upload.modal.run"
MODAL_TRIGGER_URL = "https://your-username--videx-ingestion-trigger.modal.run"
MODAL_STATUS_URL = "https://your-username--videx-ingestion-status.modal.run"
```
 
## 5. Run the app
 
```bash
uv run streamlit run app.py
```
 
This opens a local browser tab. Use the **"Add a video"** tab to ingest your
first video (file upload or Google Drive link — see note below on YouTube),
then switch to **"Ask VidEx"** once ingestion completes to start asking
questions.
 
## 6. Run the tests (optional, but recommended)
 
```bash
uv run pytest tests/ -v
```
 
This runs the pure-logic test suite (URL parsing, chunk-timing math, schema
validation, retrieval fusion ranking) — no live GPU, Qdrant, or API calls
needed, so it's a fast way to confirm your local setup is sane before
touching real infrastructure.
 
---
 
## Ingestion paths — reliability notes
 
| Method | Reliability | Notes |
|---|---|---|
| **Direct file upload** | ✅ Fully reliable | Recommended primary path |
| **Google Drive link** | ✅ Fully reliable | Link must be shared as "Anyone with the link" |
| **YouTube URL** | ⚠️ Best-effort | May fail due to YouTube's bot-detection on cloud/datacenter IP ranges, region locks, or membership-restricted videos. This is a platform-level restriction outside this project's control, not a bug. |
 
Video length is capped at 60 minutes by default (`MAX_VIDEO_DURATION_SECONDS`
in `config.py`) to keep ingestion time and Modal compute costs reasonable.
 
## Cost expectations
 
- **Qdrant Cloud** free tier: sufficient for many videos' worth of vectors.
- **Modal**: new accounts get **$30/month in free compute credits**. A short
  video (under ~5 min) costs roughly $0.01–0.02 in GPU time to ingest —
  the free tier comfortably covers hundreds of ingestions per month for
  testing/demo purposes.
- **Gemini API**: Google AI Studio's free tier includes a generous daily
  request quota for `gemini-2.5-flash`, more than enough for development
  and demo use.
## Project structure
 
```
videx/
├── app.py                 # Streamlit UI
├── modal_app.py            # Modal ingestion pipeline (GPU functions + HTTP endpoints)
├── config.py                # Centralized environment/config constants
├── schemas.py                 # Pydantic data contracts shared across modules
├── modules/
│   ├── ingest.py               # URL routing, YouTube/Drive download, file upload handling
│   ├── transcribe.py             # Whisper transcription + text embeddings
│   ├── vision.py                   # CLIP visual frame alignment
│   ├── database.py                  # Qdrant collection management
│   ├── retrieval.py                   # Dual-modality search + fusion ranking
│   └── llm_handler.py                   # Gemini-based answer generation
└── tests/                       # Unit tests for pure logic (no live services needed)
```
 
## Troubleshooting
 
- **`ModuleNotFoundError` when running `app.py` locally** — make sure you
  ran `uv sync` and are invoking commands with `uv run`, not a bare
  `python`/`streamlit` command, so the correct virtual environment is used.
- **`Missing required environment variables` error** — check your `.env`
  file exists in the project root and has all three keys set correctly.
- **Ingestion request never completes / times out** — check
  `uv run modal app logs videx-ingestion` for the actual error from the
  Modal function; this is the most direct way to see what's failing inside
  the GPU pipeline.
- **YouTube ingestion fails** — expected sometimes; see the reliability
  table above. Try a Google Drive link or direct upload instead.
