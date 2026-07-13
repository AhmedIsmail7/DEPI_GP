"""
VidEx — FastAPI Application Entry Point.

Mounts the frontend static files and exposes REST API endpoints 
for authentication, video ingestion, and AI chat.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import hashlib
import tempfile

import requests as http_requests  # renamed to avoid clash with fastapi Request
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import TEMP_ASSETS_DIR
from modules.database import db_manager
from modules.retrieval import retriever
from modules.llm_handler import llm_handler

# ---------------------------------------------------------------------------
# Endpoint URLs for the deployed Modal application
# ---------------------------------------------------------------------------
MODAL_UPLOAD_URL = "https://dark0danger--videx-ingestion-upload.modal.run"
MODAL_TRIGGER_URL = "https://dark0danger--videx-ingestion-trigger.modal.run"
MODAL_STATUS_URL = "https://dark0danger--videx-ingestion-status.modal.run"

# ---------------------------------------------------------------------------
# Local JSON authentication storage (development/testing only)
# ---------------------------------------------------------------------------
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def _load_users() -> list[dict]:
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: list[dict]):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VidEx API")

# Enable CORS for the Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve temp_assets (local video files for the player)
os.makedirs(TEMP_ASSETS_DIR, exist_ok=True)
app.mount("/temp_assets", StaticFiles(directory=TEMP_ASSETS_DIR), name="temp_assets")

# Serve frontend static files (JS pages)
app.mount("/pages", StaticFiles(directory=os.path.join("frontend", "pages")), name="pages")


# ======================= AUTH ENDPOINTS =======================

@app.post("/api/auth/signup")
async def signup(request: Request):
    body = await request.json()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")
    name = body.get("name", "")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    users = _load_users()
    if any(u["email"] == email for u in users):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    users.append({
        "email": email,
        "password": _hash_password(password),
        "name": name,
        "role": "Free",
    })
    _save_users(users)
    return {"message": "Account created successfully."}


@app.post("/api/auth/login")
async def login(request: Request):
    body = await request.json()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    users = _load_users()
    user = next((u for u in users if u["email"] == email), None)

    if not user or user["password"] != _hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    return {
        "user": {
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
        }
    }


# ======================= VIDEO LIST =======================

@app.get("/api/videos")
async def list_videos():
    """Fetch a list of available video IDs from the vector database."""
    try:
        db_manager.init_collection()
        video_ids = db_manager.get_available_video_ids()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not reach Qdrant: {e}")

    # Return a list of objects the frontend expects
    return [{"id": vid, "name": vid} for vid in video_ids]

@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """Remove a video from the vector store and delete its local asset."""
    try:
        from qdrant_client import models
        db_manager.client.delete(
            collection_name=db_manager.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id",
                        match=models.MatchValue(value=video_id),
                    )
                ]
            ),
        )
        
        local_path = os.path.join(TEMP_ASSETS_DIR, f"{video_id}.mp4")
        if os.path.exists(local_path):
            os.remove(local_path)
            
        return {"status": "success", "message": f"Deleted video {video_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {e}")


# ======================= INGESTION (proxy to Modal) =======================

@app.post("/api/ingest/url")
async def ingest_url(request: Request):
    """Forward a URL-based ingestion request to the Modal backend."""
    body = await request.json()
    video_url = body.get("video_url", "")
    if not video_url:
        raise HTTPException(status_code=400, detail="video_url is required.")

    try:
        resp = http_requests.post(
            MODAL_TRIGGER_URL,
            json={"video_url": video_url},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Cache Google Drive videos locally in the background
        # to enable native playback in the frontend video player
        from modules.ingest import detect_source, download_gdrive
        if detect_source(video_url) == "gdrive":
            import threading
            threading.Thread(target=download_gdrive, args=(video_url,), daemon=True).start()
            
        return data
    except http_requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Modal trigger failed: {e}")


@app.post("/api/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    """Forward a direct file upload to the Modal ingestion pipeline."""
    try:
        file_bytes = await file.read()
        resp = http_requests.post(
            MODAL_UPLOAD_URL,
            files={"file": (file.filename, file_bytes, file.content_type)},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Persist a local copy for frontend playback
        video_id = data.get("video_id")
        if video_id:
            local_path = os.path.join(TEMP_ASSETS_DIR, f"{video_id}.mp4")
            with open(local_path, "wb") as f:
                f.write(file_bytes)
                
        return data
    except http_requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Modal upload failed: {e}")


@app.get("/api/ingest/status/{call_id}")
async def ingest_status(call_id: str):
    """Check the progress of an active Modal ingestion task."""
    try:
        resp = http_requests.get(
            MODAL_STATUS_URL,
            params={"call_id": call_id},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except http_requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Modal status check failed: {e}")


# ======================= CHAT =======================

@app.post("/api/chat")
async def chat(
    video_id: str = Form(...),
    query: str = Form(...),
    frame: UploadFile | None = File(default=None),
):
    """
    Process a user query against a video's index.
    Retrieves context from the vector store and generates an LLM response.
    """
    if not video_id or not query:
        raise HTTPException(status_code=400, detail="video_id and query are required.")

    # Handle optional frame upload
    frame_path = None
    if frame and frame.filename:
        fd, frame_path = tempfile.mkstemp(suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(await frame.read())

    try:
        # 1. Retrieve matching chunks
        results = retriever.retrieve(query, top_k=3, video_id=video_id)

        if not results:
            return {"answer": "I couldn't find any relevant content for this question in the video.", "sources": []}

        # 2. Generate LLM answer
        answer = llm_handler.generate_response(
            query,
            results,
            video_id=video_id,
            current_frame_path=frame_path,
        )

        sources = sorted(set(answer.source_timestamps)) if answer.source_timestamps else []
        return {"answer": answer.answer, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {e}")
    finally:
        # Clean up temp frame file
        if frame_path and os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except OSError:
                pass


# ======================= FRONTEND SERVING =======================

@app.get("/app.js")
async def serve_app_js():
    return FileResponse(os.path.join("frontend", "app.js"), media_type="application/javascript")

app.mount("/assets", StaticFiles(directory=os.path.join("frontend", "assets")), name="assets")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """SPA fallback routing: serve index.html for all unmatched paths."""
    return FileResponse(os.path.join("frontend", "index.html"), media_type="text/html")
