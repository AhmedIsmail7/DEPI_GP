"""
VidEx — Streamlit UI.

Two tabs:
  1. Ask VidEx — query an already-ingested video via dual-modality
     retrieval + Gemini.
  2. Add a video — upload a file directly (primary, reliable path) or
     provide a Google Drive / YouTube URL (Drive reliable, YouTube
     best-effort). Both paths run on Modal and poll for completion.
"""

import time
import requests
import streamlit as st

from modules.database import db_manager
from modules.retrieval import retriever
from modules.llm_handler import llm_handler

MODAL_UPLOAD_URL = "https://ahmedismail7--videx-ingestion-upload.modal.run"
MODAL_TRIGGER_URL = "https://ahmedismail7--videx-ingestion-trigger.modal.run"
MODAL_STATUS_URL = "https://ahmedismail7--videx-ingestion-status.modal.run"

POLL_INTERVAL_SECONDS = 4
POLL_TIMEOUT_SECONDS = 900  # 15 min ceiling, matches Modal function timeout

st.set_page_config(page_title="VidEx", page_icon="🎓", layout="centered")
st.title("🎓 VidEx")
st.caption("Multimodal RAG assistant for educational video content")


def poll_ingestion(call_id: str, status_placeholder) -> dict:
    """
    Polls the Modal /status endpoint until the job completes, errors,
    expires, or times out. Updates a Streamlit placeholder with elapsed
    time so the user sees progress rather than a frozen spinner.
    """
    start = time.time()
    while time.time() - start < POLL_TIMEOUT_SECONDS:
        elapsed = int(time.time() - start)
        status_placeholder.info(f"Processing... ({elapsed}s elapsed)")

        try:
            resp = requests.get(MODAL_STATUS_URL, params={"call_id": call_id}, timeout=15)
            data = resp.json()
        except requests.RequestException as e:
            status_placeholder.warning(f"Network hiccup while checking status: {e}. Retrying...")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if data.get("status") == "complete":
            return data
        elif data.get("status") == "error":
            return data
        elif data.get("status") == "expired":
            return data
        # "running" — keep polling

        time.sleep(POLL_INTERVAL_SECONDS)

    return {"status": "timeout"}


tab_ask, tab_ingest = st.tabs(["Ask VidEx", "Add a video"])

# ---------------------------------------------------------------------------
# Tab 1: Ask VidEx
# ---------------------------------------------------------------------------
with tab_ask:
    st.subheader("Chat with your Video")

    try:
        video_ids = db_manager.get_available_video_ids()
    except Exception as e:
        st.error(f"Could not reach Qdrant: {e}")
        video_ids = []

    if not video_ids:
        st.info("No videos ingested yet. Use the 'Add a video' tab first.")
    else:
        selected_video_id = st.selectbox("Select Video", video_ids)
        
        # Display Video Player
        import os
        from config import TEMP_ASSETS_DIR
        local_video_path = os.path.join(TEMP_ASSETS_DIR, f"{selected_video_id}.mp4")
        
        if len(selected_video_id) == 11 and not " " in selected_video_id:
            # Most likely a YouTube video ID
            st.video(f"https://www.youtube.com/watch?v={selected_video_id}")
        elif os.path.exists(local_video_path):
            st.video(local_video_path)
        else:
            st.info("Video player unavailable: The original file is not stored locally.")

        st.divider()

        # Initialize session state for this video
        session_key = f"messages_{selected_video_id}"
        if session_key not in st.session_state:
            st.session_state[session_key] = [
                {"role": "assistant", "content": "Hello! I am the professor. What questions do you have about this video?"}
            ]

        # Display chat history
        for msg in st.session_state[session_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Optional screenshot uploader
        with st.expander("📸 Provide a screenshot of the current frame (optional)"):
            st.caption("Stuck on an equation? Take a screenshot of the video and drop it here so I can see it!")
            frame_upload = st.file_uploader("Upload Frame", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        # Chat input
        if query := st.chat_input("Ask a question about the video..."):
            # Render user message instantly
            st.session_state[session_key].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Handle uploaded frame
            frame_path = None
            if frame_upload:
                import tempfile
                fd, frame_path = tempfile.mkstemp(suffix=".png")
                with os.fdopen(fd, 'wb') as f:
                    f.write(frame_upload.getvalue())
                st.session_state[session_key].append({"role": "user", "content": "*(User provided a screenshot)*"})

            with st.chat_message("assistant"):
                with st.spinner("Searching video content..."):
                    results = retriever.retrieve(query, top_k=3, video_id=selected_video_id)

                if not results:
                    msg = "I couldn't find any relevant content for this question in the video."
                    st.warning(msg)
                    st.session_state[session_key].append({"role": "assistant", "content": msg})
                else:
                    with st.spinner("Generating answer..."):
                        answer = llm_handler.generate_response(
                            query, 
                            results, 
                            video_id=selected_video_id, 
                            current_frame_path=frame_path
                        )

                    # Build Markdown response with sources
                    full_response = answer.answer + "\n\n**Sources:**\n"
                    for ts in sorted(set(answer.source_timestamps)):
                        minutes, seconds = divmod(int(ts), 60)
                        if len(selected_video_id) == 11:
                            link = f"https://www.youtube.com/watch?v={selected_video_id}&t={int(ts)}s"
                            full_response += f"- [{minutes}:{seconds:02d}]({link})\n"
                        else:
                            full_response += f"- Timestamp {minutes}:{seconds:02d}\n"

                    st.markdown(full_response)
                    st.session_state[session_key].append({"role": "assistant", "content": full_response})

            # Clean up the temporary file
            if frame_path and os.path.exists(frame_path):
                os.remove(frame_path)

# ---------------------------------------------------------------------------
# Tab 2: Add a video
# ---------------------------------------------------------------------------
with tab_ingest:
    st.subheader("Add a video")

    upload_subtab, url_subtab = st.tabs(["Upload File (recommended)", "Video URL"])

    with upload_subtab:
        st.caption("Most reliable — works regardless of platform restrictions.")
        uploaded_file = st.file_uploader("Video file", type=["mp4", "mov", "mkv"])

        if st.button("Ingest uploaded file", disabled=not uploaded_file, key="upload_btn"):
            status_placeholder = st.empty()
            try:
                status_placeholder.info("Uploading file...")
                resp = requests.post(
                    MODAL_UPLOAD_URL,
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    timeout=120,
                )
                data = resp.json()
            except requests.RequestException as e:
                status_placeholder.error(f"Upload failed: {e}")
                st.stop()

            call_id = data.get("call_id")
            video_id = data.get("video_id")
            if not call_id:
                status_placeholder.error(f"Unexpected response: {data}")
                st.stop()

            result = poll_ingestion(call_id, status_placeholder)

            if result.get("status") == "complete":
                status_placeholder.success(f"Done! video_id: {result.get('video_id', video_id)}")
                st.info("Switch to the 'Ask VidEx' tab to query this video.")
            elif result.get("status") == "error":
                status_placeholder.error(f"Ingestion failed: {result.get('detail', 'unknown error')}")
            elif result.get("status") == "timeout":
                status_placeholder.warning("Still processing after 15 minutes — check back later or try a shorter video.")
            else:
                status_placeholder.warning(f"Unexpected final status: {result}")

    with url_subtab:
        st.caption(
            "Google Drive links work reliably. YouTube links are best-effort — "
            "some videos may fail due to region locks, membership restrictions, "
            "or platform bot-detection outside our control."
        )
        video_url = st.text_input("Google Drive or YouTube URL")

        if st.button("Ingest from URL", disabled=not video_url, key="url_btn"):
            status_placeholder = st.empty()
            try:
                status_placeholder.info("Starting ingestion...")
                resp = requests.post(MODAL_TRIGGER_URL, json={"video_url": video_url}, timeout=30)
                data = resp.json()
            except requests.RequestException as e:
                status_placeholder.error(f"Request failed: {e}")
                st.stop()

            call_id = data.get("call_id")
            if not call_id:
                status_placeholder.error(f"Unexpected response: {data}")
                st.stop()

            result = poll_ingestion(call_id, status_placeholder)

            if result.get("status") == "complete":
                status_placeholder.success(f"Done! video_id: {result.get('video_id')}")
                st.info("Switch to the 'Ask VidEx' tab to query this video.")
            elif result.get("status") == "error":
                status_placeholder.error(f"Ingestion failed: {result.get('detail', 'unknown error')}")
            elif result.get("status") == "timeout":
                status_placeholder.warning("Still processing after 15 minutes — check back later.")
            else:
                status_placeholder.warning(f"Unexpected final status: {result}")