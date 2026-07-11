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
    st.subheader("Ask a question about an ingested video")

    try:
        video_ids = db_manager.get_available_video_ids()
    except Exception as e:
        st.error(f"Could not reach Qdrant: {e}")
        video_ids = []

    if not video_ids:
        st.info("No videos ingested yet. Use the 'Add a video' tab first.")
    else:
        selected_video_id = st.selectbox("Video", video_ids)
        query = st.text_input("Your question", placeholder="e.g. What is the difference between velocity and speed?")

        if st.button("Ask", type="primary", disabled=not query):
            with st.spinner("Searching video content..."):
                results = retriever.retrieve(query, top_k=3, video_id=selected_video_id)

            if not results:
                st.warning("No relevant content found for this question in the selected video.")
            else:
                with st.spinner("Generating answer..."):
                    answer = llm_handler.generate_response(query, results, video_id=selected_video_id)

                st.markdown("### Answer")
                st.write(answer.answer)

                st.markdown("### Sources")
                for ts in sorted(set(answer.source_timestamps)):
                    minutes, seconds = divmod(int(ts), 60)
                    # Only reconstructable for YouTube-sourced videos, since
                    # video_id there is YouTube's own ID. Uploaded/Drive
                    # videos don't have a matching public URL to link to.
                    link = f"https://www.youtube.com/watch?v={selected_video_id}&t={int(ts)}s"
                    st.markdown(f"- [{minutes}:{seconds:02d}]({link})")

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