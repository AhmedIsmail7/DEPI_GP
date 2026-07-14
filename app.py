"""
Streamlit frontend app. Has tabs for chatting and uploading videos.
"""

import time
import requests
import streamlit as st

from modules.database import db_manager
from modules.retrieval import retriever
from modules.llm_handler import llm_handler

MODAL_UPLOAD_URL = "https://dark0danger--videx-ingestion-upload.modal.run"
MODAL_TRIGGER_URL = "https://dark0danger--videx-ingestion-trigger.modal.run"
MODAL_STATUS_URL = "https://dark0danger--videx-ingestion-status.modal.run"

POLL_INTERVAL_SECONDS = 4
POLL_TIMEOUT_SECONDS = 900  # 15 min ceiling, matches Modal function timeout

st.set_page_config(page_title="VidEx", page_icon="🎓", layout="centered")

# add some css to make arabic text look right
st.markdown("""
    <style>
    /* Make markdown text direction automatically adjust based on content (Arabic vs English) */
    .stMarkdown p, .stMarkdown li, .stMarkdown div {
        unicode-bidi: plaintext;
        text-align: start;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 VidEx")
st.caption("Multimodal RAG assistant for educational video content")


def poll_ingestion(call_id: str, status_placeholder) -> dict:
    """
    keep checking modal until video is done processing and show elapsed time.
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
        # still running, keep waiting

        time.sleep(POLL_INTERVAL_SECONDS)

    return {"status": "timeout"}


tab_ask, tab_ingest = st.tabs(["Ask VidEx", "Add a video"])

# ---------------------------------------------------------------------------
# --- Tab 1: Ask VidEx ---
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
        
        if "vid_start_time" not in st.session_state:
            st.session_state["vid_start_time"] = 0

        if len(selected_video_id) == 11 and not " " in selected_video_id:
            # probably youtube link
            st.video(f"https://www.youtube.com/watch?v={selected_video_id}", start_time=st.session_state["vid_start_time"])
        elif os.path.exists(local_video_path):
            st.video(local_video_path, start_time=st.session_state["vid_start_time"])
        else:
            st.info("Video player unavailable: The original file is not stored locally.")

        st.divider()

        # setup chat state
        session_key = f"messages_{selected_video_id}"
        summary_key = f"summary_{selected_video_id}"
        
        if session_key not in st.session_state:
            st.session_state[session_key] = [
                {"role": "assistant", "content": "Hello! I am the professor. What questions do you have about this video?"}
            ]
            
        if summary_key not in st.session_state:
            st.session_state[summary_key] = ""

        # show old chat
        for idx, msg in enumerate(st.session_state[session_key]):
            with st.chat_message(msg["role"]):
                st.markdown(f'<div dir="auto">{msg["content"]}</div>', unsafe_allow_html=True)
                if "sources" in msg and msg["sources"]:
                    st.markdown("<br><b>Sources (Click to jump in video):</b>", unsafe_allow_html=True)
                    cols = st.columns(len(msg["sources"]) * 2) 
                    for s_idx, ts in enumerate(msg["sources"]):
                        minutes, seconds = divmod(int(ts), 60)
                        with cols[s_idx]:
                            if st.button(f"⏱️ {minutes}:{seconds:02d}", key=f"hist_btn_{selected_video_id}_{idx}_{ts}"):
                                st.session_state["vid_start_time"] = int(ts)
                                st.rerun()

        # user can upload screenshot here
        with st.expander("📸 Provide a screenshot of the current frame (optional)"):
            st.caption("Stuck on an equation? Take a screenshot of the video and drop it here so I can see it!")
            frame_upload = st.file_uploader("Upload Frame", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        # chat box
        if query := st.chat_input("Ask a question about the video..."):
            # Render user message instantly
            st.session_state[session_key].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # deal with screenshot if user added one
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
                            current_frame_path=frame_path,
                            # don't include the current question in history
                            chat_history=st.session_state[session_key][:-1],
                            rolling_summary=st.session_state[summary_key]
                        )
                        
                    if getattr(answer, "new_summary", None):
                        st.session_state[summary_key] = answer.new_summary

                    # show answer 
                    st.markdown(f'<div dir="auto">\n\n{answer.answer}\n\n</div>', unsafe_allow_html=True)
                    
                    # save to history
                    sources_list = sorted(set(answer.source_timestamps)) if answer.source_timestamps else []
                    st.session_state[session_key].append({
                        "role": "assistant", 
                        "content": answer.answer,
                        "sources": sources_list
                    })

                    # clickable source buttons
                    if sources_list:
                        st.markdown("<br><b>Sources (Click to jump in video):</b>", unsafe_allow_html=True)
                        cols = st.columns(len(sources_list) * 2) 
                        for idx, ts in enumerate(sources_list):
                            minutes, seconds = divmod(int(ts), 60)
                            with cols[idx]:
                                # use same key so clicking works
                                hist_idx = len(st.session_state[session_key]) - 1
                                if st.button(f"⏱️ {minutes}:{seconds:02d}", key=f"hist_btn_{selected_video_id}_{hist_idx}_{ts}"):
                                    st.session_state["vid_start_time"] = int(ts)
                                    st.rerun()

            # delete temp file
            if frame_path and os.path.exists(frame_path):
                os.remove(frame_path)

# ---------------------------------------------------------------------------
# --- Tab 2: Add a video ---
# ---------------------------------------------------------------------------
with tab_ingest:
    st.subheader("Add a video")

    upload_subtab, url_subtab = st.tabs(["Upload File (recommended)", "Video URL"])

    with upload_subtab:
        st.caption("this one is the most reliable.")
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
            "drive links are good. youtube might fail sometimes because of bots."
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