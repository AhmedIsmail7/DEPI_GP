import json
import os
import streamlit as st
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded early so Streamlit
# (which runs in its own process) has access to keys like COHERE_API_KEY.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from modules.main_pipeline import run_pipeline
from query_answer import answer_query
from config import TRANSCRIPT_OUTPUT, VISUAL_OUTPUT, VIDEO_URL


def load_output_preview(file_path: str, limit: int = 5):
    if not os.path.exists(file_path):
        return {"exists": False, "count": 0, "preview": []}

    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        preview = data[:limit]
    else:
        preview = [data]

    return {
        "exists": True,
        "count": len(data) if isinstance(data, list) else 1,
        "preview": preview,
    }


st.set_page_config(page_title="Video Insight UI", page_icon="🎬", layout="wide")

st.title("Video Insight")
st.write("Run the video ingestion, transcription, and vision pipeline from a browser.")

with st.sidebar:
    st.header("Settings")
    video_url = st.text_input("Video URL", value=VIDEO_URL)
    run_button = st.button("Run Pipeline", type="primary")
    clear_button = st.button("Clear Outputs")

if run_button:
    if not video_url or not video_url.strip():
        st.error("Please enter a valid video URL.")
    else:
        with st.spinner("Processing video... This may take a few minutes."):
            try:
                run_pipeline(video_url.strip())
                st.success("Pipeline completed successfully.")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

if clear_button:
    for output_file in [TRANSCRIPT_OUTPUT, VISUAL_OUTPUT]:
        if os.path.exists(output_file):
            os.remove(output_file)
    st.info("Output files cleared.")

st.divider()
st.subheader("Ask About the Video")
query_text = st.text_area(
    "Your question",
    placeholder="Example: What is a neural network?",
    height=100,
)
ask_button = st.button("Ask Cohere", type="primary")

if ask_button:
    if not query_text or not query_text.strip():
        st.error("Please enter a question first.")
    else:
        with st.spinner("Searching the stored video context..."):
            try:
                answer, context_chunks = answer_query(query_text.strip(), top_k=3)
                st.success("Answer generated.")
                st.markdown("### Answer")
                st.write(answer)
                st.markdown("### Relevant Sources")
                for index, chunk in enumerate(context_chunks, start=1):
                    st.write(f"{index}. [{chunk.get('timestamp')}s] {chunk.get('text')}")
            except Exception as exc:
                st.error(f"Query failed: {exc}")

st.divider()
st.subheader("Generated Outputs")
col1, col2 = st.columns(2)

with col1:
    st.caption("Transcript")
    transcript_preview = load_output_preview(TRANSCRIPT_OUTPUT)
    if transcript_preview["exists"]:
        st.write(f"Found {transcript_preview['count']} item(s)")
        st.json(transcript_preview["preview"])
    else:
        st.info("No transcript output yet.")

with col2:
    st.caption("Visual Embeddings")
    visual_preview = load_output_preview(VISUAL_OUTPUT)
    if visual_preview["exists"]:
        st.write(f"Found {visual_preview['count']} item(s)")
        st.json(visual_preview["preview"])
    else:
        st.info("No visual output yet.")
