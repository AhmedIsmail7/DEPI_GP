import json
from pathlib import Path

from app import load_output_preview
from modules.database import export_results_to_json


def test_load_output_preview_handles_missing_and_existing_files(tmp_path):
    missing_file = tmp_path / "missing.json"
    missing_result = load_output_preview(str(missing_file))
    assert missing_result["exists"] is False
    assert missing_result["count"] == 0

    existing_file = tmp_path / "sample.json"
    existing_file.write_text(
        json.dumps([
            {"text": "alpha"},
            {"text": "beta"},
            {"text": "gamma"},
        ]),
        encoding="utf-8",
    )

    existing_result = load_output_preview(str(existing_file), limit=2)
    assert existing_result["exists"] is True
    assert existing_result["count"] == 3
    assert len(existing_result["preview"]) == 2
    assert existing_result["preview"][0]["text"] == "alpha"


def test_export_results_to_json_creates_output_file(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    visual_path = tmp_path / "visual.json"

    transcript_path.write_text(
        json.dumps([
            {"start": 0, "end": 10, "text": "alpha", "embedding": [0.1, 0.2]}
        ]),
        encoding="utf-8",
    )
    visual_path.write_text(
        json.dumps([
            {"chunk_index": 0, "timestamp": 5, "similarity_score": 0.9, "embedding": [0.3, 0.4]}
        ]),
        encoding="utf-8",
    )

    output_path = tmp_path / "Video_result" / "qdrant_results.json"
    exported_path = export_results_to_json(str(transcript_path), str(visual_path), str(output_path))

    assert Path(exported_path).exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["text"] == "alpha"
    assert payload[0]["source"] == "qdrant"
