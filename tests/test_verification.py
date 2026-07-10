# tests/test_verification.py
"""
Vedex - Preprocessing Modules Verification Tests
=================================================
Verifies that EmbeddingManager and VisionProcessor are correct.
"""

import unittest
import numpy as np
from PIL import Image
import torch
import os
import json
from pathlib import Path

from modules.embeddings import embedding_manager
from modules.vision import vision_engine

class TestEmbeddingManager(unittest.TestCase):
    def test_singleton(self):
        from modules.embeddings import EmbeddingManager
        manager2 = EmbeddingManager()
        self.assertIs(embedding_manager, manager2)

    def test_text_embedding_shape_and_norm(self):
        text = "Test text for verification"
        emb = embedding_manager.get_text_embedding(text)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (768,))
        # Assert L2 normalized (dot product with itself is extremely close to 1)
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_image_embedding_shape_and_norm(self):
        # Create a dummy image
        img = Image.new("RGB", (224, 224), color="red")
        emb = embedding_manager.get_image_embedding(img)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (768,))
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_batch_embeddings(self):
        texts = ["Text one", "Text two", "Text three"]
        embs = embedding_manager.batch_text_embeddings(texts)
        self.assertEqual(embs.shape, (3, 768))
        for emb in embs:
            self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=5)

class TestVisionProcessor(unittest.TestCase):
    def test_phash_computation(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add slight change to img2
        img2[10:20, 10:20] = 255
        
        hash1 = vision_engine.calculate_phash(img1)
        hash2 = vision_engine.calculate_phash(img2)
        
        self.assertEqual(len(hash1), 64)
        self.assertEqual(len(hash2), 64)
        
        sim = vision_engine.phash_similarity(hash1, hash2)
        self.assertTrue(0.0 <= sim <= 1.0)
        # Similarity of identical images should be 1.0
        self.assertEqual(vision_engine.phash_similarity(hash1, hash1), 1.0)

    def test_quality_evaluation(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        quality_data = vision_engine.evaluate_quality(img)
        
        self.assertIn("quality", quality_data)
        self.assertIn("brightness", quality_data)
        self.assertIn("sharpness", quality_data)
        self.assertIn("entropy", quality_data)
        self.assertTrue(0.0 <= quality_data["quality"] <= 1.0)

    def test_ocr_score_calculation(self):
        score_empty = vision_engine.calculate_ocr_score("", 0.0)
        self.assertEqual(score_empty, 0.0)
        
        text = "Hello world context check"
        score = vision_engine.calculate_ocr_score(text, 0.9)
        self.assertTrue(score > 0.0)
        self.assertTrue(score <= 1.0)

class TestOrchestrator(unittest.TestCase):
    def test_pipeline_state_init(self):
        from main_preprocessing import PipelineState
        state = PipelineState()
        self.assertEqual(state.completed_stages, [])
        self.assertEqual(state.skipped_stages, [])
        self.assertEqual(state.total_duration, 0.0)
        d = state.to_dict()
        self.assertIn("video_metadata", d)
        self.assertIn("stages", d)
        self.assertIn("durations", d)

    def test_orchestrator_initialization(self):
        from main_preprocessing import PreprocessingOrchestrator
        orchestrator = PreprocessingOrchestrator(force=True)
        self.assertTrue(orchestrator.force)
        self.assertIsNotNone(orchestrator.temp_dir)
        self.assertIsNotNone(orchestrator.cache_dir)

    def test_json_validation(self):
        from main_preprocessing import PreprocessingOrchestrator
        orchestrator = PreprocessingOrchestrator(force=False)
        temp_path = Path("temp_assets/test_dummy_validation.json")
        
        # Test non-existing file
        self.assertFalse(orchestrator.validate_json_file(temp_path))
        
        # Test valid list JSON
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump([{"key": "value"}], f)
        self.assertTrue(orchestrator.validate_json_file(temp_path, expected_type=list))
        
        # Clean up
        if temp_path.exists():
            os.remove(temp_path)


class TestStorageManager(unittest.TestCase):
    """Tests for modules/storage.py using mocked QdrantClient."""

    def _make_transcript(self):
        return [
            {"chunk_index": i, "text": f"transcript chunk {i}",
             "start_time": i * 30.0, "end_time": (i + 1) * 30.0,
             "embedding": [0.1] * 768}
            for i in range(3)
        ]

    def _make_visuals(self):
        return [
            {"chunk_index": i, "frame_time": i * 30.0, "frame_index": i,
             "keyframe_path": f"temp_assets/keyframes/chunk_{i:05d}.jpg",
             "semantic_text": f"ocr text {i}", "quality_score": 0.8,
             "similarity_score": 0.6, "ocr_score": 0.7,
             "start_time": i * 30.0, "end_time": (i + 1) * 30.0,
             "embedding": [0.2] * 768}
            for i in range(3)
        ]

    def _make_report(self):
        return {"video_metadata": {"path": "temp_assets/yt_video.mp4",
                                   "source": "local", "gpu_detected": False}}

    def test_schema_validation_valid(self):
        from modules.storage import _validate_transcript_chunks, _validate_visual_embeddings
        chunks = self._make_transcript()
        result = _validate_transcript_chunks(chunks, Path("transcript_chunks.json"))
        self.assertEqual(len(result), 3)

        visuals = self._make_visuals()
        result2 = _validate_visual_embeddings(visuals, Path("visual_embeddings.json"))
        self.assertEqual(len(result2), 3)

    def test_schema_validation_rejects_non_list(self):
        from modules.storage import _validate_transcript_chunks, SchemaError
        with self.assertRaises(SchemaError):
            _validate_transcript_chunks({"not": "a list"}, Path("x.json"))

    def test_schema_validation_rejects_missing_keys(self):
        from modules.storage import _validate_transcript_chunks, SchemaError
        with self.assertRaises(SchemaError):
            _validate_transcript_chunks([{"text": "hi"}], Path("x.json"))  # no embedding

    def test_deterministic_id_stable(self):
        from modules.storage import _deterministic_id
        id1 = _deterministic_id("video_abc", "text_chunks", 5)
        id2 = _deterministic_id("video_abc", "text_chunks", 5)
        self.assertEqual(id1, id2)

    def test_deterministic_id_unique_per_chunk(self):
        from modules.storage import _deterministic_id
        id1 = _deterministic_id("video_abc", "text_chunks", 0)
        id2 = _deterministic_id("video_abc", "text_chunks", 1)
        self.assertNotEqual(id1, id2)

    def test_indexing_result_to_dict(self):
        from modules.storage import IndexingResult, CollectionStats
        res = IndexingResult(success=True, video_id="test_video")
        res.text_stats.uploaded = 10
        res.image_stats.uploaded = 10
        res.total_duration_seconds = 1.23
        d = res.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["video_id"], "test_video")
        self.assertIn("text_chunks", d["collections"])
        self.assertIn("video_frames", d["collections"])
        self.assertEqual(d["collections"]["text_chunks"]["uploaded"], 10)

    def test_full_indexing_with_mock(self):
        """Full pipeline test using a mocked QdrantClient."""
        from unittest.mock import MagicMock, patch
        from modules.storage import StorageManager

        dummy_chunks  = self._make_transcript()
        dummy_visuals = self._make_visuals()
        dummy_report  = self._make_report()

        # Write temp JSON files
        tmp = Path("temp_assets")
        tmp.mkdir(exist_ok=True)
        t_path = tmp / "_test_transcript.json"
        v_path = tmp / "_test_visual.json"
        r_path = tmp / "_test_report.json"
        t_path.write_text(json.dumps(dummy_chunks),  encoding="utf-8")
        v_path.write_text(json.dumps(dummy_visuals), encoding="utf-8")
        r_path.write_text(json.dumps(dummy_report),  encoding="utf-8")

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.retrieve.return_value = []          # no existing points
        mock_client.upsert.return_value = None

        mgr = StorageManager()
        mgr._client = mock_client

        result = mgr.index_preprocessing_outputs(
            transcript_path=t_path,
            visual_path=v_path,
            report_path=r_path,
            video_id="test_video",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.text_stats.uploaded, 3)
        self.assertEqual(result.image_stats.uploaded, 3)
        self.assertEqual(result.text_stats.skipped, 0)
        self.assertEqual(result.text_stats.failed,  0)

        # Cleanup
        for p in [t_path, v_path, r_path]:
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    unittest.main()
