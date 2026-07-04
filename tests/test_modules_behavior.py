import json
from pathlib import Path

import pytest

from modules import database as database_module
from modules import llm_handler as llm_handler_module
from modules import retrieval as retrieval_module
from modules import utils as utils_module


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.created = []
        self.collection_exists_calls = []
        self.create_collection_calls = []
        self.upsert_calls = []
        self.search_calls = []

    def collection_exists(self, collection_name):
        self.collection_exists_calls.append(collection_name)
        return False

    def create_collection(self, **kwargs):
        self.create_collection_calls.append(kwargs)

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [
            type("Hit", (), {"payload": {"text": "hello", "timestamp": 12.3}, "score": 0.91})()
        ]


class FakeCohereClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.last_request = None

    def chat(self, **kwargs):
        self.last_request = kwargs
        return type("Response", (), {"message": type("Message", (), {"content": [{"text": "done"}]})()})()


class FakeEncoder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def encode(self, text):
        return [0.1, 0.2, 0.3]


def test_utils_save_and_load_json(tmp_path):
    file_path = tmp_path / "data.json"

    payload = {"a": 1, "b": [2, 3]}
    utils_module.save_json(payload, str(file_path))

    assert utils_module.file_exists(str(file_path))
    assert utils_module.load_json(str(file_path)) == payload

    utils_module.delete_file(str(file_path))
    assert not utils_module.file_exists(str(file_path))


def test_llm_handler_generates_response(monkeypatch):
    monkeypatch.setattr(llm_handler_module, "cohere", type("CohereModule", (), {"ClientV2": FakeCohereClient}))

    handler = llm_handler_module.CohereLLMHandler(api_key="abc", model="test-model")
    response = handler.generate_response("who", [{"text": "context", "timestamp": 1}])

    assert response == "done"
    assert handler._get_client().last_request["model"] == "test-model"


def test_retrieval_returns_ranked_hits(monkeypatch):
    monkeypatch.setattr(retrieval_module, "QdrantClient", FakeClient)
    monkeypatch.setattr(retrieval_module, "SentenceTransformer", FakeEncoder)

    retriever = retrieval_module.SemanticRetriever()
    results = retriever.retrieve("hello", top_k=1)

    assert results[0]["text"] == "hello"
    assert results[0]["timestamp"] == 12.3
    assert results[0]["similarity"] == 0.91


def test_database_init_and_upsert(monkeypatch, tmp_path):
    fake_client = FakeClient()
    monkeypatch.setattr(database_module, "QdrantClient", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(database_module, "models", type("Models", (), {
        "VectorParams": lambda *args, **kwargs: {"vector": args, "kwargs": kwargs},
        "Distance": type("Distance", (), {"COSINE": "cosine"}),
        "PointStruct": lambda **kwargs: kwargs,
    }))

    transcript_path = tmp_path / "transcript.json"
    visual_path = tmp_path / "visual.json"
    transcript_path.write_text(json.dumps([{"start": 0, "end": 1, "text": "hello", "embedding": [0.1, 0.2]}]), encoding="utf-8")
    visual_path.write_text(json.dumps([{"chunk_index": 0, "timestamp": 1.0, "embedding": [0.3, 0.4], "similarity_score": 0.9}]), encoding="utf-8")

    manager = database_module.QdrantManager()
    manager.init_collection()
    manager.upsert_data(str(transcript_path), str(visual_path))

    assert fake_client.collection_exists_calls == ["video_knowledge"]
    assert fake_client.create_collection_calls
    assert fake_client.upsert_calls
