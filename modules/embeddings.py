"""
Unified embedding module — SigLIP encodes both text and images into the
SAME 768-dim space, unlike the previous MiniLM (text) + CLIP (image) setup
where the two embedding spaces were unrelated and only combined via a
weighted-average heuristic in retrieval.py. With a shared space, a single
query embedding can be meaningfully compared against both text and image
vectors directly.

Includes disk-based caching by content hash, so re-processing the same
text/frame (e.g. across retries) doesn't recompute an embedding that's
already been generated.
"""

import os
import json
import hashlib
import threading
from pathlib import Path
from typing import Union

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from config import SIGLIP_MODEL_NAME, TEMP_ASSETS_DIR


class EmbeddingManager:
    """Singleton — loads SigLIP once, reused for both text and image encoding."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

        cache_dir = Path(TEMP_ASSETS_DIR) / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._text_cache_path = cache_dir / "text_cache.json"
        self._image_cache_path = cache_dir / "image_cache.json"
        self._text_cache = self._load_cache(self._text_cache_path)
        self._image_cache = self._load_cache(self._image_cache_path)
        self._cache_lock = threading.Lock()

        self._initialized = True

    @property
    def model(self):
        if self._model is None:
            print(f"Loading SigLIP model '{SIGLIP_MODEL_NAME}' on {self.device}...")
            self._model = SiglipModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device)
            self._model.eval()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        return self._processor

    def _load_cache(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self, cache: dict, path: Path):
        with self._cache_lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache, f)

    def flush_caches(self):
        """Call after a batch of embeddings to persist the cache to disk."""
        self._save_cache(self._text_cache, self._text_cache_path)
        self._save_cache(self._image_cache, self._image_cache_path)

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _hash_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        if isinstance(image, np.ndarray):
            return hashlib.sha256(image.tobytes()).hexdigest()
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    @torch.inference_mode()
    def get_text_embedding(self, text: str) -> list[float]:
        """Returns an L2-normalized 768-dim embedding, in the same space as image embeddings."""
        h = self._hash_text(text)
        if h in self._text_cache:
            return self._text_cache[h]

        inputs = self.processor(
            text=[text], padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        ).to(self.device)
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        emb = features[0].cpu().tolist()

        self._text_cache[h] = emb
        return emb

    @torch.inference_mode()
    def get_image_embedding(self, image: Union[Image.Image, np.ndarray]) -> list[float]:
        """Returns an L2-normalized 768-dim embedding, in the same space as text embeddings."""
        h = self._hash_image(image)
        if h in self._image_cache:
            return self._image_cache[h]

        pil_image = image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image[..., ::-1])  # BGR (OpenCV) -> RGB

        inputs = self.processor(images=[pil_image], return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        emb = features[0].cpu().tolist()

        self._image_cache[h] = emb
        return emb


embedding_manager = EmbeddingManager()