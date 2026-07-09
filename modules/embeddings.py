# modules/embeddings.py
"""
Vedex - Preprocessing Embeddings Module
========================================
This module implements the singleton EmbeddingManager using Google SigLIP2
for generating L2-normalized text and image embeddings with persistent caching.
"""

import os
import json
import hashlib
import io
import logging
import threading
from pathlib import Path
from typing import List, Union, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, SiglipModel

# Initialize logger
logger = logging.getLogger("vedex.embeddings")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EmbeddingManager:
    """
    Singleton class managing Google SigLIP2 model inference and embedding caches.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(EmbeddingManager, cls).__new__(cls, *args, **kwargs)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "google/siglip2-base-patch16-224"
        
        self.model = None
        self.processor = None
        
        # Paths for persistent caches
        self.cache_dir = Path("temp_assets/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.text_cache_path = self.cache_dir / "text_embeddings_cache.json"
        self.image_cache_path = self.cache_dir / "image_embeddings_cache.json"
        
        # Load caches into memory
        self.text_cache = self._load_cache(self.text_cache_path)
        self.image_cache = self._load_cache(self.image_cache_path)
        
        # Lock for cache writing
        self.cache_lock = threading.Lock()
        
        # Cache dirty flags
        self.text_cache_dirty = False
        self.image_cache_dirty = False
        
        # Lazy model loading trigger
        self._load_model()
        self._initialized = True

    def _load_model(self):
        logger.info(f"Loading model '{self.model_name}' on device '{self.device}'...")
        try:
            self.model = SiglipModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
            except Exception as pe:
                logger.warning(f"Failed to load SigLIP2 processor ({pe}). Falling back to SigLIP1 processor...")
                self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            logger.info("Model and processor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model/processor: {e}")
            raise RuntimeError(f"SigLIP2 initialization failed: {e}")

    def _load_cache(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache from {path}: {e}. Starting fresh.")
        return {}

    def _save_cache(self, cache: dict, path: Path):
        with self.cache_lock:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to write cache to {path}: {e}")

    def flush_caches(self):
        """
        Flush dirty caches to disk.
        """
        if self.text_cache_dirty:
            self._save_cache(self.text_cache, self.text_cache_path)
            self.text_cache_dirty = False
            logger.info("Text embedding cache persisted to disk.")
        if self.image_cache_dirty:
            self._save_cache(self.image_cache, self.image_cache_path)
            self.image_cache_dirty = False
            logger.info("Image embedding cache persisted to disk.")

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _hash_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        if isinstance(image, np.ndarray):
            return hashlib.sha256(image.tobytes()).hexdigest()
        elif hasattr(image, "tobytes"):
            return hashlib.sha256(image.tobytes()).hexdigest()
        else:
            # PIL Image
            img_byte_arr = io.BytesIO()
            # Save as PNG to capture content stably
            image.save(img_byte_arr, format='PNG')
            return hashlib.sha256(img_byte_arr.getvalue()).hexdigest()

    @torch.inference_mode()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get L2-normalized text embedding as a 1D numpy array.
        """
        h = self._hash_text(text)
        if h in self.text_cache:
            return np.array(self.text_cache[h], dtype=np.float32)

        inputs = self.processor(text=[text], padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        emb = features[0].cpu().numpy()
        
        self.text_cache[h] = emb.tolist()
        self.text_cache_dirty = True
        return emb

    @torch.inference_mode()
    def get_image_embedding(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Get L2-normalized image embedding as a 1D numpy array.
        """
        h = self._hash_image(image)
        if h in self.image_cache:
            return np.array(self.image_cache[h], dtype=np.float32)

        pil_image = image
        if isinstance(image, np.ndarray):
            # Convert OpenCV BGR to PIL RGB
            pil_image = Image.fromarray(image[..., ::-1])

        inputs = self.processor(images=[pil_image], return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        emb = features[0].cpu().numpy()
        
        self.image_cache[h] = emb.tolist()
        self.image_cache_dirty = True
        return emb

    @torch.inference_mode()
    def batch_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get L2-normalized text embeddings as a 2D numpy array.
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        hashes = [self._hash_text(t) for t in texts]
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        for idx, h in enumerate(hashes):
            if h in self.text_cache:
                results[idx] = self.text_cache[h]
            else:
                missing_indices.append(idx)
                missing_texts.append(texts[idx])

        if missing_texts:
            inputs = self.processor(text=missing_texts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            embs = features.cpu().numpy()

            for idx, raw_idx in enumerate(missing_indices):
                emb_list = embs[idx].tolist()
                results[raw_idx] = emb_list
                self.text_cache[hashes[raw_idx]] = emb_list

            self.text_cache_dirty = True

        return np.array(results, dtype=np.float32)

    @torch.inference_mode()
    def batch_image_embeddings(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Get L2-normalized image embeddings as a 2D numpy array.
        """
        if not images:
            return np.empty((0, 768), dtype=np.float32)

        hashes = [self._hash_image(img) for img in images]
        results = [None] * len(images)
        missing_indices = []
        missing_images = []

        for idx, h in enumerate(hashes):
            if h in self.image_cache:
                results[idx] = self.image_cache[h]
            else:
                missing_indices.append(idx)
                # Convert BGR array to PIL RGB if necessary
                img = images[idx]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img[..., ::-1])
                missing_images.append(img)

        if missing_images:
            inputs = self.processor(images=missing_images, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            embs = features.cpu().numpy()

            for idx, raw_idx in enumerate(missing_indices):
                emb_list = embs[idx].tolist()
                results[raw_idx] = emb_list
                self.image_cache[hashes[raw_idx]] = emb_list

            self.image_cache_dirty = True

        return np.array(results, dtype=np.float32)

    def get_embeddings(self, texts: Union[str, List[str], None] = None, images: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]], None] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None]:
        """
        Flexible getter supporting single or batched text and/or images.
        """
        text_emb = None
        image_emb = None

        if texts is not None:
            if isinstance(texts, list):
                text_emb = self.batch_text_embeddings(texts)
            else:
                text_emb = self.get_text_embedding(texts)

        if images is not None:
            if isinstance(images, list):
                image_emb = self.batch_image_embeddings(images)
            else:
                image_emb = self.get_image_embedding(images)

        if text_emb is not None and image_emb is not None:
            return text_emb, image_emb
        elif text_emb is not None:
            return text_emb
        elif image_emb is not None:
            return image_emb
        return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors (or batches).
        """
        # Since embeddings are already L2 normalized, cosine similarity is simply the dot product
        v1 = vec1.flatten()
        v2 = vec2.flatten()
        return float(np.dot(v1, v2))

# Instantiate global singleton
embedding_manager = EmbeddingManager()
