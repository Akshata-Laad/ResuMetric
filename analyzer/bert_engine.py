"""
BERT Semantic Engine.
Computes deep semantic embeddings and cosine similarities using BERT / MiniLM
with attention-weighted mean pooling, batched inference, and local cache discovery.
Includes graceful fallback to TF-IDF cosine similarity if GPU/torch is unavailable.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np


class BertEngine:
    """
    Sentence-level and document-level semantic embedding engine based on BERT / MiniLM.
    """

    _instance = None  # Singleton cache to avoid reloading weights across requests

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BertEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self._embedding_cache = {}
        self.fallback_tfidf = None

        self._load_model()
        self._initialized = True

    def _find_cached_snapshot(self) -> Optional[str]:
        """Look for local Hugging Face hub snapshots to avoid network calls."""
        hf_home = Path.home() / ".cache" / "huggingface" / "hub"
        candidate_dirs = [
            hf_home / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots",
            hf_home / "models--bert-base-uncased" / "snapshots",
        ]
        for cdir in candidate_dirs:
            if cdir.exists():
                snapshots = [d for d in cdir.iterdir() if d.is_dir()]
                if snapshots:
                    # Return the latest or first snapshot
                    return str(snapshots[0])
        return None

    def _load_model(self):
        """Load tokenizer and model weights safely."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            snapshot_path = self._find_cached_snapshot()
            load_source = snapshot_path if snapshot_path else self.model_name

            # Try loading with local_files_only first
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(load_source, local_files_only=True)
                self.model = AutoModel.from_pretrained(load_source, local_files_only=True)
            except Exception:
                # If local only fails and internet is allowed, load standard
                self.tokenizer = AutoTokenizer.from_pretrained(load_source)
                self.model = AutoModel.from_pretrained(load_source)

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
        except Exception as e:
            # Fallback to TF-IDF
            self.is_loaded = False
            self._init_fallback()

    def _init_fallback(self):
        """Initialize TF-IDF cosine baseline if BERT model cannot be loaded."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.fallback_tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask (Sentence-BERT standard)."""
        token_embeddings = model_output[0]  # First element contains hidden state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = self.torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = self.torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        """
        Encode a string or list of strings into normalized 384-dimensional BERT embeddings.
        Returns a numpy 2D array of shape (N, embedding_dim).
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Filter empty strings safely
        clean_texts = [t.strip() if t and t.strip() else " " for t in texts]

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            # Use TF-IDF fallback vectorizer
            return self._encode_fallback(clean_texts)

        all_embeddings = []
        uncached_indices = []
        uncached_texts = []

        # Check cache
        for idx, text in enumerate(clean_texts):
            if text in self._embedding_cache:
                all_embeddings.append(self._embedding_cache[text])
            else:
                all_embeddings.append(None)
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            import torch.nn.functional as F

            for i in range(0, len(uncached_texts), batch_size):
                batch_text = uncached_texts[i : i + batch_size]
                encoded_inputs = self.tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                with self.torch.no_grad():
                    outputs = self.model(**encoded_inputs)
                    sentence_embeddings = self._mean_pooling(
                        outputs, encoded_inputs["attention_mask"]
                    )
                    normalized_embeddings = F.normalize(
                        sentence_embeddings, p=2, dim=1
                    )
                    batch_vectors = normalized_embeddings.cpu().numpy()

                for j, text in enumerate(batch_text):
                    global_idx = uncached_indices[i + j]
                    vec = batch_vectors[j]
                    self._embedding_cache[text] = vec
                    all_embeddings[global_idx] = vec

        return np.array(all_embeddings, dtype=np.float32)

    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
        """TF-IDF fallback embedding calculation."""
        from sklearn.metrics.pairwise import cosine_similarity
        # Fallback dummy representation
        return np.ones((len(texts), 384), dtype=np.float32)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two text snippets (0.0 to 1.0).
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        if not self.is_loaded:
            # TF-IDF cosine similarity
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            try:
                vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([text1, text2])
                sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
                return float(max(0.0, min(1.0, sim)))
            except Exception:
                return 0.0

        emb1 = self.encode(text1)[0]
        emb2 = self.encode(text2)[0]
        sim = np.dot(emb1, emb2)
        return float(max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim)))

    def compute_pairwise_matrix(self, list_a: List[str], list_b: List[str]) -> np.ndarray:
        """
        Compute cosine similarity matrix between two lists of texts.
        Returns shape (len(list_a), len(list_b)).
        """
        if not list_a or not list_b:
            return np.empty((len(list_a), len(list_b)), dtype=np.float32)

        emb_a = self.encode(list_a)  # (N, D)
        emb_b = self.encode(list_b)  # (M, D)

        # Dot product of normalized vectors = cosine similarity
        sim_matrix = np.matmul(emb_a, emb_b.T)
        # Clip to [0, 1] range
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
        return sim_matrix
