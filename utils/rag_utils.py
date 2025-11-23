# utils/rag_utils.py

import os
import logging
from typing import List, Tuple
from models.embeddings import get_embeddings
from config.config import VECTOR_DIR, TOP_K

import faiss
import numpy as np

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple sliding window chunker by words.
    """
    if not text:
        return []

    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


class FAISSStore:
    def __init__(self, dim: int, path: str = VECTOR_DIR):
        self.dim = dim
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.index_file = os.path.join(path, "faiss.index")
        self.meta_file = os.path.join(path, "meta.npy")
        self.index = None
        self.metadata: List[dict] = []

        self._load_or_init()

    def _load_or_init(self):
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                self.metadata = list(np.load(self.meta_file, allow_pickle=True).tolist())
            else:
                self.index = faiss.IndexFlatL2(self.dim)
                self.metadata = []
        except Exception as e:
            logging.exception("Error loading faiss index: %s", e)
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []

    def save(self):
        try:
            faiss.write_index(self.index, self.index_file)
            np.save(self.meta_file, np.array(self.metadata, dtype=object))
        except Exception as e:
            logging.exception("Failed to save FAISS index: %s", e)

    def add(self, vectors: List[List[float]], metas: List[dict]):
        try:
            arr = np.array(vectors).astype("float32")
            self.index.add(arr)
            self.metadata.extend(metas)
            self.save()
        except Exception as e:
            logging.exception("Failed to add vectors: %s", e)

    def query(self, vector: List[float], k: int = TOP_K) -> List[Tuple[dict, float]]:
        """
        Safe query — handles empty index and metadata bounds.
        """
        try:
            if self.index.ntotal == 0:
                return []

            vec = np.array([vector]).astype("float32")
            D, I = self.index.search(vec, k)
            results = []
            for idx, dist in zip(I[0], D[0]):
                if 0 <= idx < len(self.metadata):
                    results.append((self.metadata[idx], float(dist)))
            return results
        except Exception as e:
            logging.exception("FAISS query failed: %s", e)
            return []

    def clear(self):
        """
        Clear index and metadata, and delete saved files.
        """
        try:
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            if os.path.exists(self.meta_file):
                os.remove(self.meta_file)
        except Exception as e:
            logging.exception("Failed to clear FAISS store: %s", e)

    def list_sources(self) -> List[Tuple[str, int]]:
        """
        Returns list of (source_name, chunk_count) for UI display.
        """
        if not self.metadata:
            return []

        counts = {}
        for m in self.metadata:
            src = m.get("source", "Unknown")
            counts[src] = counts.get(src, 0) + 1

        return sorted(counts.items(), key=lambda x: x[0])


def build_or_update_index_from_documents(docs: List[dict], store: FAISSStore):
    """
    docs: List[{"id":..., "text":..., "source":...}]
    Chunk, embed, upsert.
    """
    all_chunks = []
    metas = []
    for doc in docs:
        chunks = chunk_text(doc.get("text", ""))
        for i, c in enumerate(chunks):
            metas.append({
                "source": doc.get("source"),
                "doc_id": doc.get("id"),
                "chunk_index": i,
                "text": c
            })
            all_chunks.append(c)

    embeddings = get_embeddings(all_chunks)
    if embeddings:
        store.add(embeddings, metas)
