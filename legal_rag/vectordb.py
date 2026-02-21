from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from .config import Config

@dataclass
class LegalVectorDB:
    """ChromaDB-backed vector store for legal chunks."""

    cfg: Config

    def __post_init__(self) -> None:
        self.embedding_model = SentenceTransformer(self.cfg.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(self.cfg.CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name="us_law_knowledge_base",
            metadata={"description": "US Law documents with embeddings"},
        )

    def count(self) -> int:
        return self.collection.count()

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return

        texts = [d["content"] for d in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # deterministic-ish ids per run; for production use stable ids based on source+chunk
        ids = [f"doc_{i}_{abs(hash(text))}" for i, text in enumerate(texts)]
        metadatas = [d["metadata"] for d in documents]

        batch_size = 128
        for i in range(0, len(texts), batch_size):
            j = min(i + batch_size, len(texts))
            self.collection.add(
                embeddings=embeddings[i:j].tolist(),
                documents=texts[i:j],
                metadatas=metadatas[i:j],
                ids=ids[i:j],
            )

    def search(self, query: str, n_results: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filters,
        )

        out: List[Dict[str, Any]] = []
        if results.get("documents"):
            for i in range(len(results["documents"][0])):
                out.append(
                    {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )
        return out
