from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, JSON, delete, select
    from sqlalchemy.orm import Session
    HAS_PGVECTOR = True
except Exception:
    HAS_PGVECTOR = False

from openai import OpenAI

@dataclass
class ChunkItem:
    chunk_id: str
    filename: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] | None = None

class RetrievalService:
    def __init__(self):
        self.case_chunks: dict[str, list[ChunkItem]] = {}
        self.case_matrices = {}
        self.vectorizer = TfidfVectorizer(stop_words=None)
        self.vector_backend = os.getenv("VECTOR_BACKEND", "local").lower()
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "1200"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
        self.database_url = os.getenv("DATABASE_URL", "")
        self.collection = os.getenv("PGVECTOR_COLLECTION", "gabbi_chunks")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self._db_engine = None
        self._table = None
        if self.vector_backend == "pgvector" and HAS_PGVECTOR and self.database_url:
            self._init_pgvector()

    def status(self) -> dict[str, Any]:
        return {"backend": self.vector_backend, "pgvector_ready": bool(self._table is not None), "local_cases_indexed": len(self.case_chunks)}

    def build_case_index(self, case_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        chunks = self._build_chunks(documents)
        self.case_chunks[case_id] = chunks
        texts = [c.text for c in chunks] or [""]
        matrix = self.vectorizer.fit_transform(texts)
        self.case_matrices[case_id] = matrix
        publish_info = {"published": False, "backend": self.vector_backend, "chunks": len(chunks)}
        if self.vector_backend == "pgvector" and self._table is not None and self.openai_api_key:
            try:
                self._publish_pgvector(case_id, chunks)
                publish_info = {"published": True, "backend": "pgvector", "chunks": len(chunks)}
            except Exception as exc:
                publish_info = {"published": False, "backend": "pgvector", "error": str(exc), "chunks": len(chunks)}
        return publish_info

    def search(self, case_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.vector_backend == "pgvector" and self._table is not None and self.openai_api_key:
            try:
                found = self._search_pgvector(case_id, query, top_k)
                if found:
                    return found
            except Exception:
                pass
        chunks = self.case_chunks.get(case_id, [])
        if not chunks:
            return []
        matrix = self.case_matrices.get(case_id)
        query_vec = self.vectorizer.transform([query])
        scores = (matrix @ query_vec.T).toarray().ravel()
        order = np.argsort(-scores)[:top_k]
        items = []
        for idx in order:
            chunk = chunks[int(idx)]
            items.append({
                "filename": chunk.filename,
                "chunk_id": chunk.chunk_id,
                "type": "text",
                "score": round(float(scores[idx]), 4),
                "excerpt": chunk.text[:1000],
                "metadata": chunk.metadata or {},
            })
        return items

    def _build_chunks(self, documents: list[dict[str, Any]]) -> list[ChunkItem]:
        chunks = []
        for doc in documents:
            parsed = doc.get("parsed", {})
            text = (parsed.get("text") or "").strip()
            if not text:
                continue
            parts = self._chunk_text(text)
            for i, part in enumerate(parts, start=1):
                chunks.append(ChunkItem(chunk_id=f"{doc.get('id','doc')}_{i}", filename=doc.get("filename", "arquivo"), text=part, metadata={"document_id": doc.get("id"), "source_path": doc.get("path")}))
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        text = " ".join(text.split())
        if len(text) <= self.max_chunk_size:
            return [text]
        out = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            out.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return out

    def _init_pgvector(self):
        self._db_engine = create_engine(self.database_url, future=True)
        metadata = MetaData()
        self._table = Table(
            self.collection,
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("case_id", String(120), index=True),
            Column("chunk_id", String(255), index=True),
            Column("filename", String(255)),
            Column("text", Text),
            Column("meta", JSON),
            Column("embedding", Vector(1536)),
            extend_existing=True,
        )
        metadata.create_all(self._db_engine)

    def _embedding_client(self):
        return OpenAI(api_key=self.openai_api_key)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = self._embedding_client()
        response = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in response.data]

    def _publish_pgvector(self, case_id: str, chunks: list[ChunkItem]):
        texts = [c.text for c in chunks]
        if not texts:
            return
        embeddings = self._embed_texts(texts)
        with Session(self._db_engine) as session:
            session.execute(delete(self._table).where(self._table.c.case_id == case_id))
            for chunk, emb in zip(chunks, embeddings):
                session.execute(self._table.insert().values(case_id=case_id, chunk_id=chunk.chunk_id, filename=chunk.filename, text=chunk.text, meta=chunk.metadata or {}, embedding=emb))
            session.commit()

    def _search_pgvector(self, case_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        [query_embedding] = self._embed_texts([query])
        distance = self._table.c.embedding.cosine_distance(query_embedding)
        stmt = select(self._table.c.filename, self._table.c.chunk_id, self._table.c.text, self._table.c.meta, distance.label("distance")).where(self._table.c.case_id == case_id).order_by(distance).limit(top_k)
        with Session(self._db_engine) as session:
            rows = session.execute(stmt).all()
        out = []
        for row in rows:
            score = max(0.0, 1 - float(row.distance))
            out.append({"filename": row.filename, "chunk_id": row.chunk_id, "type": "text", "score": round(score, 4), "excerpt": row.text[:1000], "metadata": row.meta or {}})
        return out
