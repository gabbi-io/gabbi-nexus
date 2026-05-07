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
    """
    Serviço de recuperação por case.

    Ajustes principais:
    - índice é SEMPRE por case_id;
    - permite reconstruir o índice no ask usando documents, caso o processo atual não tenha memória do upload;
    - aplica boost para uploads do usuário;
    - mantém knowledge base disponível, mas com peso menor;
    - filtra por case_id também no pgvector.
    """

    def __init__(self):
        self.case_chunks: dict[str, list[ChunkItem]] = {}
        self.case_matrices = {}
        self.case_vectorizers: dict[str, TfidfVectorizer] = {}

        self.vector_backend = os.getenv("VECTOR_BACKEND", "local").lower()
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "1200"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
        self.database_url = os.getenv("DATABASE_URL", "")
        self.collection = os.getenv("PGVECTOR_COLLECTION", "gabbi_chunks")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.case_upload_boost = float(os.getenv("GABBI_CASE_UPLOAD_BOOST", "0.28"))
        self.knowledge_boost = float(os.getenv("GABBI_KNOWLEDGE_BASE_BOOST", "0.04"))
        self.file_focus_extra_boost = float(os.getenv("GABBI_FILE_FOCUS_EXTRA_BOOST", "0.35"))
        self._db_engine = None
        self._table = None

        if self.vector_backend == "pgvector" and HAS_PGVECTOR and self.database_url:
            self._init_pgvector()

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.vector_backend,
            "pgvector_ready": bool(self._table is not None),
            "local_cases_indexed": len(self.case_chunks),
            "local_vectorizers_indexed": len(self.case_vectorizers),
            "case_upload_boost": self.case_upload_boost,
            "knowledge_boost": self.knowledge_boost,
        }

    def has_case_index(self, case_id: str) -> bool:
        return bool(
            case_id in self.case_chunks
            and case_id in self.case_matrices
            and case_id in self.case_vectorizers
            and self.case_chunks.get(case_id)
        )

    def build_case_index(self, case_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        chunks = self._build_chunks(documents)
        self.case_chunks[case_id] = chunks

        texts = [c.text for c in chunks if c.text and c.text.strip()]

        if not texts:
            self.case_vectorizers.pop(case_id, None)
            self.case_matrices.pop(case_id, None)
            return {
                "published": False,
                "backend": self.vector_backend,
                "chunks": 0,
                "warning": "Nenhum texto encontrado para indexação.",
            }

        vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), lowercase=True)
        matrix = vectorizer.fit_transform(texts)

        self.case_vectorizers[case_id] = vectorizer
        self.case_matrices[case_id] = matrix

        publish_info = {
            "published": False,
            "backend": self.vector_backend,
            "chunks": len(chunks),
            "matrix_shape": list(matrix.shape),
        }

        if self.vector_backend == "pgvector" and self._table is not None and self.openai_api_key:
            try:
                self._publish_pgvector(case_id, chunks)
                publish_info = {
                    "published": True,
                    "backend": "pgvector",
                    "chunks": len(chunks),
                    "matrix_shape": list(matrix.shape),
                }
            except Exception as exc:
                publish_info = {
                    "published": False,
                    "backend": "pgvector",
                    "error": str(exc),
                    "chunks": len(chunks),
                    "matrix_shape": list(matrix.shape),
                }

        return publish_info

    def search(
        self,
        case_id: str,
        query: str,
        top_k: int = 5,
        documents: list[dict[str, Any]] | None = None,
        focus: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Busca evidências no índice do case.

        `documents` é opcional e serve para reconstruir o índice quando o processo atual
        não tem o índice em memória, cenário comum quando upload e ask caem em workers diferentes.
        """
        if documents and not self.has_case_index(case_id):
            self.build_case_index(case_id, documents)

        if self.vector_backend == "pgvector" and self._table is not None and self.openai_api_key:
            try:
                found = self._search_pgvector(case_id, query, max(top_k * 4, top_k))
                if found:
                    reranked = self._rerank_results(found, top_k=top_k, focus=focus)
                    # Se houver foco em arquivo e o pgvector não trouxer upload, complementa com índice local.
                    if focus in {"case_upload_first", "hybrid_case_first"} and not self._has_case_upload_result(reranked):
                        local = self._search_local(case_id, query, top_k=top_k, focus=focus)
                        return self._merge_results(local, reranked, top_k)
                    return reranked
            except Exception:
                pass

        return self._search_local(case_id, query, top_k=top_k, focus=focus)

    def clear_case_index(self, case_id: str) -> None:
        self.case_chunks.pop(case_id, None)
        self.case_matrices.pop(case_id, None)
        self.case_vectorizers.pop(case_id, None)

    def _search_local(self, case_id: str, query: str, top_k: int, focus: str | None = None) -> list[dict[str, Any]]:
        chunks = self.case_chunks.get(case_id, [])
        if not chunks:
            return []

        matrix = self.case_matrices.get(case_id)
        vectorizer = self.case_vectorizers.get(case_id)

        if matrix is None or vectorizer is None:
            return []

        try:
            query_vec = vectorizer.transform([query])

            if matrix.shape[1] != query_vec.shape[1]:
                return []

            raw_scores = (matrix @ query_vec.T).toarray().ravel()
            scores = np.array(
                [
                    float(score) + self._source_boost(chunks[int(idx)], focus=focus)
                    for idx, score in enumerate(raw_scores)
                ]
            )
        except ValueError:
            return []

        # Mesmo em pergunta genérica, retorna os chunks com melhor boost do case atual.
        order = np.argsort(-scores)[:top_k]
        items = []

        for idx in order:
            chunk = chunks[int(idx)]
            items.append(
                {
                    "filename": chunk.filename,
                    "chunk_id": chunk.chunk_id,
                    "type": "text",
                    "score": round(float(scores[idx]), 4),
                    "excerpt": chunk.text[:1600],
                    "metadata": chunk.metadata or {},
                }
            )

        return items

    def _build_chunks(self, documents: list[dict[str, Any]]) -> list[ChunkItem]:
        chunks: list[ChunkItem] = []

        for doc in documents:
            parsed = doc.get("parsed", {}) or {}
            text = (parsed.get("text") or "").strip()

            if not text:
                # Fallback: alguns fluxos salvam conteúdo cru em fields diferentes.
                text = (doc.get("text") or doc.get("content") or "").strip()

            if not text:
                continue

            parts = self._chunk_text(text)
            source_type = self._document_source_type(doc)

            for i, part in enumerate(parts, start=1):
                chunks.append(
                    ChunkItem(
                        chunk_id=f"{doc.get('id', 'doc')}_{i}",
                        filename=doc.get("filename", "arquivo"),
                        text=part,
                        metadata={
                            "document_id": doc.get("id"),
                            "source_path": str(doc.get("path", "")),
                            "content_type": doc.get("content_type"),
                            "source": doc.get("source") or source_type,
                            "source_type": source_type,
                            "external_id": doc.get("external_id"),
                            "topic_id": doc.get("topic_id"),
                        },
                    )
                )

        return chunks

    def _document_source_type(self, doc: dict[str, Any]) -> str:
        content_type = str(doc.get("content_type") or "").lower()
        source = str(doc.get("source") or "").lower()
        filename = str(doc.get("filename") or "").lower()

        if "database-record" in content_type or "postgres" in source or "article" in source:
            return "knowledge_base"
        if filename.startswith("gabbi_article_"):
            return "knowledge_base"
        if filename.startswith("gabbi_knowledge_"):
            return "knowledge_base"
        return "case_upload"

    def _source_boost(self, chunk: ChunkItem, focus: str | None = None) -> float:
        metadata = chunk.metadata or {}
        source_type = str(metadata.get("source_type") or metadata.get("source") or "").lower()
        filename = (chunk.filename or "").lower()

        boost = 0.0
        if source_type == "case_upload":
            boost += self.case_upload_boost
            if focus in {"case_upload_first", "hybrid_case_first"}:
                boost += self.file_focus_extra_boost
        elif source_type == "knowledge_base":
            boost += self.knowledge_boost
        elif any(filename.endswith(ext) for ext in [".xlsx", ".xls", ".csv", ".pdf", ".docx", ".txt", ".md"]):
            boost += self.case_upload_boost / 2

        # Nunca deixa o CSV legado dominar caso ainda exista em cases antigos.
        if filename.startswith("gabbi_knowledge_table_active"):
            boost -= 1.0

        return boost

    def _rerank_results(self, found: list[dict[str, Any]], top_k: int, focus: str | None = None) -> list[dict[str, Any]]:
        reranked = []
        for item in found:
            cloned = dict(item)
            metadata = cloned.get("metadata") or {}
            pseudo_chunk = ChunkItem(
                chunk_id=str(cloned.get("chunk_id", "")),
                filename=str(cloned.get("filename", "")),
                text=str(cloned.get("excerpt", "")),
                metadata=metadata,
            )
            cloned["score"] = round(float(cloned.get("score") or 0.0) + self._source_boost(pseudo_chunk, focus=focus), 4)
            reranked.append(cloned)
        return sorted(reranked, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]

    def _merge_results(self, first: list[dict[str, Any]], second: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        out = []
        seen = set()
        for item in (first or []) + (second or []):
            key = (str(item.get("filename")), str(item.get("chunk_id")), str(item.get("type")))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return sorted(out, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]

    def _has_case_upload_result(self, items: list[dict[str, Any]]) -> bool:
        for item in items or []:
            metadata = item.get("metadata") or {}
            if str(metadata.get("source_type") or metadata.get("source") or "").lower() == "case_upload":
                return True
        return False

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
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _publish_pgvector(self, case_id: str, chunks: list[ChunkItem]):
        texts = [c.text for c in chunks]

        if not texts:
            return

        embeddings = self._embed_texts(texts)

        with Session(self._db_engine) as session:
            session.execute(delete(self._table).where(self._table.c.case_id == case_id))

            for chunk, emb in zip(chunks, embeddings):
                session.execute(
                    self._table.insert().values(
                        case_id=case_id,
                        chunk_id=chunk.chunk_id,
                        filename=chunk.filename,
                        text=chunk.text,
                        meta=chunk.metadata or {},
                        embedding=emb,
                    )
                )

            session.commit()

    def _search_pgvector(self, case_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        [query_embedding] = self._embed_texts([query])
        distance = self._table.c.embedding.cosine_distance(query_embedding)

        stmt = (
            select(
                self._table.c.filename,
                self._table.c.chunk_id,
                self._table.c.text,
                self._table.c.meta,
                distance.label("distance"),
            )
            .where(self._table.c.case_id == case_id)
            .order_by(distance)
            .limit(top_k)
        )

        with Session(self._db_engine) as session:
            rows = session.execute(stmt).all()

        out = []

        for row in rows:
            score = max(0.0, 1 - float(row.distance))
            out.append(
                {
                    "filename": row.filename,
                    "chunk_id": row.chunk_id,
                    "type": "text",
                    "score": round(score, 4),
                    "excerpt": row.text[:1600],
                    "metadata": row.meta or {},
                }
            )

        return out
