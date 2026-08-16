from __future__ import annotations

import hashlib
import json
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
    Retrieval isolado por case.

    Garantias:
    - Não usa vectorizer global.
    - Reindexa quando os documentos do case mudam.
    - Busca local do case atual sempre vem antes do pgvector.
    - Upload do usuário recebe boost quando a pergunta fala de arquivo/anexo.
    """

    def __init__(self):
        self.case_chunks: dict[str, list[ChunkItem]] = {}
        self.case_matrices: dict[str, Any] = {}
        self.case_vectorizers: dict[str, TfidfVectorizer] = {}
        self.case_document_fingerprints: dict[str, str] = {}

        self.vector_backend = os.getenv("VECTOR_BACKEND", "local").lower()
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "1600"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "160"))
        self.database_url = os.getenv("DATABASE_URL", "")
        self.collection = os.getenv("PGVECTOR_COLLECTION", "gabbi_chunks")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.case_upload_boost = float(os.getenv("GABBI_CASE_UPLOAD_BOOST", "0.55"))
        self.knowledge_boost = float(os.getenv("GABBI_KNOWLEDGE_BASE_BOOST", "0.04"))
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

    def ensure_case_index(self, case_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        fingerprint = self._fingerprint_documents(documents)
        if (
            case_id in self.case_chunks
            and self.case_document_fingerprints.get(case_id) == fingerprint
            and self.case_matrices.get(case_id) is not None
            and self.case_vectorizers.get(case_id) is not None
        ):
            return {"rebuilt": False, "chunks": len(self.case_chunks.get(case_id, []))}
        info = self.build_case_index(case_id, documents)
        info["rebuilt"] = True
        return info

    def build_case_index(self, case_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        chunks = self._build_chunks(documents)
        self.case_chunks[case_id] = chunks
        self.case_document_fingerprints[case_id] = self._fingerprint_documents(documents)

        texts = [c.text for c in chunks if c.text and c.text.strip()]
        if not texts:
            self.case_vectorizers.pop(case_id, None)
            self.case_matrices.pop(case_id, None)
            return {"published": False, "backend": self.vector_backend, "chunks": 0, "warning": "Nenhum texto encontrado para indexação."}

        vectorizer = TfidfVectorizer(
            stop_words=None,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w[\w:/\-\.]+\b",
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(texts)

        self.case_vectorizers[case_id] = vectorizer
        self.case_matrices[case_id] = matrix

        info = {"published": False, "backend": self.vector_backend, "chunks": len(chunks), "matrix_shape": list(matrix.shape)}

        if self.vector_backend == "pgvector" and self._table is not None and self.openai_api_key:
            try:
                self._publish_pgvector(case_id, chunks)
                info.update({"published": True, "backend": "pgvector"})
            except Exception as exc:
                info.update({"published": False, "backend": "pgvector", "error": str(exc)})
        return info

    def search(self, case_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        local = self._search_local(case_id, query, top_k=max(top_k, 8))
        if local:
            return local[:top_k]

        if self.vector_backend == "pgvector" and self._table is not None and self.openai_api_key:
            try:
                found = self._search_pgvector(case_id, query, max(top_k * 3, top_k))
                if found:
                    return self._rerank_results(found, top_k=top_k)
            except Exception:
                pass

        return []

    def search_with_document_coverage(
        self,
        case_id: str,
        query: str,
        *,
        top_k: int = 16,
        per_document: int = 1,
    ) -> list[dict[str, Any]]:
        """Busca semântica com cobertura mínima por documento.

        Indicada para perguntas case-wide, como "analise todos os documentos",
        "faça a triagem completa" ou "considere todos os anexos".

        Estratégia:
        1. calcula relevância de todos os chunks do case;
        2. seleciona os melhores ``per_document`` chunks de cada documento;
        3. completa o resultado com os melhores chunks globais ainda não usados;
        4. preserva o isolamento por ``case_id``.

        Isso evita que um único arquivo com muitos chunks ocupe todo o TOP-K e
        faça outros documentos desaparecerem do contexto enviado ao LLM.
        """
        chunks = self.case_chunks.get(case_id, [])
        matrix = self.case_matrices.get(case_id)
        vectorizer = self.case_vectorizers.get(case_id)

        if not chunks or matrix is None or vectorizer is None:
            return []

        scores = self._score_local_chunks(case_id, query)
        if scores is None:
            return []

        # Agrupa pelo document_id; filename é fallback para índices antigos.
        by_document: dict[str, list[tuple[int, float]]] = {}
        for idx, chunk in enumerate(chunks):
            metadata = chunk.metadata or {}
            document_key = str(
                metadata.get("document_id")
                or chunk.filename
                or f"document_{idx}"
            )
            by_document.setdefault(document_key, []).append((idx, float(scores[idx])))

        selected: list[int] = []
        selected_set: set[int] = set()
        per_document = max(1, int(per_document or 1))

        # Primeiro garante cobertura: pelo menos um chunk de cada documento.
        document_heads: list[tuple[float, str, list[tuple[int, float]]]] = []
        for document_key, document_chunks in by_document.items():
            ranked = sorted(document_chunks, key=lambda item: item[1], reverse=True)
            best_score = ranked[0][1] if ranked else 0.0
            document_heads.append((best_score, document_key, ranked))

        # Mantém documentos ordenados por relevância, sem perder cobertura.
        document_heads.sort(key=lambda item: item[0], reverse=True)
        for _, _, ranked in document_heads:
            for idx, _score in ranked[:per_document]:
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)

        # Em case-wide, top_k nunca pode ser menor que o número de documentos.
        desired_total = max(int(top_k or 0), len(by_document) * per_document)

        # Depois complementa com os melhores chunks do ranking global.
        for idx in np.argsort(-scores):
            idx = int(idx)
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= desired_total:
                break

        return [self._chunk_to_result(chunks[idx], float(scores[idx])) for idx in selected]

    def get_case_chunks(self, case_id: str) -> list[ChunkItem]:
        return self.case_chunks.get(case_id, [])

    def clear_case_index(self, case_id: str) -> None:
        self.case_chunks.pop(case_id, None)
        self.case_matrices.pop(case_id, None)
        self.case_vectorizers.pop(case_id, None)
        self.case_document_fingerprints.pop(case_id, None)

    def _score_local_chunks(self, case_id: str, query: str) -> np.ndarray | None:
        chunks = self.case_chunks.get(case_id, [])
        matrix = self.case_matrices.get(case_id)
        vectorizer = self.case_vectorizers.get(case_id)

        if not chunks or matrix is None or vectorizer is None:
            return None

        try:
            query_vec = vectorizer.transform([query])
            if matrix.shape[1] != query_vec.shape[1]:
                return None
            raw_scores = (matrix @ query_vec.T).toarray().ravel()
            scores = np.array([
                float(score) + self._source_boost(chunks[int(idx)])
                for idx, score in enumerate(raw_scores)
            ])
        except Exception:
            return None

        q = str(query or "").lower()
        wants_file = any(
            term in q
            for term in [
                "arquivo", "arquivos", "anexo", "anexos", "documento", "documentos",
                "planilha", "upload", "pdf", "txt", "csv", "xlsx", "xls",
            ]
        )
        if wants_file:
            for idx, chunk in enumerate(chunks):
                if self._source_type(chunk) == "case_upload":
                    scores[idx] += 1.0
        return scores

    @staticmethod
    def _chunk_to_result(chunk: ChunkItem, score: float) -> dict[str, Any]:
        return {
            "filename": chunk.filename,
            "chunk_id": chunk.chunk_id,
            "type": "text",
            "score": round(float(score), 4),
            "excerpt": chunk.text[:2200],
            "metadata": chunk.metadata or {},
        }

    def _search_local(self, case_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        chunks = self.case_chunks.get(case_id, [])
        scores = self._score_local_chunks(case_id, query)
        if not chunks or scores is None:
            return []

        order = np.argsort(-scores)[:top_k]
        return [
            self._chunk_to_result(chunks[int(idx)], float(scores[int(idx)]))
            for idx in order
        ]

    def _build_chunks(self, documents: list[dict[str, Any]]) -> list[ChunkItem]:
        chunks: list[ChunkItem] = []
        for doc in documents or []:
            parsed = doc.get("parsed", {}) or {}
            text = (parsed.get("text") or "").strip()
            if not text:
                continue

            source_type = self._document_source_type(doc)
            for i, part in enumerate(self._chunk_text(text), start=1):
                chunks.append(
                    ChunkItem(
                        chunk_id=f"{doc.get('id', 'doc')}_{i}",
                        filename=str(doc.get("filename") or "arquivo"),
                        text=part,
                        metadata={
                            "document_id": doc.get("id"),
                            "source_path": str(doc.get("path", "")),
                            "content_type": doc.get("content_type"),
                            "source": doc.get("source") or source_type,
                            "source_type": source_type,
                            "external_id": doc.get("external_id"),
                            "topic_id": doc.get("topic_id") or (doc.get("metadata") or {}).get("topic_id"),
                        },
                    )
                )
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        text = " ".join(str(text or "").split())
        if not text:
            return []
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

    def _document_source_type(self, doc: dict[str, Any]) -> str:
        content_type = str(doc.get("content_type") or "").lower()
        source = str(doc.get("source") or "").lower()
        filename = str(doc.get("filename") or "").lower()
        if "database-record" in content_type or "postgres" in source or "article" in source:
            return "knowledge_base"
        if filename.startswith("gabbi_article_") or filename.startswith("gabbi_knowledge_"):
            return "knowledge_base"
        return "case_upload"

    def _source_type(self, chunk: ChunkItem) -> str:
        metadata = chunk.metadata or {}
        return str(metadata.get("source_type") or metadata.get("source") or "").lower()

    def _source_boost(self, chunk: ChunkItem) -> float:
        st = self._source_type(chunk)
        if st == "case_upload":
            return self.case_upload_boost
        if st == "knowledge_base":
            return self.knowledge_boost
        return 0.0

    def _rerank_results(self, found: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        reranked = []
        for item in found:
            cloned = dict(item)
            pseudo = ChunkItem(
                chunk_id=str(cloned.get("chunk_id", "")),
                filename=str(cloned.get("filename", "")),
                text=str(cloned.get("excerpt", "")),
                metadata=cloned.get("metadata") or {},
            )
            cloned["score"] = round(float(cloned.get("score") or 0.0) + self._source_boost(pseudo), 4)
            reranked.append(cloned)
        return sorted(reranked, key=lambda i: i.get("score", 0), reverse=True)[:top_k]

    def _fingerprint_documents(self, documents: list[dict[str, Any]]) -> str:
        payload = []
        for doc in documents or []:
            parsed = doc.get("parsed") or {}
            payload.append(
                {
                    "id": doc.get("id"),
                    "filename": doc.get("filename"),
                    "content_type": doc.get("content_type"),
                    "text_len": len(parsed.get("text") or ""),
                    "path": str(doc.get("path", "")),
                    "source": doc.get("source"),
                }
            )
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

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
                    "excerpt": row.text[:2200],
                    "metadata": row.meta or {},
                }
            )
        return out
