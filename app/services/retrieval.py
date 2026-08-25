from __future__ import annotations

import hashlib
import json
import math
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
    Retrieval documental agnóstico por case.

    Princípios:
    - O Nexus não precisa conhecer domínio, agente, seguro, contrato, RH, jurídico etc.
    - A unidade básica continua sendo chunk semântico.
    - Para documentos grandes, preserva cobertura espacial (início/meio/fim/regiões).
    - Para múltiplos documentos, preserva cobertura entre arquivos.
    - Depois da cobertura, completa com os chunks semanticamente mais relevantes.
    - Busca focada continua privilegiando relevância; busca ampla não deixa regiões
      ou documentos desaparecerem por causa de TOP-K global.
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

        # Controles genéricos de cobertura.
        self.long_document_chunk_threshold = int(os.getenv("GABBI_LONG_DOCUMENT_CHUNK_THRESHOLD", "10"))
        self.long_document_regions = int(os.getenv("GABBI_LONG_DOCUMENT_REGIONS", "6"))
        self.max_coverage_regions_per_document = int(os.getenv("GABBI_MAX_COVERAGE_REGIONS_PER_DOCUMENT", "8"))
        self.coverage_semantic_extra = int(os.getenv("GABBI_COVERAGE_SEMANTIC_EXTRA", "8"))
        self.focused_semantic_min_score = float(os.getenv("GABBI_FOCUSED_SEMANTIC_MIN_SCORE", "0.16"))
        self.focused_semantic_margin = float(os.getenv("GABBI_FOCUSED_SEMANTIC_MARGIN", "0.07"))

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
            "long_document_chunk_threshold": self.long_document_chunk_threshold,
            "long_document_regions": self.long_document_regions,
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
            return {
                "published": False,
                "backend": self.vector_backend,
                "chunks": 0,
                "warning": "Nenhum texto encontrado para indexação.",
            }

        vectorizer = TfidfVectorizer(
            stop_words=None,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w[\w:/\-\.]+\b",
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(texts)

        self.case_vectorizers[case_id] = vectorizer
        self.case_matrices[case_id] = matrix

        info = {
            "published": False,
            "backend": self.vector_backend,
            "chunks": len(chunks),
            "matrix_shape": list(matrix.shape),
            "documents": len({
                str((c.metadata or {}).get("document_id") or c.filename)
                for c in chunks
            }),
        }

        if self.vector_backend == "pgvector" and self._table is not None and self.openai_api_key:
            try:
                self._publish_pgvector(case_id, chunks)
                info.update({"published": True, "backend": "pgvector"})
            except Exception as exc:
                info.update({"published": False, "backend": "pgvector", "error": str(exc)})
        return info

    # ------------------------------------------------------------------
    # APIs públicas de busca
    # ------------------------------------------------------------------

    def search(self, case_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Busca focada tradicional."""
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

    def search_adaptive(
        self,
        case_id: str,
        query: str,
        *,
        top_k: int = 16,
        prefer_coverage: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Busca documental agnóstica.

        Decide estruturalmente entre:
        - focused: pergunta muito específica -> semântico;
        - coverage: case com múltiplos arquivos ou documento longo -> cobertura
          por arquivo/região + complemento semântico.

        `prefer_coverage=True` pode ser usado pelo orquestrador quando a chamada
        já é sabidamente documental/ampla. Mesmo nesse caso, os melhores chunks
        semânticos continuam presentes.
        """
        chunks = self.case_chunks.get(case_id, [])
        if not chunks:
            return []

        semantic_scores, final_scores = self._score_components(case_id, query)
        if semantic_scores is None or final_scores is None:
            return []

        profile = self.coverage_profile(case_id, query, semantic_scores=semantic_scores)
        use_coverage = bool(prefer_coverage) if prefer_coverage is not None else bool(profile["recommended"])

        if not use_coverage:
            order = np.argsort(-final_scores)[:max(1, int(top_k))]
            return [
                self._chunk_to_result(
                    chunks[int(idx)],
                    float(final_scores[int(idx)]),
                    coverage_role="semantic",
                )
                for idx in order
            ]

        return self._search_with_structural_coverage(
            case_id=case_id,
            query=query,
            top_k=top_k,
            semantic_scores=semantic_scores,
            final_scores=final_scores,
        )

    def search_with_document_coverage(
        self,
        case_id: str,
        query: str,
        *,
        top_k: int = 16,
        per_document: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Compatibilidade com V4.

        Agora usa cobertura estrutural completa, incluindo regiões internas
        de um único PDF grande. `per_document` é preservado como piso lógico.
        """
        semantic_scores, final_scores = self._score_components(case_id, query)
        if semantic_scores is None or final_scores is None:
            return []
        return self._search_with_structural_coverage(
            case_id=case_id,
            query=query,
            top_k=max(top_k, per_document),
            semantic_scores=semantic_scores,
            final_scores=final_scores,
            per_document_floor=max(1, int(per_document or 1)),
        )

    def coverage_profile(
        self,
        case_id: str,
        query: str,
        *,
        semantic_scores: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Perfil estrutural/semântico do case, sem conhecimento de domínio."""
        chunks = self.case_chunks.get(case_id, [])
        if not chunks:
            return {
                "recommended": False,
                "document_count": 0,
                "chunk_count": 0,
                "long_document_count": 0,
            }

        if semantic_scores is None:
            semantic_scores, _ = self._score_components(case_id, query)
        if semantic_scores is None:
            semantic_scores = np.zeros(len(chunks), dtype=float)

        by_document = self._group_chunk_indexes_by_document(chunks)
        long_docs = sum(
            1 for indexes in by_document.values()
            if len(indexes) >= self.long_document_chunk_threshold
        )

        ranked = np.sort(np.asarray(semantic_scores, dtype=float))[::-1]
        top1 = float(ranked[0]) if len(ranked) else 0.0
        top2 = float(ranked[1]) if len(ranked) > 1 else 0.0
        margin = top1 - top2

        # Pergunta muito específica tende a ter um pico semântico claro.
        strongly_focused = (
            top1 >= self.focused_semantic_min_score
            and margin >= self.focused_semantic_margin
        )

        # Cobertura é estruturalmente necessária quando:
        # - há mais de um documento; ou
        # - há documento suficientemente longo;
        # exceto quando a consulta tem sinal muito forte de foco pontual.
        structurally_wide = len(by_document) > 1 or long_docs > 0
        recommended = bool(structurally_wide and not strongly_focused)

        return {
            "recommended": recommended,
            "strongly_focused": strongly_focused,
            "document_count": len(by_document),
            "chunk_count": len(chunks),
            "long_document_count": long_docs,
            "top_semantic_score": round(top1, 4),
            "semantic_margin": round(margin, 4),
        }

    def get_case_chunks(self, case_id: str) -> list[ChunkItem]:
        return self.case_chunks.get(case_id, [])

    def clear_case_index(self, case_id: str) -> None:
        self.case_chunks.pop(case_id, None)
        self.case_matrices.pop(case_id, None)
        self.case_vectorizers.pop(case_id, None)
        self.case_document_fingerprints.pop(case_id, None)

    # ------------------------------------------------------------------
    # Cobertura estrutural
    # ------------------------------------------------------------------

    def _search_with_structural_coverage(
        self,
        *,
        case_id: str,
        query: str,
        top_k: int,
        semantic_scores: np.ndarray,
        final_scores: np.ndarray,
        per_document_floor: int = 1,
    ) -> list[dict[str, Any]]:
        chunks = self.case_chunks.get(case_id, [])
        if not chunks:
            return []

        by_document = self._group_chunk_indexes_by_document(chunks)
        selected: list[int] = []
        selected_set: set[int] = set()
        role_by_idx: dict[int, str] = {}

        # 1) Cada documento recebe pelo menos o melhor chunk semântico.
        for _, indexes in self._ordered_documents(by_document, semantic_scores):
            ranked = sorted(indexes, key=lambda idx: float(final_scores[idx]), reverse=True)
            for idx in ranked[:max(1, per_document_floor)]:
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                    role_by_idx[idx] = "document_semantic_anchor"

        # 2) Documento longo recebe âncoras posicionais distribuídas.
        # Isso resolve o caso "um único PDF contendo vários documentos/seções".
        for _, indexes in self._ordered_documents(by_document, semantic_scores):
            if len(indexes) < self.long_document_chunk_threshold:
                continue

            region_count = min(
                self.max_coverage_regions_per_document,
                self.long_document_regions,
                len(indexes),
            )
            for idx in self._pick_region_representatives(
                indexes=indexes,
                semantic_scores=semantic_scores,
                region_count=region_count,
            ):
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                    role_by_idx[idx] = "positional_coverage"

        # 3) Quantidade desejada: cobertura + espaço para relevância semântica.
        coverage_count = len(selected)
        desired_total = max(
            int(top_k or 0),
            coverage_count + self.coverage_semantic_extra,
        )

        # 4) Completa com ranking semântico global.
        for idx in np.argsort(-final_scores):
            idx = int(idx)
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            role_by_idx[idx] = "semantic"
            if len(selected) >= desired_total:
                break

        # 5) Ordenação final preserva cobertura primeiro e semântica depois.
        results = []
        for idx in selected:
            results.append(
                self._chunk_to_result(
                    chunks[idx],
                    float(final_scores[idx]),
                    coverage_role=role_by_idx.get(idx, "semantic"),
                )
            )
        return results

    def _group_chunk_indexes_by_document(self, chunks: list[ChunkItem]) -> dict[str, list[int]]:
        by_document: dict[str, list[int]] = {}
        for idx, chunk in enumerate(chunks):
            metadata = chunk.metadata or {}
            key = str(metadata.get("document_id") or chunk.filename or f"document_{idx}")
            by_document.setdefault(key, []).append(idx)

        # Ordem interna pela posição real do chunk no documento.
        for key, indexes in by_document.items():
            indexes.sort(key=lambda i: int((chunks[i].metadata or {}).get("chunk_index") or i))
        return by_document

    def _ordered_documents(
        self,
        by_document: dict[str, list[int]],
        semantic_scores: np.ndarray,
    ) -> list[tuple[str, list[int]]]:
        ranked: list[tuple[float, str, list[int]]] = []
        for key, indexes in by_document.items():
            best = max((float(semantic_scores[i]) for i in indexes), default=0.0)
            ranked.append((best, key, indexes))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [(key, indexes) for _, key, indexes in ranked]

    def _pick_region_representatives(
        self,
        *,
        indexes: list[int],
        semantic_scores: np.ndarray,
        region_count: int,
    ) -> list[int]:
        """Divide o documento em regiões e pega o melhor chunk de cada região."""
        if not indexes:
            return []
        region_count = max(1, min(int(region_count), len(indexes)))
        edges = np.linspace(0, len(indexes), region_count + 1, dtype=int)

        picked: list[int] = []
        for region_idx in range(region_count):
            start = int(edges[region_idx])
            end = int(edges[region_idx + 1])
            region = indexes[start:end]
            if not region:
                continue
            best = max(region, key=lambda idx: float(semantic_scores[idx]))
            picked.append(best)
        return picked

    # ------------------------------------------------------------------
    # Scoring / index local
    # ------------------------------------------------------------------

    def _score_components(self, case_id: str, query: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        chunks = self.case_chunks.get(case_id, [])
        matrix = self.case_matrices.get(case_id)
        vectorizer = self.case_vectorizers.get(case_id)

        if not chunks or matrix is None or vectorizer is None:
            return None, None

        try:
            query_vec = vectorizer.transform([query])
            if matrix.shape[1] != query_vec.shape[1]:
                return None, None
            semantic_scores = (matrix @ query_vec.T).toarray().ravel().astype(float)
        except Exception:
            return None, None

        final_scores = np.array([
            float(score) + self._source_boost(chunks[idx])
            for idx, score in enumerate(semantic_scores)
        ])

        q = str(query or "").lower()
        wants_file = any(
            token in q
            for token in ["arquivo", "anexo", "documento", "upload", "pdf", "docx", "txt", "csv", "xlsx", "xls"]
        )
        if wants_file:
            for idx, chunk in enumerate(chunks):
                if self._source_type(chunk) == "case_upload":
                    final_scores[idx] += 1.0

        return semantic_scores, final_scores

    def _search_local(self, case_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        chunks = self.case_chunks.get(case_id, [])
        _, scores = self._score_components(case_id, query)
        if not chunks or scores is None:
            return []

        order = np.argsort(-scores)[:top_k]
        return [
            self._chunk_to_result(
                chunks[int(idx)],
                float(scores[int(idx)]),
                coverage_role="semantic",
            )
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
            parts = self._chunk_text_with_positions(text)
            chunk_count = len(parts)

            for i, part in enumerate(parts, start=1):
                chunk_text, char_start, char_end = part
                ratio = 0.0 if len(text) <= 1 else min(1.0, max(0.0, char_start / max(1, len(text) - 1)))

                chunks.append(
                    ChunkItem(
                        chunk_id=f"{doc.get('id', 'doc')}_{i}",
                        filename=str(doc.get("filename") or "arquivo"),
                        text=chunk_text,
                        metadata={
                            "document_id": doc.get("id"),
                            "source_path": str(doc.get("path", "")),
                            "content_type": doc.get("content_type"),
                            "source": doc.get("source") or source_type,
                            "source_type": source_type,
                            "external_id": doc.get("external_id"),
                            "topic_id": doc.get("topic_id") or (doc.get("metadata") or {}).get("topic_id"),
                            "chunk_index": i - 1,
                            "chunk_count": chunk_count,
                            "char_start": char_start,
                            "char_end": char_end,
                            "document_text_length": len(text),
                            "position_ratio": round(ratio, 6),
                            "coverage_bucket": self._coverage_bucket(ratio),
                        },
                    )
                )
        return chunks

    def _chunk_text_with_positions(self, text: str) -> list[tuple[str, int, int]]:
        text = " ".join(str(text or "").split())
        if not text:
            return []
        if len(text) <= self.max_chunk_size:
            return [(text, 0, len(text))]

        out: list[tuple[str, int, int]] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            out.append((text[start:end], start, end))
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return out

    def _chunk_text(self, text: str) -> list[str]:
        """Compatibilidade com chamadas antigas."""
        return [part for part, _, _ in self._chunk_text_with_positions(text)]

    def _coverage_bucket(self, ratio: float) -> str:
        # Bucket estrutural genérico; não representa domínio ou seção semântica.
        bucket_count = max(1, self.long_document_regions)
        idx = min(bucket_count - 1, int(max(0.0, min(0.999999, ratio)) * bucket_count))
        return f"region_{idx + 1:02d}_of_{bucket_count:02d}"

    def _chunk_to_result(
        self,
        chunk: ChunkItem,
        score: float,
        *,
        coverage_role: str = "semantic",
    ) -> dict[str, Any]:
        metadata = dict(chunk.metadata or {})
        metadata["coverage_role"] = coverage_role
        return {
            "filename": chunk.filename,
            "chunk_id": chunk.chunk_id,
            "type": "text",
            "score": round(float(score), 4),
            "excerpt": chunk.text[:2200],
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Tipos de fonte / fingerprint
    # ------------------------------------------------------------------

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
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------
    # PgVector
    # ------------------------------------------------------------------

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
            out.append({
                "filename": row.filename,
                "chunk_id": row.chunk_id,
                "type": "text",
                "score": round(score, 4),
                "excerpt": row.text[:2200],
                "metadata": row.meta or {},
            })
        return out
