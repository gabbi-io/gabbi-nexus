from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.repositories.gabbi_article_repository import GabbiArticleRepository


class GabbiArticleIngestionService:
    def __init__(self, article_repository: GabbiArticleRepository):
        self.article_repository = article_repository

    def build_documents_from_articles(
        self,
        limit: int = 1000,
        updated_after: datetime | None = None,
    ) -> list[dict]:
        articles = self.article_repository.list_published_articles(
            limit=limit,
            updated_after=updated_after,
        )

        documents: list[dict] = []

        for article in articles:
            text = self._build_text(article)

            documents.append(
                {
                    "id": uuid4().hex[:12],
                    "filename": f"gabbi_article_{article.ref_id or article.id}.txt",
                    "path": None,
                    "content_type": "text/database-record",
                    "source": "gabbi_postgres_article",
                    "external_id": article.id,
                    "parsed": {
                        "text": text,
                        "tables": [],
                    },
                    "metadata": {
                        "article_id": article.id,
                        "ref_id": article.ref_id,
                        "topic_id": article.topic_id,
                        "topic_name": article.topic_name,
                        "counter": article.counter,
                        "published": article.published,
                        "created_on": article.created_on.isoformat() if article.created_on else None,
                        "updated_on": article.updated_on.isoformat() if article.updated_on else None,
                        "created_by": article.created_by,
                        "updated_by": article.updated_by,
                    },
                }
            )

        return documents

    def _build_text(self, article) -> str:
        parts = []

        if article.topic_name:
            parts.append(f"Área/Tópico: {article.topic_name}")

        if article.ref_id:
            parts.append(f"Referência: {article.ref_id}")

        parts.append("Conteúdo do artigo:")
        parts.append(article.article)

        if article.document:
            parts.append("\nDocumento complementar:")
            parts.append(str(article.document))

        return "\n\n".join(parts)