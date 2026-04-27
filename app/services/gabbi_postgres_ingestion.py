from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.repositories.gabbi_postgres_repository import GabbiPostgresRepository


class GabbiPostgresIngestionService:
    def __init__(self, repository: GabbiPostgresRepository):
        self.repository = repository

    def build_documents_from_articles(
        self,
        *,
        conversation_id: str,
        topic_id: str | None = None,
        limit: int = 100,
        updated_after: datetime | None = None,
    ) -> list[dict]:
        chat = self.repository.get_chat_by_conversation_id(conversation_id)

        articles = self.repository.list_articles_for_ingestion(
            topic_id=topic_id,
            limit=limit,
            updated_after=updated_after,
        )

        documents: list[dict] = []

        for article in articles:
            text = self._build_article_text(article)

            if not text.strip():
                continue

            safe_ref = article.ref_id or article.id

            documents.append(
                {
                    "id": uuid4().hex[:12],
                    "filename": f"gabbi_article_{safe_ref}.txt",
                    "path": None,
                    "content_type": "text/database-record",
                    "source": "gabbi_postgres_article",
                    "external_id": article.id,
                    "external_conversation_id": conversation_id,
                    "external_chat_id": chat.id if chat else None,
                    "external_session_id": chat.session_id if chat else None,
                    "parsed": {
                        "text": text,
                        "tables": [],
                    },
                    "metadata": {
                        "source": "gabbi_postgres_article",
                        "conversation_id": conversation_id,
                        "chat_id": chat.id if chat else None,
                        "session_id": chat.session_id if chat else None,
                        "article_id": article.id,
                        "ref_id": article.ref_id,
                        "topic_id": article.topic_id,
                        "topic_name": article.topic_name,
                        "topic_description": article.topic_description,
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

    def _build_article_text(self, article) -> str:
        parts: list[str] = []

        if article.topic_name:
            parts.append(f"Tópico: {article.topic_name}")

        if article.topic_description:
            parts.append(f"Descrição do tópico: {article.topic_description}")

        if article.ref_id:
            parts.append(f"Referência do artigo: {article.ref_id}")

        parts.append("Conteúdo do artigo:")
        parts.append(article.article or "")

        if article.document:
            parts.append("Documento complementar:")
            parts.append(article.document)

        return "\n\n".join(parts)