from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.repositories.gabbi_postgres_repository import GabbiArticleRecord, GabbiPostgresRepository


class GabbiChatIngestionService:
    """Transforma dados do Gabbi em documentos lógicos compatíveis com o fluxo atual do Nexus."""

    def __init__(self, repository: GabbiPostgresRepository):
        self.repository = repository

    def get_chat_metadata(self, conversation_id: str) -> dict:
        chat = self.repository.get_chat_by_conversation_id(conversation_id)
        if not chat:
            return {
                "found": False,
                "conversation_id": conversation_id,
            }

        return {
            "found": True,
            "chat_id": chat.id,
            "session_id": chat.session_id,
            "conversation_id": chat.conversation_id,
            "created_on": chat.created_on.isoformat() if chat.created_on else None,
            "updated_on": chat.updated_on.isoformat() if chat.updated_on else None,
        }

    def build_documents_from_articles(
        self,
        *,
        conversation_id: str,
        topic_id: int | None = None,
        limit: int = 100,
        updated_after: datetime | None = None,
    ) -> list[dict]:
        articles = self.repository.list_articles_for_ingestion(
            topic_id=topic_id,
            limit=limit,
            updated_after=updated_after,
        )

        return [
            self._article_to_document(
                article=article,
                conversation_id=conversation_id,
                topic_filter=topic_id,
            )
            for article in articles
        ]

    def _article_to_document(
        self,
        *,
        article: GabbiArticleRecord,
        conversation_id: str,
        topic_filter: int | None,
    ) -> dict:
        text = self._build_article_text(article)

        return {
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
                "source_system": "gabbi",
                "source_table": "Article",
                "article_id": article.id,
                "ref_id": article.ref_id,
                "topic_id": article.topic_id,
                "topic_filter": topic_filter,
                "conversation_id": conversation_id,
                "counter": article.counter,
                "published": article.published,
                "created_on": article.created_on.isoformat() if article.created_on else None,
                "updated_on": article.updated_on.isoformat() if article.updated_on else None,
                "created_by": article.created_by,
                "updated_by": article.updated_by,
            },
        }

    @staticmethod
    def _build_article_text(article: GabbiArticleRecord) -> str:
        parts: list[str] = []

        if article.topic_id is not None:
            parts.append(f"Tópico/Área de conhecimento: {article.topic_id}")

        if article.ref_id is not None:
            parts.append(f"Referência do artigo: {article.ref_id}")

        parts.append("Conteúdo do artigo:")
        parts.append(article.article)

        if article.document:
            parts.append("Documento complementar:")
            parts.append(str(article.document))

        return "\n\n".join(parts)
