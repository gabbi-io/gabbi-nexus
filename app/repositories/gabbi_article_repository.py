from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.settings import settings


@dataclass
class GabbiArticleRecord:
    id: str
    ref_id: int | None
    article: str
    counter: int | None
    published: bool | None
    topic_id: int | None
    topic_name: str | None
    created_on: datetime | None
    updated_on: datetime | None
    created_by: str | None
    updated_by: str | None
    document: str | None


class GabbiArticleRepository:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or settings.GABBI_DATABASE_URL
        if not self.database_url:
            raise RuntimeError("GABBI_DATABASE_URL não configurada no .env")

        self.engine: Engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )

    def list_published_articles(
        self,
        limit: int = 1000,
        updated_after: datetime | None = None,
    ) -> list[GabbiArticleRecord]:
        query = """
            SELECT
                a."id",
                a."refId" AS ref_id,
                a."article",
                a."counter",
                a."published",
                a."topicId" AS topic_id,
                t."name" AS topic_name,
                a."createdOn" AS created_on,
                a."updatedOn" AS updated_on,
                a."createdBy" AS created_by,
                a."updatedBy" AS updated_by,
                a."document"
            FROM "Article" a
            LEFT JOIN "Topic" t ON t."id" = a."topicId"
            WHERE COALESCE(a."deleted", false) = false
              AND COALESCE(a."published", true) = true
              AND NULLIF(TRIM(a."article"), '') IS NOT NULL
        """

        params: dict[str, Any] = {"limit": limit}

        if updated_after:
            query += ' AND a."updatedOn" >= :updated_after '
            params["updated_after"] = updated_after

        query += ' ORDER BY a."updatedOn" DESC NULLS LAST LIMIT :limit '

        with self.engine.connect() as conn:
            rows = conn.execute(text(query), params).mappings().all()

        return [
            GabbiArticleRecord(
                id=str(row["id"]),
                ref_id=row.get("ref_id"),
                article=row.get("article") or "",
                counter=row.get("counter"),
                published=row.get("published"),
                topic_id=row.get("topic_id"),
                topic_name=row.get("topic_name"),
                created_on=row.get("created_on"),
                updated_on=row.get("updated_on"),
                created_by=row.get("created_by"),
                updated_by=row.get("updated_by"),
                document=row.get("document"),
            )
            for row in rows
        ]