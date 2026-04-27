from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.settings import settings


def normalize_decimal(value):
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


@dataclass(slots=True)
class GabbiArticleRecord:
    id: str
    ref_id: int | float | None
    article: str
    counter: int | float | None
    published: bool | None
    topic_id: int | float | None
    topic_name: str | None
    created_on: datetime | None
    updated_on: datetime | None
    created_by: str | None
    updated_by: str | None
    document: str | None


class GabbiArticleRepository:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or settings.resolved_gabbi_database_url

        self.engine: Engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": 10},
        )

    def test_connection(self) -> dict[str, Any]:
        with self.engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).scalar()

            total_articles = conn.execute(
                text('SELECT COUNT(*) FROM "Article";')
            ).scalar()

        return {
            "status": "ok",
            "host": settings.PG_HOST,
            "port": settings.PG_PORT,
            "database": settings.PG_DB,
            "user": settings.PG_USER,
            "postgres_version": version,
            "total_articles": normalize_decimal(total_articles),
        }

    def list_published_articles(
        self,
        limit: int = 1000,
        updated_after: datetime | None = None,
        topic_id: int | None = None,
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
            LEFT JOIN "Topic" t
                ON t."id" = a."topicId"
            WHERE COALESCE(a."deleted", false) = false
              AND COALESCE(a."published", true) = true
              AND NULLIF(TRIM(a."article"), '') IS NOT NULL
        """

        params: dict[str, Any] = {"limit": limit}

        if updated_after:
            query += ' AND a."updatedOn" >= :updated_after '
            params["updated_after"] = updated_after

        if topic_id is not None:
            query += ' AND a."topicId" = :topic_id '
            params["topic_id"] = topic_id

        query += ' ORDER BY a."updatedOn" DESC NULLS LAST LIMIT :limit '

        with self.engine.connect() as conn:
            rows = conn.execute(text(query), params).mappings().all()

        return [
            GabbiArticleRecord(
                id=str(row["id"]),
                ref_id=normalize_decimal(row.get("ref_id")),
                article=row.get("article") or "",
                counter=normalize_decimal(row.get("counter")),
                published=row.get("published"),
                topic_id=normalize_decimal(row.get("topic_id")),
                topic_name=row.get("topic_name"),
                created_on=row.get("created_on"),
                updated_on=row.get("updated_on"),
                created_by=row.get("created_by"),
                updated_by=row.get("updated_by"),
                document=str(row.get("document")) if row.get("document") is not None else None,
            )
            for row in rows
        ]