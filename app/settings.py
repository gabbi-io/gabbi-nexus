from __future__ import annotations

from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PG_HOST: str = "192.168.230.108"
    PG_PORT: str = "5432"
    PG_DB: str = "gabbi-io"
    PG_USER: str = "gabbi_io"
    PG_PASSWORD: str = "lrc2An*gvNP%00SkW%bY5cFLQV6S0o5v7^"

    GABBI_DATABASE_URL: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @property
    def resolved_gabbi_database_url(self) -> str:
        if self.GABBI_DATABASE_URL:
            return self.GABBI_DATABASE_URL

        password_encoded = quote_plus(self.PG_PASSWORD)

        return (
            f"postgresql+psycopg://{self.PG_USER}:{password_encoded}"
            f"@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
        )


settings = Settings()