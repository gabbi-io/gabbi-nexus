from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GABBI_DATABASE_URL: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()