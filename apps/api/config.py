from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Model Configuration: OpenAI
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002" # text-embedding-3-small
    OPENAI_CHAT_MODEL: str = "gpt-4-turbo-preview"

    # Database Configuration
    database_url: str
    postgres_user: str = 'postgres'
    postgres_password: str = 'postgres123'
    postgres_db: str = 'thesis_agents'
    postgres_host: str = 'localhost'
    postgres_port: int = 5432

    # Application Configuration
    app_name: str = 'Optimization AI Tutor'
    app_version: str = '0.1.0'
    debug: bool = True
    environment: str = 'dev'

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    similarity_threshold: float = 0.7

    # API Configuration
    api_host: str = '0.0.0.0'
    api_port: int = 8000
    cors_origins: List[str] = ['http://localhost:8501', 'http://localhost:3000']

    @property
    def async_database_url(self) -> str:
        return self.database_url.replace('postgresql://', 'postgresql+asyncpg://')

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace('postgresql://', 'postgresql+psycopg2://')

# Global settings
settings = Settings()