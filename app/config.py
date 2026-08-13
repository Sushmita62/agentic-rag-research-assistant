from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    data_dir: Path = Path("data")
    db_path: Path = Path("data/index/app.db")
    faiss_path: Path = Path("data/index/faiss.index")
    bm25_path: Path = Path("data/index/bm25.pkl")
    max_upload_mb: int = 30


settings = Settings()
