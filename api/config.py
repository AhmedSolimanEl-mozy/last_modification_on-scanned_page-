"""
Configuration management using Pydantic Settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "arab_rag_db"
    db_user: str = "arab_rag"
    db_password: str = "arab_rag_pass_2024"
    
    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    device: str = "cpu"
    
    # Groq API
    groq_api_key: str
    groq_model: str = "deepseek-r1-distill-llama-70b"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1000
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True
    
    # Retrieval
    top_k_semantic: int = 10
    similarity_threshold: float = 0.3
    enable_paragraph_expansion: bool = True
    
    class Config:
        env_file = "api/.env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
