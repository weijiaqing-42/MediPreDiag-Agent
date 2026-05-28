from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    llm_model: str = "qwen-plus"
    llm_vision_model: str = "qwen-vl-max"
    llm_fast_model: str = "qwen-turbo"
    embedding_model: str = "text-embedding-v3"
    rerank_model: str = "qwen3-rerank"
    rerank_api_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    amap_api_key: str = ""

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "medagent"
    mysql_password: str = "medagent123"
    mysql_database: str = "mediprediag"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    milvus_host: str = "localhost"
    milvus_port: int = 19530

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    session_ttl: int = 1800
    max_retry: int = 3
    node_timeout: int = 15

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()