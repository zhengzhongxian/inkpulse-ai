import os
import yaml
from pathlib import Path
from pydantic import BaseModel

class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

class DatabaseConfig(BaseModel):
    url: str

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    password: str = ""

class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    vision_model: str = "moondream"
    embedding_model: str = "nomic-embed-text"

class CachingConfig(BaseModel):
    prompt_ttl: int = 86400
    vision_ttl: int = 43200
    embeddings_ttl: int = 604800
    prompt_prefix: str = "ai:prompt:"
    vision_prefix: str = "ai:vision:image:"
    embeddings_prefix: str = "ai:embeddings:"
    embeddings_cache_name: str = "ollama_embeddings_cache"

class Config(BaseModel):
    app: AppConfig
    database: DatabaseConfig
    redis: RedisConfig
    ollama: OllamaConfig
    caching: CachingConfig

def load_config() -> Config:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    # Load YAML defaults
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
        
    # Override from environment variables if present
    # We map environment variables to the nested keys
    
    # Database
    db_url = os.getenv("DB_URL")
    if db_url:
        yaml_data["database"]["url"] = db_url
        
    # Redis
    redis_host = os.getenv("REDIS_HOST")
    if redis_host:
        yaml_data["redis"]["host"] = redis_host
    redis_port = os.getenv("REDIS_PORT")
    if redis_port:
        yaml_data["redis"]["port"] = int(redis_port)
    redis_pass = os.getenv("REDIS_PASSWORD")
    if redis_pass is not None:
        yaml_data["redis"]["password"] = redis_pass
        
    # Ollama
    ollama_host = os.getenv("OLLAMA_HOST")
    if ollama_host:
        yaml_data["ollama"]["host"] = ollama_host
    ollama_vision_model = os.getenv("OLLAMA_VISION_MODEL")
    if ollama_vision_model:
        yaml_data["ollama"]["vision_model"] = ollama_vision_model
    ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
    if ollama_embedding_model:
        yaml_data["ollama"]["embedding_model"] = ollama_embedding_model
        
    # Caching TTLs
    prompt_ttl = os.getenv("AI_PROMPT_CACHE_TTL")
    if prompt_ttl:
        yaml_data["caching"]["prompt_ttl"] = int(prompt_ttl)
    vision_ttl = os.getenv("AI_VISION_CACHE_TTL")
    if vision_ttl:
        yaml_data["caching"]["vision_ttl"] = int(vision_ttl)
    embeddings_ttl = os.getenv("AI_EMBEDDINGS_CACHE_TTL")
    if embeddings_ttl:
        yaml_data["caching"]["embeddings_ttl"] = int(embeddings_ttl)
        
    # Caching Prefixes & Name
    prompt_prefix = os.getenv("AI_PROMPT_CACHE_PREFIX")
    if prompt_prefix:
        yaml_data["caching"]["prompt_prefix"] = prompt_prefix
    vision_prefix = os.getenv("AI_VISION_CACHE_PREFIX")
    if vision_prefix:
        yaml_data["caching"]["vision_prefix"] = vision_prefix
    embeddings_prefix = os.getenv("AI_EMBEDDINGS_CACHE_PREFIX")
    if embeddings_prefix:
        yaml_data["caching"]["embeddings_prefix"] = embeddings_prefix
    embeddings_cache_name = os.getenv("AI_EMBEDDINGS_CACHE_NAME")
    if embeddings_cache_name:
        yaml_data["caching"]["embeddings_cache_name"] = embeddings_cache_name
        
    return Config(**yaml_data)

settings = load_config()
