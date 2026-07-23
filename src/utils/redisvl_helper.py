import logging
from typing import List, Optional
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize import OllamaTextVectorizer
from src.config import settings
from src.constants import KeyConstants

logger = logging.getLogger(__name__)

class RedisVLOllamaHelper:
    def __init__(self):
        self.vectorizer = None
        self.cache = None
        
    def initialize(self):
        # Build Redis connection URL
        if settings.redis.password:
            redis_url = f"redis://:{settings.redis.password}@{settings.redis.host}:{settings.redis.port}"
        else:
            redis_url = f"redis://{settings.redis.host}:{settings.redis.port}"
            
        try:
            # Initialize Ollama Vectorizer
            # OllamaTextVectorizer connects to Ollama endpoint to generate embeddings
            self.vectorizer = OllamaTextVectorizer(
                model=settings.ollama.embedding_model,
                host=settings.ollama.host
            )
            # Initialize EmbeddingsCache
            self.cache = EmbeddingsCache(
                name=KeyConstants.EMBEDDINGS_CACHE_NAME,
                redis_url=redis_url,
                ttl=KeyConstants.EMBEDDINGS_CACHE_TTL
            )
            logger.info("RedisVL Ollama vectorizer & EmbeddingsCache initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing RedisVL Ollama components: {e}")
            
    def get_embedding(self, text: str) -> Optional[List[float]]:
        if not self.vectorizer or not self.cache:
            self.initialize()
            
        if not self.cache:
            # Fallback to direct vectorizer if cache init failed
            if self.vectorizer:
                try:
                    return self.vectorizer.embed(text)
                except Exception as e:
                    logger.error(f"Vectorizer failed: {e}")
            return None
            
        try:
            # Check cache first
            # EmbeddingsCache stores textual vectors in Redis automatically
            cached_vectors = self.cache.get(content=text, model_name=settings.ollama.embedding_model)
            if cached_vectors:
                logger.info("EmbeddingsCache HIT.")
                # Under the hood, cache.get returns the list or nested list depending on search
                # We return the first item or raw vector
                if isinstance(cached_vectors, list) and len(cached_vectors) > 0:
                    if isinstance(cached_vectors[0], list):
                        return cached_vectors[0]
                    return cached_vectors
                return cached_vectors
                
            logger.info("EmbeddingsCache MISS. Vectorizing via Ollama...")
            # Miss: vectorize
            vector = self.vectorizer.embed(text)
            # Cache it
            self.cache.set(content=text, embedding=vector, model_name=settings.ollama.embedding_model)
            return vector
        except Exception as e:
            logger.error(f"Error in RedisVL embedding cache flow: {e}")
            # Fallback to direct vectorizer
            if self.vectorizer:
                try:
                    return self.vectorizer.embed(text)
                except Exception as embed_err:
                    logger.error(f"Ollama vectorizer fallback failed: {embed_err}")
            return None

redisvl_helper = RedisVLOllamaHelper()
