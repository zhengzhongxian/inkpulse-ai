from src.config import settings

class KeyConstants:
    # Cache prefixes loaded from central settings
    PROMPT_CACHE_PREFIX = settings.caching.prompt_prefix
    VISION_CACHE_PREFIX = settings.caching.vision_prefix
    EMBEDDINGS_CACHE_PREFIX = settings.caching.embeddings_prefix
    CHAT_SESSION_PREFIX = "chat_session_"
    SEMANTIC_CACHE_NAME = "llmcache"
    
    # Cache names
    EMBEDDINGS_CACHE_NAME = settings.caching.embeddings_cache_name

    # TTLs in seconds loaded from central settings
    PROMPT_CACHE_TTL = settings.caching.prompt_ttl
    VISION_CACHE_TTL = settings.caching.vision_ttl
    EMBEDDINGS_CACHE_TTL = settings.caching.embeddings_ttl
    # Session constants
    MAX_SESSION_MESSAGES = 10
    CHAT_SESSION_TTL = 3600  # 1 hour

