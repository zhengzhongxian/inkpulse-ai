import logging
from typing import Optional
import redis.asyncio as aioredis
from src.config import settings

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        # Build Redis connection URL
        if settings.redis.password:
            url = f"redis://:{settings.redis.password}@{settings.redis.host}:{settings.redis.port}"
        else:
            url = f"redis://{settings.redis.host}:{settings.redis.port}"
        self.redis_url = url
        self.client: Optional[aioredis.Redis] = None

    async def connect(self):
        try:
            self.client = aioredis.from_url(
                self.redis_url, 
                encoding="utf-8", 
                decode_responses=True
            )
            await self.client.ping()
            logger.info("Connected to Redis Stack successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis Stack: {e}")
            raise e

    async def close(self):
        if self.client:
            await self.client.close()
            logger.info("Redis Stack connection closed.")

    async def get(self, key: str) -> Optional[str]:
        if not self.client:
            await self.connect()
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error(f"Error getting key {key} from Redis: {e}")
            return None

    async def set(self, key: str, value: str, ttl: int) -> bool:
        if not self.client:
            await self.connect()
        try:
            await self.client.set(key, value, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Error setting key {key} in Redis: {e}")
            return False

    async def delete(self, key: str) -> bool:
        if not self.client:
            await self.connect()
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key} from Redis: {e}")
            return False

cache_service = CacheService()
