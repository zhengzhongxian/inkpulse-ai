import logging
from pathlib import Path
import ollama
from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from src.config import settings

logger = logging.getLogger(__name__)

# Load RedisVL Index Schema from YAML file
schema_path = Path(__file__).parent.parent.parent / "config" / "book_rag_schema.yaml"
schema = IndexSchema.from_yaml(str(schema_path))


class RAGService:
    def __init__(self):
        if settings.redis.password:
            self.redis_url = f"redis://:{settings.redis.password}@{settings.redis.host}:{settings.redis.port}"
        else:
            self.redis_url = f"redis://{settings.redis.host}:{settings.redis.port}"

        self._async_index = None
        self.ollama_client = ollama.AsyncClient(host=settings.ollama.host)

    @property
    def async_index(self):
        if self._async_index is None:
            self._async_index = AsyncSearchIndex(schema, redis_url=self.redis_url)
        return self._async_index

    async def initialize(self):
        try:
            if not await self.async_index.exists():
                logger.info("[RAGService] Creating RedisVL search index...")
                await self.async_index.create(overwrite=True, drop=True)
            else:
                logger.info("[RAGService] RedisVL search index already exists.")
        except Exception as e:
            logger.warning(f"[RAGService] Index init skipped: {e}")

    async def embed_query(self, query: str):
        """Asynchronously generate embedding vector via Ollama AsyncClient (non-blocking)."""
        try:
            logger.info(f"[RAGService] Requesting embedding from Ollama (model={settings.ollama.embedding_model})...")
            response = await self.ollama_client.embeddings(
                model=settings.ollama.embedding_model,
                prompt=query
            )
            emb = response.get("embedding")
            if emb:
                logger.info(f"[RAGService] Embedding received (dim={len(emb)})")
            return emb
        except Exception as e:
            logger.warning(f"[RAGService] Async embedding skipped ({e})")
            return None

    async def retrieve_context_with_docs(self, query: str, top_k: int = 3):
        try:
            query_vector = await self.embed_query(query)
            if not query_vector:
                return "", []

            v_query = VectorQuery(
                vector=query_vector,
                vector_field_name="vector",
                num_results=top_k,
                return_fields=["id", "title", "author", "category", "price", "content"],
                return_score=True
            )

            results = await self.async_index.query(v_query)
            if not results:
                return "", []

            contexts = []
            found_books = []
            for r in results:
                title = r.get("title", "")
                author = r.get("author", "")
                category = r.get("category", "Sách InkPulse")
                price = r.get("price", "")
                content = r.get("content", "")
                
                contexts.append(f"Sách: {title} | Tác giả: {author}\nNội dung: {content}")
                if title:
                    found_books.append({
                        "id": r.get("id", ""),
                        "title": title,
                        "author": author,
                        "category": category,
                        "price": price
                    })

            return "\n\n".join(contexts), found_books
        except Exception as e:
            logger.warning(f"[RAGService] Context retrieval skipped: {e}")
            return "", []

    async def retrieve_context(self, query: str, top_k: int = 3) -> str:
        context, _ = await self.retrieve_context_with_docs(query, top_k)
        return context


rag_service = RAGService()
