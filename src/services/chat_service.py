import json
import logging
from typing import AsyncGenerator, List
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from src.config import settings
from src.constants import KeyConstants
from src.database import AsyncSessionLocal
from src.services.rag_service import rag_service
from src.services.cache_service import cache_service
from src.services.prompt_service import prompt_service

logger = logging.getLogger(__name__)


class ChatAIService:
    def __init__(self):
        logger.info(f"[ChatAIService] Initializing ChatOllama with base_url={settings.ollama.host}, model={settings.ollama.vision_model}")
        self.llm = ChatOllama(
            base_url=settings.ollama.host,
            model=settings.ollama.vision_model,
            temperature=0.2
        )

    def _session_key(self, user_id: str) -> str:
        return f"{KeyConstants.CHAT_SESSION_PREFIX}{user_id}"

    async def _get_session_messages(self, user_id: str) -> List:
        """Get recent session messages from Redis as LangChain message objects."""
        try:
            key = self._session_key(user_id)
            raw = await cache_service.get(key)
            if raw:
                messages_data = json.loads(raw)
                result = []
                for m in messages_data[-KeyConstants.MAX_SESSION_MESSAGES:]:
                    if m.get("role") == "user":
                        result.append(HumanMessage(content=m["content"]))
                    elif m.get("role") == "assistant":
                        result.append(AIMessage(content=m["content"]))
                return result
        except Exception as e:
            logger.warning(f"[ChatAIService] Session load skipped: {e}")
        return []

    async def _save_session(self, user_id: str, user_query: str, response: str):
        """Append user+assistant messages to Redis session."""
        try:
            key = self._session_key(user_id)
            raw = await cache_service.get(key)
            messages = json.loads(raw) if raw else []
            messages.append({"role": "user", "content": user_query})
            messages.append({"role": "assistant", "content": response})
            messages = messages[-KeyConstants.MAX_SESSION_MESSAGES:]
            await cache_service.set(key, json.dumps(messages, ensure_ascii=False), ttl=KeyConstants.CHAT_SESSION_TTL)
        except Exception as e:
            logger.warning(f"[ChatAIService] Session save skipped: {e}")

    async def stream_chat(self, user_id: str, user_query: str) -> AsyncGenerator[str, None]:
        """Stream answer chunk by chunk using LangChain ChatOllama pipeline."""
        # 1. Session History
        logger.info(f"[ChatAIService] Step 1: Loading session history for user={user_id}")
        session_messages = await self._get_session_messages(user_id)

        # 2. Retrieve RAG Context & Book Metadata
        logger.info(f"[ChatAIService] Step 2: Retrieving RAG context for query='{user_query}'...")
        context_str, found_books = await rag_service.retrieve_context_with_docs(user_query, top_k=3)
        logger.info(f"[ChatAIService] Step 2: RAG context retrieved (len={len(context_str)}, books={len(found_books)})")

        # 3. Load System Prompt via Cache-Aside Pattern (PostgreSQL -> Redis -> Fallback)
        system_prompt_str = None
        try:
            async with AsyncSessionLocal() as db:
                system_prompt_str = await prompt_service.get_prompt("chat_system_prompt", db)
        except Exception as e:
            logger.error(f"[ChatAIService] Failed to load chat_system_prompt: {e}")

        if not system_prompt_str:
            system_prompt_str = "Bạn là trợ lý AI của InkPulse.\nDANH SÁCH SÁCH VÀ THÔNG TIN TẠI NHÀ SÁCH INKPULSE:\n{context}"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_str),
            MessagesPlaceholder(variable_name="session"),
            ("user", "{user_prompt}")
        ])

        # 4. Stream LLM response via LangChain pipeline
        logger.info(f"[ChatAIService] Step 4: Invoking LangChain LLM astream...")
        rag_chain = prompt_template | self.llm

        full_response = ""
        first_token = True
        try:
            async for chunk in rag_chain.astream({
                "context": context_str if context_str else "Hiện chưa có thông tin ngữ cảnh.",
                "session": session_messages,
                "user_prompt": user_query
            }):
                content = chunk.content
                if first_token and content:
                    logger.info(f"[ChatAIService] First token streamed from Ollama: '{content[:30]}'")
                    first_token = False
                full_response += content
                yield content
            logger.info(f"[ChatAIService] LLM stream finished (total_len={len(full_response)})")
            
            # Stream structured METADATA JSON payload for UI Product Cards
            if found_books:
                meta_json = json.dumps({"type": "metadata", "products": found_books}, ensure_ascii=False)
                yield f"\n[[METADATA]]{meta_json}"
        except Exception as e:
            logger.error(f"[ChatAIService] Error streaming LLM response: {e}")
            error_msg = "\nRất tiếc, hệ thống gặp sự cố. Vui lòng thử lại sau."
            full_response += error_msg
            yield error_msg

        # 5. Save to Session History
        if full_response.strip():
            await self._save_session(user_id, user_query, full_response)


chat_ai_service = ChatAIService()
