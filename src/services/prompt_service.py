import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.models.prompt import Prompt
from src.services.cache_service import cache_service
from src.constants import KeyConstants

logger = logging.getLogger(__name__)

# Fallback prompts if database is empty or not seeded yet
DEFAULT_PROMPTS = {
    "vision_book_analysis": (
        "Analyze the provided image. You must answer using the exact format below. Do not output JSON.\n\n"
        "Instructions:\n"
        "1. First, analyze the image layout and read all visible text in English under \"Thought:\".\n"
        "2. Based on your analysis, determine if this is a book (Book).\n"
        "3. Based on your analysis, determine if there are electronic markings (Marked).\n"
        "4. Summarize your decision in Vietnamese under \"Reason:\".\n\n"
        "Format:\n"
        "Thought: [Detailed analysis of image layout and OCR text in English]\n"
        "Book: [Yes/No]\n"
        "Marked: [Yes/No]\n"
        "Reason: [Short summary in Vietnamese explaining the decision]"
    ),
    "chat_system_prompt": (
        "You are InkPulse AI Assistant, an enthusiastic, knowledgeable, and polite sales & support assistant for InkPulse Bookstore.\n"
        "Always respond to customers in polite, natural, friendly Vietnamese.\n\n"
        "MANDATORY RULES:\n"
        "1. IDENTITY: You are InkPulse AI Assistant. If the user asks who you are, what AI/LLM model you are, or who created you, you MUST state clearly that you are InkPulse AI Assistant for InkPulse Bookstore. NEVER claim or state that you are Claude, Claude 2.0, ChatGPT, OpenAI, Llama, or any other model.\n"
        "2. BOOK CONSULTATION: Enthusiastically use the book information provided in the list below to consult, recommend, and introduce books to customers. When customers ask for new, popular, or recommended books, actively select matching books from the list below.\n"
        "3. NATURAL CONVERSATION: Do NOT mention technical terms like \"provided data\", \"context\", \"data file\", \"input text\". Speak naturally like a professional bookstore consultant.\n"
        "4. SCOPE RESTRICTION: ONLY answer questions related to books, authors, genres, orders, services, or policies of InkPulse Bookstore.\n"
        "5. OUT OF SCOPE QUESTIONS: If the user asks about completely unrelated topics (e.g. sports, weather, coding, cooking, politics...), politely decline and state that you can only assist with information related to InkPulse Bookstore.\n\n"
        "INKPULSE BOOKSTORE BOOK LIST AND INFORMATION:\n"
        "{context}"
    )
}

class PromptService:
    async def get_prompt(self, key: str, db: AsyncSession) -> str:
        # Cache-Aside pattern:
        # 1. Build Redis key using key prefix
        cache_key = f"{KeyConstants.PROMPT_CACHE_PREFIX}{key}"
        
        # 2. Try to fetch from Redis
        cached_prompt = await cache_service.get(cache_key)
        if cached_prompt:
            logger.info(f"Cache HIT for prompt key: {key}")
            return cached_prompt
            
        logger.info(f"Cache MISS for prompt key: {key}. Querying database...")
        
        # 3. Cache miss: Query Postgres
        prompt_content = None
        try:
            stmt = select(Prompt).where(Prompt.key == key)
            result = await db.execute(stmt)
            prompt_record = result.scalars().first()
            if prompt_record:
                prompt_content = prompt_record.content
        except Exception as e:
            logger.error(f"Error querying database for prompt key '{key}': {e}. Using python defaults.")

        # 4. If not found in DB, use python defaults if available
        if not prompt_content:
            if key in DEFAULT_PROMPTS:
                logger.warning(f"Prompt key '{key}' not found in DB. Seeding fallback default prompt...")
                prompt_content = DEFAULT_PROMPTS[key]
                # Auto-seed the database if write succeeds
                try:
                    new_prompt = Prompt(key=key, content=prompt_content, description="Auto-seeded default prompt")
                    db.add(new_prompt)
                    await db.commit()
                except Exception as seed_err:
                    logger.error(f"Failed to auto-seed prompt '{key}' to DB: {seed_err}")
            else:
                raise ValueError(f"Prompt with key '{key}' not found in database or default fallbacks.")
        
        # 5. Save to Redis
        await cache_service.set(cache_key, prompt_content, KeyConstants.PROMPT_CACHE_TTL)
        return prompt_content

prompt_service = PromptService()
