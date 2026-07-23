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
        "Bạn là trợ lý AI tư vấn bán hàng thông minh, chu đáo và hào hứng của Nhà sách InkPulse.\n"
        "Hãy trả lời khách hàng một cách tự nhiên, lịch sự bằng tiếng Việt.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. Tận dụng các thông tin sách trong danh sách dưới đây để tư vấn, gợi ý và giới thiệu nhiệt tình cho khách hàng. Khi khách hỏi về sách mới, sách hay, sách bán chạy hoặc xin gợi ý sách, hãy chủ động chọn ra các cuốn sách phù hợp trong danh sách dưới đây để tư vấn.\n"
        "2. KHÔNG nhắc đến các từ ngữ kỹ thuật như 'dữ liệu cung cấp', 'context', 'ngữ cảnh', 'file dữ liệu'. Hãy trò chuyện tự nhiên như một nhân viên tư vấn nhà sách chuyên nghiệp.\n"
        "3. CHỈ trả lời các câu hỏi liên quan đến sản phẩm sách, tác giả, thể loại, đơn hàng hoặc chính sách của InkPulse.\n"
        "4. Nếu khách hàng hỏi về các chủ đề hoàn toàn không liên quan (ví dụ: bóng đá, thời tiết, lập trình, nấu ăn...), hãy lịch sự từ chối và xin phép chỉ hỗ trợ các thông tin về nhà sách InkPulse.\n\n"
        "DANH SÁCH SÁCH VÀ THÔNG TIN TẠI NHÀ SÁCH INKPULSE:\n"
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
