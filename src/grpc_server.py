import asyncio
import logging
import json
import grpc
from proto import vision_service_pb2, vision_service_pb2_grpc
from proto import chat_service_pb2, chat_service_pb2_grpc
from src.database import AsyncSessionLocal
from src.services.cache_service import cache_service
from src.services.prompt_service import prompt_service
from src.services.ai_service import ai_service
from src.services.chat_service import chat_ai_service
from src.utils import calculate_sha256
from src.constants import KeyConstants

logger = logging.getLogger(__name__)

class VisionServiceServicer(vision_service_pb2_grpc.VisionServiceServicer):
    async def AnalyzeImage(self, request, context):
        image_bytes = request.image_data
        file_name = request.file_name
        content_type = request.content_type
        
        if not image_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Image data cannot be empty.")
            return vision_service_pb2.ImageAnalysisResponse()
            
        # Calculate SHA-256 hash of the image
        image_hash = calculate_sha256(image_bytes)
        redis_cache_key = f"{KeyConstants.VISION_CACHE_PREFIX}{image_hash}"
        
        logger.info(f"[gRPC] Analyze request - File: {file_name}, Size: {len(image_bytes)} bytes, Hash: {image_hash}")
        
        # 1. Check Cache
        try:
            cached_result_str = await cache_service.get(redis_cache_key)
            if cached_result_str:
                logger.info(f"[gRPC] Image analysis cache HIT for hash: {image_hash}")
                cached_data = json.loads(cached_result_str)
                return vision_service_pb2.ImageAnalysisResponse(
                    is_book=cached_data.get("is_book", False),
                    is_electronically_marked=cached_data.get("is_electronically_marked", False),
                    confidence=float(cached_data.get("confidence", 1.0)),
                    reason=cached_data.get("reason", ""),
                    cached=True
                )
        except Exception as e:
            logger.error(f"[gRPC] Error checking cache: {e}. Proceeding to analyze...")

        # 2. Cache MISS: Query PostgreSQL for prompt template
        logger.info(f"[gRPC] Image analysis cache MISS for hash: {image_hash}. Querying prompt from DB...")
        try:
            async with AsyncSessionLocal() as db:
                prompt_template = await prompt_service.get_prompt("vision_book_analysis", db)
        except Exception as e:
            logger.error(f"[gRPC] Failed to load prompt template: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Prompt template not found.")
            return vision_service_pb2.ImageAnalysisResponse()

        # 3. Execute vision analysis using Ollama
        try:
            analysis_result = await ai_service.analyze_image_with_vision(image_bytes, prompt_template)
        except Exception as e:
            logger.error(f"[gRPC] Ollama vision execution failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"AI Vision service execution failed: {str(e)}")
            return vision_service_pb2.ImageAnalysisResponse()

        # 4. Save results to Redis Cache
        try:
            await cache_service.set(
                redis_cache_key,
                json.dumps(analysis_result),
                KeyConstants.VISION_CACHE_TTL
            )
        except Exception as cache_err:
            logger.error(f"[gRPC] Failed to cache vision result: {cache_err}")

        return vision_service_pb2.ImageAnalysisResponse(
            is_book=analysis_result.get("is_book", False),
            is_electronically_marked=analysis_result.get("is_electronically_marked", False),
            confidence=float(analysis_result.get("confidence", 1.0)),
            reason=analysis_result.get("reason", ""),
            cached=False
        )

class ChatServiceServicer(chat_service_pb2_grpc.ChatServiceServicer):
    async def ChatStream(self, request, context):
        user_id = request.user_id
        message = request.message

        logger.info(f"[gRPC] ChatStream request - User ID: {user_id}, Message: {message}")

        try:
            async for chunk in chat_ai_service.stream_chat(user_id, message):
                yield chat_service_pb2.ChatResponse(text=chunk, is_end=False)

            # Send final completion signal
            yield chat_service_pb2.ChatResponse(text="", is_end=True)
        except Exception as e:
            logger.error(f"[gRPC] Error streaming chat for user {user_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Chat streaming error: {str(e)}")

async def serve(port: int = 50051):
    server = grpc.aio.server()
    vision_service_pb2_grpc.add_VisionServiceServicer_to_server(
        VisionServiceServicer(), server
    )
    chat_service_pb2_grpc.add_ChatServiceServicer_to_server(
        ChatServiceServicer(), server
    )
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()
    return server

