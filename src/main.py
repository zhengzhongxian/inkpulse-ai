import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.services.cache_service import cache_service
from src.services.prompt_service import prompt_service
from src.services.ai_service import ai_service
from src.utils import calculate_sha256
from src.constants import KeyConstants, VisionMessageConstants
from src.grpc_server import serve as start_grpc_server

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

grpc_server = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global grpc_server
    # Connect to Redis Stack on startup
    await cache_service.connect()
    
    # Start gRPC Server in the background on port 50051
    grpc_server = await start_grpc_server(port=50051)
    
    yield
    # Clean up connections on shutdown
    if grpc_server:
        logger.info("Stopping gRPC server...")
        await grpc_server.stop(grace=5)
    await cache_service.close()

app = FastAPI(
    title="InkPulse AI Service",
    description="Professional Python AI Vision & RAG Helper Service with Redis caching",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "inkpulse-ai"}

@app.post("/api/v1/vision/analyze", tags=["AI Vision"])
async def analyze_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze if an image is a book and if it contains digital markup/drawings.
    Utilizes Redis for result caching via image SHA-256 hash to optimize performance.
    """
    # 1. Read file bytes and validate
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=VisionMessageConstants.ERROR_IMAGE_EMPTY
        )
        
    # 2. Compute SHA-256 hash of the image
    image_hash = calculate_sha256(image_bytes)
    redis_cache_key = f"{KeyConstants.VISION_CACHE_PREFIX}{image_hash}"
    logger.info(f"Analyze request - File: {file.filename}, Size: {len(image_bytes)} bytes, Hash: {image_hash}")
    
    # 3. Check Cache
    cached_result_str = await cache_service.get(redis_cache_key)
    if cached_result_str:
        logger.info(f"Image analysis cache HIT for hash: {image_hash}")
        try:
            cached_data = json.loads(cached_result_str)
            cached_data["cached"] = True
            return {
                "success": True,
                "message": VisionMessageConstants.SUCCESS_ANALYSIS,
                "data": cached_data
            }
        except Exception as e:
            logger.error(f"Error parsing cached vision JSON: {e}. Re-running analysis...")
            
    # 4. Cache MISS: Retrieve prompt template from PostgreSQL (with Cache-Aside)
    logger.info(f"Image analysis cache MISS for hash: {image_hash}. Querying prompt from DB...")
    try:
        prompt_template = await prompt_service.get_prompt("vision_book_analysis", db)
    except Exception as e:
        logger.error(f"Failed to load prompt template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=VisionMessageConstants.ERROR_PROMPT_NOT_FOUND
        )
        
    # 5. Execute vision analysis using Ollama (moondream)
    try:
        analysis_result = await ai_service.analyze_image_with_vision(image_bytes, prompt_template)
    except Exception as e:
        logger.error(f"Ollama vision execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=VisionMessageConstants.ERROR_OLLAMA_FAILED
        )
        
    # 6. Save results to Redis Cache
    try:
        await cache_service.set(
            redis_cache_key,
            json.dumps(analysis_result),
            KeyConstants.VISION_CACHE_TTL
        )
    except Exception as cache_err:
        logger.error(f"Failed to cache vision result: {cache_err}")
        
    analysis_result["cached"] = False
    
    return {
        "success": True,
        "message": VisionMessageConstants.SUCCESS_ANALYSIS,
        "data": analysis_result
    }

@app.post("/api/v1/prompts/reset-cache", tags=["Admin"])
async def reset_prompt_cache(key: str):
    """
    Invalidates the cached prompt template in Redis.
    Forces cache-aside reload on next request.
    """
    cache_key = f"{KeyConstants.PROMPT_CACHE_PREFIX}{key}"
    deleted = await cache_service.delete(cache_key)
    return {
        "success": True,
        "message": f"Cache for prompt '{key}' reset: {deleted}"
    }


