import json
import logging
import base64
from typing import Dict, Any
from ollama import AsyncClient
from src.config import settings

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        # AsyncClient wraps the Ollama HTTP API
        self.client = AsyncClient(host=settings.ollama.host)

    async def analyze_image_with_vision(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """
        Send image bytes and prompt to Ollama vision model and parse JSON/Text output.
        """
        try:
            logger.info(f"Sending image to Ollama ({settings.ollama.vision_model}) at {settings.ollama.host}...")
            
            # Convert bytes to base64 encoding to send over Ollama chat API safely
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            # Call Ollama async
            response = await self.client.chat(
                model=settings.ollama.vision_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [base64_image]
                }],
                options={
                    'temperature': 0.2,
                    'num_predict': 300
                }
            )
            
            content = response.get('message', {}).get('content', '').strip()
            logger.info(f"Ollama vision raw response: {content}")
            
            # Clean markdown code blocks from VLM output if returned (e.g. ```json ... ```)
            cleaned_content = content
            if cleaned_content.startswith("```"):
                lines = cleaned_content.splitlines()
                if len(lines) > 2:
                    if lines[0].startswith("```json") or lines[0].startswith("```"):
                        lines = lines[1:-1]
                    cleaned_content = "\n".join(lines).strip()
            
            # Sanitize Python style booleans if model outputs capital True/False
            cleaned_content = cleaned_content.replace(": True", ": true").replace(": False", ": false")
            cleaned_content = cleaned_content.replace(":true", ": true").replace(":false", ": false")
            
            # Check if VLM returned a JSON-like structure
            if "{" in cleaned_content and "}" in cleaned_content:
                # Parse JSON response
                try:
                    result = json.loads(cleaned_content)
                    is_book = result.get('is_book', False)
                    is_marked = result.get('is_electronically_marked', False)
                    confidence = result.get('confidence', 1.0)
                    reason = result.get('reason', '')
                    return {
                        "is_book": is_book,
                        "is_electronically_marked": is_marked,
                        "confidence": confidence,
                        "reason": reason
                    }
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse VLM response as JSON. Falling back to regex. Raw content: {cleaned_content}")
                    
                    import re
                    # Extract is_book via regex
                    is_book_match = re.search(r'"is_book"\s*:\s*(true|false|True|False)', cleaned_content, re.IGNORECASE)
                    if is_book_match:
                        is_book = is_book_match.group(1).lower() == "true"
                    else:
                        is_book = "true" in cleaned_content.lower() or "yes" in cleaned_content.lower()
                    
                    # Extract is_electronically_marked
                    is_marked_match = re.search(r'"is_electronically_marked"\s*:\s*(true|false|True|False)', cleaned_content, re.IGNORECASE)
                    if is_marked_match:
                        is_marked = is_marked_match.group(1).lower() == "true"
                    else:
                        is_marked = "marked" in cleaned_content.lower() or "yes" in cleaned_content.lower()
                    
                    # Extract confidence
                    confidence_match = re.search(r'"confidence"\s*:\s*(\d+\.\d+|\d+)', cleaned_content)
                    if confidence_match:
                        confidence = float(confidence_match.group(1))
                    else:
                        confidence = 0.7
                    
                    # Extract reason
                    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned_content)
                    if reason_match:
                        reason = reason_match.group(1)
                    else:
                        reason = cleaned_content
                    
                    return {
                        "is_book": is_book,
                        "is_electronically_marked": is_marked,
                        "confidence": confidence,
                        "reason": reason
                    }
            else:
                # VLM returned plain text / line-by-line format. Parse key-values.
                logger.info("Parsing VLM response as plain text lines.")
                is_book = False
                is_marked = False
                confidence = 0.95
                reason = cleaned_content
                
                for line in cleaned_content.splitlines():
                    line = line.strip()
                    if ":" in line:
                        parts = line.split(":", 1)
                        key = parts[0].strip().lower()
                        val = parts[1].strip()
                        if "book" in key:
                            is_book = "yes" in val.lower() or "true" in val.lower()
                        elif "mark" in key or "edit" in key:
                            is_marked = "yes" in val.lower() or "true" in val.lower()
                        elif "reason" in key:
                            reason = val
                            
                return {
                    "is_book": is_book,
                    "is_electronically_marked": is_marked,
                    "confidence": confidence,
                    "reason": reason
                }
                
        except Exception as e:
            logger.error(f"Error calling Ollama vision: {e}")
            raise Exception(f"Lỗi kết nối hoặc xử lý với mô hình AI Vision: {e}")

ai_service = AIService()
