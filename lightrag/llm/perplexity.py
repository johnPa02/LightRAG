"""
Perplexity API integration for web search enrichment.

This module provides a client to call Perplexity's Sonar API for web search,
which can be used to enrich RAG context with external knowledge.
"""

import os
import aiohttp
from typing import Optional
from lightrag.utils import logger


PERPLEXITY_DEFAULT_MODEL = "sonar"  # or "sonar-pro" for higher quality


async def perplexity_search(
    query: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: str = PERPLEXITY_DEFAULT_MODEL,
    language: str = "vi",
    max_tokens: int = 1024,
    timeout: int = 30,
) -> Optional[str]:
    """
    Call Perplexity Sonar API to get web search results.
    
    Args:
        query: The search query
        api_key: Perplexity API key. If not provided, reads from PPLX_API_KEY env var
        api_url: Perplexity API URL. If not provided, reads from PPLX_API_URL env var
        model: Model to use ("sonar" or "sonar-pro")
        language: Response language preference
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        
    Returns:
        str: The search result content, or None if failed
    """
    api_key = api_key or os.getenv("PPLX_API_KEY")
    api_url = api_url or os.getenv("PPLX_API_URL", "https://api.perplexity.ai")
    
    # Ensure URL ends with /chat/completions
    if not api_url.endswith("/chat/completions"):
        api_url = api_url.rstrip("/") + "/chat/completions"
    
    if not api_key:
        logger.warning("Perplexity API key not configured (PPLX_API_KEY). Skipping web search enrichment.")
        return None
    
    logger.info(f"[Perplexity] Calling API: {api_url}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    
    # Build system prompt for legal/healthcare context
    system_prompt = f"""Bạn là trợ lý tìm kiếm thông tin pháp luật Việt Nam.
Hãy tìm kiếm và tóm tắt thông tin liên quan đến câu hỏi.
Ưu tiên các nguồn chính thống: văn bản pháp luật, cổng thông tin chính phủ, báo chính thống.
Trả lời bằng tiếng Việt, ngắn gọn và có trích dẫn nguồn nếu có."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "return_citations": True,
        "return_related_questions": False,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Perplexity API error: {response.status} - {error_text}")
                    return None
                
                data = await response.json()
                
                # Extract content from response
                choices = data.get("choices", [])
                if not choices:
                    logger.warning("Perplexity returned empty choices")
                    return None
                
                content = choices[0].get("message", {}).get("content", "")
                
                # Extract citations if available
                citations = data.get("citations", [])
                if citations:
                    content += "\n\n**Nguồn tham khảo:**\n"
                    for i, citation in enumerate(citations[:5], 1):  # Limit to 5 citations
                        content += f"[{i}] {citation}\n"
                
                logger.debug(f"Perplexity search completed: {len(content)} chars")
                return content
                
    except aiohttp.ClientError as e:
        logger.error(f"Perplexity API connection error: {e}")
        return None
    except Exception as e:
        logger.error(f"Perplexity search failed: {e}")
        return None


async def enrich_context_with_perplexity(
    query: str,
    rag_context: str,
    api_key: Optional[str] = None,
    model: str = PERPLEXITY_DEFAULT_MODEL,
) -> str:
    """
    Enrich RAG context with Perplexity web search results.
    
    Args:
        query: The original user query
        rag_context: The context retrieved from RAG
        api_key: Optional Perplexity API key
        model: Perplexity model to use
        
    Returns:
        str: Enriched context combining RAG and Perplexity results
    """
    perplexity_result = await perplexity_search(
        query=query,
        api_key=api_key,
        model=model,
    )
    
    if not perplexity_result:
        # If Perplexity fails, return original context
        return rag_context
    
    # Combine Perplexity results with RAG context (Perplexity at TOP to avoid truncation)
    enriched_context = f"""**Thông tin tham khảo từ tìm kiếm web (Perplexity):**

{perplexity_result}

-----

**LƯU Ý QUAN TRỌNG:** Thông tin từ Perplexity chỉ để tham khảo. 
LUÔN ƯU TIÊN thông tin từ văn bản pháp luật trong hệ thống (phần dưới).

-----

{rag_context}"""

    return enriched_context
