"""
Domain configuration base class and utilities.

This module provides the DomainConfig dataclass for defining domain-specific
configurations and the get_prompt() utility for retrieving prompts with
domain override support.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class DomainConfig:
    """
    Domain configuration with optional prompt overrides.
    
    Each domain (business, healthcare, etc.) can override specific prompts
    while falling back to the base prompts in prompt.py for non-overridden ones.
    
    Attributes:
        name: Domain identifier (e.g., "business", "healthcare")
        rag_response: Override for RAG response prompt
        keywords_extraction: Override for keywords extraction prompt
        entity_extraction_system: Override for entity extraction system prompt
        entity_extraction_examples: Override for entity extraction examples
        summarize_entity_descriptions: Override for entity description summarization
    
    Example:
        healthcare_config = DomainConfig(
            name="healthcare",
            rag_response=HEALTHCARE_RAG_PROMPT,  # Override this
            # keywords_extraction=None  -> uses default from prompt.py
        )
    """
    name: str
    
    # Prompt overrides - None means use default from prompt.py
    rag_response: Optional[str] = None
    keywords_extraction: Optional[str] = None
    entity_extraction_system: Optional[str] = None
    entity_extraction_examples: Optional[list] = None
    summarize_entity_descriptions: Optional[str] = None
    
    # Add more prompts here as needed without modifying operate.py


def get_prompt(key: str, domain: Optional[DomainConfig] = None) -> str:
    """
    Get prompt from domain config with fallback to base PROMPTS.
    
    This function checks if the domain has an override for the requested
    prompt key. If not, it falls back to the base PROMPTS dictionary.
    
    Args:
        key: Prompt key (e.g., "rag_response", "keywords_extraction")
        domain: Optional domain configuration with overrides
        
    Returns:
        The prompt string from domain override or base PROMPTS
        
    Example:
        # With domain override
        prompt = get_prompt("rag_response", healthcare_config)
        
        # Without domain (uses base PROMPTS)
        prompt = get_prompt("rag_response")
    """
    # Import here to avoid circular imports
    from lightrag.prompt import PROMPTS
    
    if domain is not None:
        # Map prompt keys to DomainConfig attributes
        # Handle different naming conventions
        attr_name = key.replace("_prompt", "").replace("_system", "_system")
        
        # Direct attribute lookup
        override = getattr(domain, attr_name, None)
        if override is not None:
            return override
            
        # Try with common suffixes removed
        for suffix in ["_prompt", "_system_prompt"]:
            if key.endswith(suffix):
                base_key = key[:-len(suffix)]
                override = getattr(domain, base_key, None)
                if override is not None:
                    return override
    
    # Fallback to base PROMPTS
    return PROMPTS.get(key, "")
