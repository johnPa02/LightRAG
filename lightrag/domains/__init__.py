from .base import DomainConfig, get_prompt
from .business import business_config
from .healthcare import healthcare_config

__all__ = [
    "DomainConfig",
    "get_prompt",
    "business_config",
    "healthcare_config",
]
