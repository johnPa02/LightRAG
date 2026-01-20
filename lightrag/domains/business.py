"""
Business Law domain configuration.

This domain uses the default prompts from prompt.py without overrides,
as they were originally designed for business/enterprise law.
"""

from .base import DomainConfig


# Business law uses default prompts - no overrides needed
business_config = DomainConfig(
    name="business",
    # All prompts use defaults from prompt.py
)
