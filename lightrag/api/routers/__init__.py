"""
This module contains all the routers for the LightRAG API.
"""

from .document_routes import router as document_router
from .query_routes import create_query_routes
from .graph_routes import router as graph_router
from .ollama_api import OllamaAPI

__all__ = ["document_router", "create_query_routes", "graph_router", "OllamaAPI"]
