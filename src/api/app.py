"""
Main FastAPI application entry point for Docker/production deployment.

This module re-exports the FastAPI app instance from api_server.py
to provide a consistent entry point for ASGI servers like uvicorn.
"""

from src.api.api_server import app

__all__ = ["app"]
