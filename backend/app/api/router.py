"""
Main API router module for the Mini RAG application.

This module configures the FastAPI router and includes all API sub-routers
for different functional areas of the application.
"""

from fastapi import APIRouter
from app.api import chat, documents, health

# Create main router
router = APIRouter()

# Include sub-routers with appropriate prefixes and tags
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(documents.router, prefix="/documents", tags=["documents"])
router.include_router(health.router, prefix="/health", tags=["health"])