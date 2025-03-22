"""
Data models for the Mini-RAG application.

This module contains Pydantic models for data validation and serialization
across the application.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document with metadata."""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchQuery(BaseModel):
    """Search query model with optional filters."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5

class ChatMessage(BaseModel):
    """Chat message with role and content."""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Chat request with messages and optional settings."""
    messages: List[ChatMessage]
    use_rag: bool = True
    temperature: float = 0.7
    max_tokens: int = 1024 