"""
Main application module for the Mini RAG backend.

This module initializes and configures the FastAPI application,
including middleware, routers, and API documentation.
"""

import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import os
from contextlib import asynccontextmanager

from app.api.router import router
from app.api.chat import openai_router
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup: Create necessary directories
    os.makedirs(os.path.join("backend", "data", "uploads"), exist_ok=True)
    os.makedirs(os.path.join("backend", "data", "processed"), exist_ok=True)
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    
    logger.info(f"Starting {settings.APP_NAME} backend service")
    
    # Check environment
    logger.info(f"Running with DEBUG={settings.DEBUG}")
    
    yield
    
    # Shutdown cleanup
    logger.info(f"Shutting down {settings.APP_NAME} backend service")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Retrieval-Augmented Generation system with optimized resource usage",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,  # Disable docs in production
    redoc_url="/redoc" if settings.DEBUG else None,  # Disable redoc in production
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance logging middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to log request processing time.
    
    Args:
        request: Incoming HTTP request
        call_next: Next middleware or route handler
        
    Returns:
        HTTP response
    """

    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        # Log requests that take longer than expected
        if process_time > 1.0:  # Threshold for slow requests
            logger.warning(f"Slow request: {request.method} {request.url.path} took {process_time:.4f}s")
        
        return response
    except Exception as e:
        logger.exception(f"Request error: {str(e)}")
        process_time = time.time() - start_time
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again later."}
        )

# Include main API router
app.include_router(router, prefix=settings.API_PREFIX)

# Include OpenAI-compatible API router (no prefix, matches OpenAI API paths)
app.include_router(openai_router)