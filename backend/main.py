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
import gc
import psutil
from contextlib import asynccontextmanager

from app.api.router import router
from app.api.chat import openai_router
from app.api.test import router as test_router  # Import test router
from app.services.llm import setup_automatic_cleanup  # Import cleanup functions
from app.services.vectorstore import setup_vectorstore_cleanup
from config import settings, get_path

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
    # Startup: Create necessary directories using proper path resolution
    os.makedirs(get_path("data/uploads"), exist_ok=True)
    os.makedirs(get_path("data/processed"), exist_ok=True)
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    
    # Log system resources
    mem = psutil.virtual_memory()
    logger.info(f"System memory: {mem.total / (1024**3):.2f}GB total, {mem.available / (1024**3):.2f}GB available")
    logger.info(f"CPU cores: {os.cpu_count()}")
    
    logger.info(f"Starting {settings.APP_NAME} backend service")
    
    # Set up automatic cleanup tasks
    setup_automatic_cleanup(app)
    setup_vectorstore_cleanup(app)
    
    # Check environment
    logger.info(f"Running with DEBUG={settings.DEBUG}")
    
    # Initial garbage collection
    gc.collect()
    
    yield
    
    # Shutdown cleanup
    logger.info(f"Shutting down {settings.APP_NAME} backend service")
    
    # Force final garbage collection
    gc.collect()

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

# Add resource monitoring middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to log request processing time and monitor resources.
    
    Args:
        request: Incoming HTTP request
        call_next: Next middleware or route handler
        
    Returns:
        HTTP response
    """
    start_time = time.time()
    
    # Log resource usage before request
    if settings.DEBUG:
        mem_before = psutil.virtual_memory()
        logger.debug(f"Request {request.method} {request.url.path} - Memory before: {mem_before.percent}%")
    
    try:
        # Process the request
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        # Log requests that take longer than expected
        if process_time > 1.0:  # Threshold for slow requests
            logger.warning(f"Slow request: {request.method} {request.url.path} took {process_time:.4f}s")
        else:
            logger.info(f"Request: {request.method} {request.url.path} - {process_time:.4f}s")
        
        # Log resource usage after request for DEBUG mode
        if settings.DEBUG:
            mem_after = psutil.virtual_memory()
            logger.debug(f"Request {request.method} {request.url.path} - Memory after: {mem_after.percent}%")
            
            # Run garbage collection if memory usage is high
            if mem_after.percent > 80:
                logger.warning(f"High memory usage ({mem_after.percent}%) - Running garbage collection")
                gc.collect()
        
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

# Include test router for lightweight testing endpoints
app.include_router(test_router, prefix=f"{settings.API_PREFIX}/test", tags=["test"])

# Add simplified health check endpoint for monitoring
@app.get("/healthz")
async def simple_health_check():
    """Simple health check endpoint for monitoring tools."""
    return {"status": "ok", "timestamp": time.time()}

if __name__ == "__main__":
    # Run the application with optimized settings
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        workers=1  # Single worker for better resource control
    )