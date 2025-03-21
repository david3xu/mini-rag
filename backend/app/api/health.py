"""
Health API module for the Mini RAG application.

This module provides endpoints for monitoring system health and status,
including service availability and resource utilization.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import psutil
import os
from datetime import datetime

from app.services.vectorstore import vector_store
from config import settings

# Create router
router = APIRouter()

# Model definitions
class ServiceStatus(BaseModel):
    """Status of an individual service component."""
    status: str
    details: Optional[str] = None

class ResourceMetrics(BaseModel):
    """System resource utilization metrics."""
    cpu_percent: float
    memory_used_mb: float
    memory_available_mb: float
    memory_percent: float

class HealthResponse(BaseModel):
    """Complete health status response."""
    status: str
    timestamp: str
    services: Dict[str, ServiceStatus]
    resources: ResourceMetrics
    config: Dict[str, Any]

async def get_vector_store_status() -> ServiceStatus:
    """
    Check the status of the vector store service.
    
    Returns:
        Status information for the vector store
    """
    try:
        # Check if vector store is available by performing a simple operation
        vector_store.collection.count()
        count = vector_store.collection.count()
        return ServiceStatus(
            status="ok",
            details=f"Vector store available. Document count: {count}"
        )
    except Exception as e:
        return ServiceStatus(
            status="error",
            details=f"Vector store error: {str(e)}"
        )

async def get_model_status() -> ServiceStatus:
    """
    Check the status of language model service.
    
    Returns:
        Status information for the LLM
    """
    try:
        # Check if model file exists
        if os.path.exists(settings.MODEL_PATH):
            size_mb = os.path.getsize(settings.MODEL_PATH) / (1024 * 1024)
            return ServiceStatus(
                status="ok",
                details=f"Model file available. Size: {size_mb:.1f}MB"
            )
        else:
            return ServiceStatus(
                status="error",
                details=f"Model file not found: {settings.MODEL_PATH}"
            )
    except Exception as e:
        return ServiceStatus(
            status="error",
            details=f"Model error: {str(e)}"
        )

async def get_resource_metrics() -> ResourceMetrics:
    """
    Get current system resource utilization metrics.
    
    Returns:
        CPU and memory utilization metrics
    """
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    return ResourceMetrics(
        cpu_percent=cpu_percent,
        memory_used_mb=memory.used / (1024 * 1024),
        memory_available_mb=memory.available / (1024 * 1024),
        memory_percent=memory.percent
    )

@router.get("", response_model=HealthResponse)
async def get_health():
    """
    Get the current health status of the system.
    
    Returns comprehensive health information about all system components,
    including service availability and resource utilization.
    
    Returns:
        Health status response with detailed component information
    """
    # Check status of individual services
    vector_store_status = await get_vector_store_status()
    model_status = await get_model_status()
    resource_metrics = await get_resource_metrics()
    
    # Determine overall status based on component status
    if vector_store_status.status == "error" or model_status.status == "error":
        overall_status = "error"
    elif resource_metrics.memory_percent > 90 or resource_metrics.cpu_percent > 95:
        overall_status = "warning"
    else:
        overall_status = "ok"
    
    # Filter sensitive configuration values
    filtered_config = {
        "APP_NAME": settings.APP_NAME,
        "MODEL_PATH": settings.MODEL_PATH,
        "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
        "CHUNK_SIZE": settings.CHUNK_SIZE,
        "DEFAULT_BATCH_SIZE": settings.DEFAULT_BATCH_SIZE
    }
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        services={
            "vector_store": vector_store_status,
            "language_model": model_status
        },
        resources=resource_metrics,
        config=filtered_config
    )

@router.get("/readiness", response_model=dict)
async def get_readiness():
    """
    Check if the system is ready to handle requests.
    
    This is a lightweight endpoint suitable for Kubernetes readiness probes.
    
    Returns:
        Simple status response
    """
    try:
        # Check vector store connection
        vector_store.collection.count()
        return {"status": "ready"}
    except:
        return {"status": "not_ready"}

@router.get("/liveness", response_model=dict)
async def get_liveness():
    """
    Check if the system is alive and running.
    
    This is a very lightweight endpoint suitable for Kubernetes liveness probes.
    
    Returns:
        Simple status response
    """
    return {"status": "alive"}