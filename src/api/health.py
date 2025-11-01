"""
Health Check Endpoints for Container Orchestration

Provides liveness and readiness probes for Kubernetes/Docker orchestration.
Implements comprehensive health checks for all critical dependencies.

Week 9 Day 1: Docker Containerization
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
import psutil
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import asyncio

# Initialize router
router = APIRouter(prefix="/health", tags=["health"])

# Track application startup time
_START_TIME = time.time()


class HealthStatus(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str = "2.15.0"


class ReadinessStatus(BaseModel):
    """Readiness check response model"""
    status: str
    checks: Dict[str, str]
    timestamp: str


class DetailedHealthStatus(BaseModel):
    """Detailed health information"""
    status: str
    version: str
    uptime_seconds: float
    timestamp: str
    database: Optional[Dict[str, Any]] = None
    redis: Optional[Dict[str, Any]] = None
    ml_models: Optional[Dict[str, Any]] = None
    system: Optional[Dict[str, Any]] = None


@router.get("/", response_model=HealthStatus, summary="Liveness Probe")
async def health_check() -> HealthStatus:
    """
    Basic health check endpoint - Liveness probe
    
    Returns 200 OK if the application is running.
    Use this for Kubernetes liveness probes.
    
    Returns:
        HealthStatus: Basic health information
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="2.15.0"
    )


@router.get("/ready", response_model=ReadinessStatus, summary="Readiness Probe")
async def readiness_check() -> ReadinessStatus:
    """
    Readiness check endpoint - Readiness probe
    
    Checks if the application can handle requests by verifying:
    - Database connectivity
    - Redis connectivity
    - ML models loaded
    
    Returns 200 if ready, 503 if not ready.
    Use this for Kubernetes readiness probes.
    
    Returns:
        ReadinessStatus: Readiness status with dependency checks
        
    Raises:
        HTTPException: 503 if any critical dependency is unavailable
    """
    checks = {}
    all_healthy = True
    
    # Check database connection
    try:
        from src.database import get_db_session
        db = next(get_db_session())
        # Try a simple query
        db.execute("SELECT 1")
        checks["database"] = "ok"
        db.close()
    except Exception as e:
        checks["database"] = f"error: {str(e)[:50]}"
        all_healthy = False
    
    # Check Redis connection
    try:
        import redis
        from src.config import settings
        r = redis.from_url(settings.REDIS_URL if hasattr(settings, 'REDIS_URL') else "redis://localhost:6379")
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)[:50]}"
        all_healthy = False
    
    # Check ML models (optional - don't fail if not loaded)
    try:
        # Check if models directory exists and has models
        import os
        models_path = os.getenv("ML_MODEL_PATH", "/app/models")
        if os.path.exists(models_path):
            model_files = os.listdir(models_path)
            if model_files:
                checks["ml_models"] = "ok"
            else:
                checks["ml_models"] = "no models found"
        else:
            checks["ml_models"] = "not configured"
    except Exception as e:
        checks["ml_models"] = f"check failed: {str(e)[:50]}"
    
    if not all_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "checks": checks,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    return ReadinessStatus(
        status="ready",
        checks=checks,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/detailed", response_model=DetailedHealthStatus, summary="Detailed Health")
async def detailed_health() -> DetailedHealthStatus:
    """
    Detailed health check with comprehensive system information
    
    Provides detailed information about:
    - Application version and uptime
    - Database connection pool status
    - Redis connection and memory usage
    - ML models loaded
    - System resources (CPU, memory, disk)
    
    Returns:
        DetailedHealthStatus: Comprehensive health information
    """
    uptime = time.time() - _START_TIME
    
    # Database information
    database_info = {}
    try:
        from src.database import engine, get_db_session
        db = next(get_db_session())
        
        # Get pool status
        pool = engine.pool
        database_info = {
            "connected": True,
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "database": str(engine.url.database)
        }
        db.close()
    except Exception as e:
        database_info = {
            "connected": False,
            "error": str(e)[:100]
        }
    
    # Redis information
    redis_info = {}
    try:
        import redis
        from src.config import settings
        r = redis.from_url(settings.REDIS_URL if hasattr(settings, 'REDIS_URL') else "redis://localhost:6379")
        info = r.info()
        redis_info = {
            "connected": True,
            "version": info.get("redis_version"),
            "memory_used": info.get("used_memory_human"),
            "memory_peak": info.get("used_memory_peak_human"),
            "connected_clients": info.get("connected_clients"),
            "uptime_days": info.get("uptime_in_days")
        }
    except Exception as e:
        redis_info = {
            "connected": False,
            "error": str(e)[:100]
        }
    
    # ML models information
    ml_models_info = {}
    try:
        import os
        import glob
        models_path = os.getenv("ML_MODEL_PATH", "/app/models")
        
        if os.path.exists(models_path):
            model_files = glob.glob(f"{models_path}/*.pkl") + glob.glob(f"{models_path}/*.joblib")
            ml_models_info = {
                "path": models_path,
                "models_found": len(model_files),
                "model_names": [os.path.basename(f) for f in model_files]
            }
        else:
            ml_models_info = {
                "path": models_path,
                "status": "directory not found"
            }
    except Exception as e:
        ml_models_info = {
            "error": str(e)[:100]
        }
    
    # System information
    system_info = {}
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = {
            "cpu_percent": round(cpu_percent, 2),
            "cpu_count": psutil.cpu_count(),
            "memory_percent": round(memory.percent, 2),
            "memory_available_mb": round(memory.available / 1024 / 1024, 2),
            "memory_total_mb": round(memory.total / 1024 / 1024, 2),
            "disk_usage_percent": round(disk.percent, 2),
            "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
            "disk_total_gb": round(disk.total / 1024 / 1024 / 1024, 2)
        }
    except Exception as e:
        system_info = {
            "error": str(e)[:100]
        }
    
    return DetailedHealthStatus(
        status="healthy",
        version="2.15.0",
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.utcnow().isoformat(),
        database=database_info,
        redis=redis_info,
        ml_models=ml_models_info,
        system=system_info
    )


@router.get("/ping", summary="Simple Ping")
async def ping() -> Dict[str, str]:
    """
    Simple ping endpoint
    
    Returns:
        dict: {"ping": "pong"}
    """
    return {"ping": "pong"}


@router.get("/version", summary="Version Information")
async def version_info() -> Dict[str, str]:
    """
    Get application version information
    
    Returns:
        dict: Version, build, and timestamp information
    """
    return {
        "version": "2.15.0",
        "name": "SynFinance",
        "description": "Enterprise Fraud Detection Platform",
        "build_date": "2025-11-01",
        "uptime_seconds": str(round(time.time() - _START_TIME, 2))
    }
