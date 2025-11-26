"""
API routes with proper authentication and dependency injection
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .core.context import get_context, set_context
from .core.auth import TokenManager, rate_limiter
from .core.config import get_config
from .core.processor import VLLMProcessorFactory
from .core.inference import InferencePipeline
from .core.models import (
    BatchInferenceRequest, BatchInferenceResponse,
    AuthBatchJobRequest, BatchJobResponse,
    InferenceResponse
)

logger = logging.getLogger(__name__)

# Initialize components
config = get_config()
token_manager = TokenManager(os.getenv("JWT_SECRET", "default-secret"))
processor_factory = VLLMProcessorFactory()
inference_pipeline = InferencePipeline()

# Security
security = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token with rate limiting"""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    
    # Verify token
    payload = token_manager.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check rate limits
    user_id = payload.get("user_id", "anonymous")
    tier = payload.get("tier", "basic")
    
    if not rate_limiter.check_rate_limit(user_id, tier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    logger.info(f"Token verified for user {user_id}, tier {tier}")
    return token

# FastAPI app
app = FastAPI(
    title="Ray Data vLLM Batch Inference",
    version="2.0.0",
    description="Production-ready batch inference with authentication"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ray-data-vllm-authenticated",
        "version": "2.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    context = get_context()
    return {
        "status": "healthy",
        "service": "ray-data-vllm-authenticated",
        "processor_initialized": context.processor is not None,
        "config_loaded": bool(context.config),
        "version": "2.0.0"
    }

@app.post("/generate_batch", response_model=BatchInferenceResponse)
async def generate_batch(
    request: BatchInferenceRequest,
    token: str = Depends(verify_token)
):
    """Authenticated batch inference"""
    try:
        context = get_context()
        if not context.processor:
            # Initialize processor on first request
            from .core.config import get_config
            processor_config = get_config()
            context.processor = processor_factory.build_processor(processor_config)
            set_context(context)
        
        # Execute inference
        results, total_time = inference_pipeline.execute_batch(
            request.prompts,
            context.processor
        )
        
        # Create response
        response_results = [
            InferenceResponse(
                text=result.get("response", ""),
                tokens_generated=result.get("tokens", 0),
                inference_time=total_time / len(request.prompts)
            )
            for result in results
        ]
        
        return BatchInferenceResponse(
            results=response_results,
            total_time=total_time,
            total_prompts=len(request.prompts),
            throughput=len(request.prompts) / total_time if total_time > 0 else 0,
        )
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start", response_model=BatchJobResponse)
async def start_batch_job(
    request: AuthBatchJobRequest,
    token: str = Depends(verify_token)
):
    """Start authenticated batch job with SLA tracking"""
    try:
        context = get_context()
        
        # Initialize processor if needed
        if not context.processor:
            from .core.config import get_config
            processor_config = get_config()
            context.processor = processor_factory.build_processor(processor_config)
            set_context(context)
        
        # Execute batch job
        results, total_time = inference_pipeline.execute_batch(
            [f"sample_prompt_{i}" for i in range(request.num_samples)],
            context.processor
        )
        
        return BatchJobResponse(
            job_id=f"job_{int(time.time()) % 10000}",
            status="started",
            message=f"Batch job started with {request.num_samples} samples",
            estimated_completion_hours=total_time / 3600 if total_time > 0 else 0.1
        )
        
    except Exception as e:
        logger.error(f"Batch job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    context = get_context()
    if context.monitor:
        return {
            "timestamp": time.time(),
            "active_jobs": 1,  # Simplified for demo
            "throughput_req_per_sec": context.monitor.metrics.throughput_per_sec(),
            "tokens_per_sec": context.monitor.metrics.tokens_per_sec(),
            "sla_status": "healthy" if context.monitor.check_sla() else "at_risk"
        }
    else:
        return {"status": "no_monitor"}