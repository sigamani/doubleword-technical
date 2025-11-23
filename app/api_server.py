#!/usr/bin/env python3
"""
Simple API wrapper for batch inference server
Provides REST endpoints for testing SLA functionality
"""
import logging
import sys
import time
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

# Import our batch inference server
sys.path.append('/workspace')
from batch_inference_simple import BatchMetrics, SLAMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class BatchRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 100
    temperature: float = 0.7
    batch_size: int = 10

class BatchResponse(BaseModel):
    results: List[Dict]
    total_time: float
    total_prompts: int
    throughput: float

class HealthResponse(BaseModel):
    status: str
    service: str
    cluster_nodes: int
    active_batches: int

# FastAPI app
app = FastAPI(title="Batch Inference API", version="1.0.0")

# Global server instance
batch_server = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "healthy", "service": "batch-inference-api"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        if batch_server:
            return HealthResponse(
                status="healthy",
                service="batch-inference-api",
                cluster_nodes=1,  # Simplified for demo
                active_batches=0
            )
        else:
            return {"status": "initializing", "service": "batch-inference-api"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/generate_batch", response_model=BatchResponse)
async def generate_batch(request: BatchRequest):
    """Batch inference endpoint with SLA monitoring"""
    try:
        logger.info(f"Received batch request: {len(request.prompts)} prompts")
        
        # Create temporary config for this request
        temp_config = {
            "model": {
                "name": "Qwen/Qwen2.5-0.5B-Instruct"
            },
            "inference": {
                "batch_size": request.batch_size,
                "concurrency": 2,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            },
            "data": {
                "input_path": "api_request",
                "output_path": "/tmp/api_output.json",
                "num_samples": len(request.prompts)
            },
            "sla": {
                "target_hours": 24
            }
        }
        
        # Create dataset from prompts
        import ray
        from ray import data
        
        if not ray.is_initialized():
            ray.init(address="auto", _redis_password="ray123")
        
        # Create dataset from request prompts
        ds = data.from_items([{"prompt": prompt} for prompt in request.prompts])
        
        # Initialize metrics
        metrics = BatchMetrics(
            total_requests=len(request.prompts),
            dataset_name="api_request",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            sla_hours=24
        )
        
        sla_monitor = SLAMonitor(metrics, check_interval=10)  # Check every 10 seconds
        
        logger.info(f"Starting batch inference: {metrics.total_requests} requests")
        
        # Simple processing for demo
        start_time = time.time()
        results = []
        
        for i, prompt in enumerate(request.prompts):
            # Simulate processing time
            time.sleep(0.05)  # 50ms per request
            
            result = {
                "text": f"Processed: {prompt[:50]}{'...' if len(prompt) > 50 else ''}",
                "tokens_generated": len(prompt.split()),
                "inference_time": 0.05,
                "worker_id": "api_worker"
            }
            results.append(result)
            
            # Update SLA monitor
            sla_monitor.update_and_check(batch_size=1, tokens=result["tokens_generated"])
        
        total_time = time.time() - start_time
        throughput = len(request.prompts) / total_time if total_time > 0 else 0
        
        logger.info(f"Batch completed in {total_time:.2f}s: {throughput:.2f} req/s")
        
        return BatchResponse(
            results=results,
            total_time=total_time,
            total_prompts=len(request.prompts),
            throughput=throughput
        )
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        metrics_data = generate_latest()
        return Response(metrics_data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main function to start the API server"""
    logger.info("Starting Batch Inference API Server")
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()