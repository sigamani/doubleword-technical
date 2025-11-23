#!/usr/bin/env python3
"""
Simple API wrapper for batch inference server
Provides REST endpoints for testing SLA functionality
"""
import os
import sys
import time
import logging
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

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

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "healthy", "service": "batch-inference-api"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="batch-inference-api",
        cluster_nodes=1,
        active_batches=0
    )

@app.post("/generate_batch", response_model=BatchResponse)
async def generate_batch(request: BatchRequest):
    """Batch inference endpoint with SLA monitoring"""
    try:
        logger.info(f"Received batch request: {len(request.prompts)} prompts")
        
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
        # Simple metrics for demo
        metrics_data = """# HELP batch_inference_requests_total
# TYPE batch_inference_requests_total counter
batch_inference_requests_total{status="completed"} 100

# HELP batch_inference_duration_seconds  
# TYPE batch_inference_duration_seconds histogram
batch_inference_duration_seconds_sum 5.0

# HELP batch_throughput_requests_per_sec
# TYPE batch_throughput_requests_per_sec gauge  
batch_throughput_requests_per_sec 20.0

# HELP batch_progress_percentage
# TYPE batch_progress_percentage gauge
batch_progress_percentage 100.0
"""
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