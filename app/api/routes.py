import logging
import time
import uuid
import json
import os
from app.core.sla import SLAManager
from app.utils.schemas import JobStore
from app.core.priority import calculate_priority

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.processor import InferencePipeline

logger = logging.getLogger(__name__)
pipeline = InferencePipeline()
executor = ThreadPoolExecutor(max_workers=4)

sla_manager = SLAManager()
job_store = JobStore()

app = FastAPI(
    title="Ray Data vLLM Batch Inference",
    version="1.0.0",
    description="Minimal batch inference PoC"
)

# ---------------------------
# Models
# ---------------------------
class BatchRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 256
    temperature: float = 0.7

class BatchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_time: float
    total_prompts: int
    throughput: float

class OpenAIBatchRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    input: List[Dict[str, str]]
    max_tokens: int = 256
    temperature: float = 0.7

class OpenAIBatchResponse(BaseModel):
    id: str
    object: str = "batch"
    created_at: int
    status: str

# ---------------------------
# Helpers
# ---------------------------
BATCH_DIR = "/tmp"

async def execute_batch_async(prompts: List[str]) -> List[Dict[str, Any]]:
    loop = None
    try:
        import asyncio
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop:
        return await loop.run_in_executor(executor, pipeline.execute_batch, prompts)
    else:
        return pipeline.execute_batch(prompts)

def save_batch(batch_id: str, model: str, results: List[Dict[str, Any]]):
    path = os.path.join(BATCH_DIR, f"batch_{batch_id}.json")
    data = {"id": batch_id, "model": model, "results": results, "created_at": int(time.time())}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

def load_batch(batch_id: str):
    path = os.path.join(BATCH_DIR, f"batch_{batch_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Batch not found")
    with open(path, "r") as f:
        return json.load(f)

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
async def root():
    return {"status": "healthy", "service": "ray-data-vllm-batch-inference", "version": "1.0.0"}

@app.post("/generate_batch", response_model=BatchResponse)
async def generate_batch(request: BatchRequest):
    start_time = time.time()
    try:
        results = await execute_batch_async(request.prompts)
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail="Batch inference failed")

    total_time = time.time() - start_time
    return BatchResponse(
        results=results,
        total_time=total_time,
        total_prompts=len(request.prompts),
        throughput=len(request.prompts)/total_time if total_time > 0 else 0
    )

@app.post("/v1/batches", response_model=OpenAIBatchResponse)
async def create_openai_batch(request: OpenAIBatchRequest):
    prompts = [item.get("prompt", "") for item in request.input]
    created_at = float(time.time())

    job_data = {"id": "placeholder", "model": request.model, "status": "queued", "created_at": created_at, "num_prompts": len(prompts)}
    batch_id = job_store.create(job_data)

    sla_manager.create_job(batch_id, len(prompts))

    priority = calculate_priority(created_at, len(prompts))
    logger.info(f"Enqueuing batch {batch_id} with priority {priority} and {len(prompts)} prompts")
    return OpenAIBatchResponse(
        id=batch_id,
        created_at=int(created_at),
        status="queued"
    )

@app.get("/v1/batches/{batch_id}")
async def get_openai_batch(batch_id: str):
    try:
        job=job_store.get(batch_id)
        return {
            "id": job["id"],
            "model": job.get("model", "unknown"),
            "created_at": job.get("created_at", 0),
            "completed_at": job.get("completed_at", 0),
            "status": job.get("status", "unknown"),
            "total_prompts": job.get("num_prompts", 0)
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Batch not found")


@app.get("/v1/batches/{batch_id}/results")
async def get_openai_batch_results(batch_id: str):
    batch = load_batch(batch_id)
    return {"object": "list", "data": batch["results"]}
