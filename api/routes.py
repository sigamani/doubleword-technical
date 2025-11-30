""" API routes for batch inference service. """

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from config import EnvironmentConfig, ModelConfig, BatchConfig

import logging
import time
import uuid
import json

from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from api.models import BatchRequest, BatchResponse, OpenAIBatchRequest, OpenAIBatchResponse, priorityLevels

logger = logging.getLogger(__name__)

from config import EnvironmentConfig, ModelConfig
from pipeline.ray_batch import RayBatchProcessor

env_config = EnvironmentConfig.from_env()
model_config = ModelConfig.default()
pipeline = RayBatchProcessor(model_config, env_config)

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

from api.job_queue import SimpleQueue
job_queue = SimpleQueue()


from api.worker import BatchWorker
try:
    batch_worker = BatchWorker(job_queue)
    batch_worker.start()
    logger.info("Batch worker started successfully")
except Exception as e:
    logger.error(f"Failed to start worker: {e}")
    import traceback
    traceback.print_exc()



app = FastAPI(
    title="Ray Data vLLM Batch Inference",
    version="1.0.0",
    description="Minimal batch inference PoC"
)


batch_config = BatchConfig()
BATCH_DIR = batch_config.batch_dir
   
def calculate_priority(created_at: float, num_prompts: int, deadline_hours: float = 24.0) -> priorityLevels:
    """
    Calculate the priority of a task based on its creation time, number of prompts, and deadline.
    
    Priority levels:
    - LOW: Normal priority
    - HIGH: Urgent priority
    
    A job gets HIGH priority if:
    - Less than 4 hours remaining until deadline, OR
    - Large job (> 100 prompts) with less than 12 hours remaining
    """
    import time
    current_time = time.time()
    deadline_time = created_at + (deadline_hours * 3600)
    time_remaining = deadline_time - current_time
    if time_remaining < 4 * 3600:
        return priorityLevels.HIGH  
    elif num_prompts > 100 and time_remaining < 12 * 3600:
        return priorityLevels.HIGH
    else:
        return priorityLevels.LOW
    

async def execute_batch_async(prompts: List[str]) -> List[Dict[str, Any]]:
    loop = None
    try:
        import asyncio
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop:
        return await loop.run_in_executor(executor, pipeline.process_batch, prompts)
    else:
        return pipeline.process_batch(prompts)
    

@app.get("/debug/worker")
async def debug_worker():
    """Debug endpoint to check worker status"""
    return {
        "worker_running": batch_worker.running if 'batch_worker' in globals() else False,
        "queue_depth": job_queue.get_depth(),
        "worker_thread_alive": batch_worker.worker_thread.is_alive() if 'batch_worker' in globals() and batch_worker.worker_thread else False
    }

@app.get("/debug/gpu-pools")
async def debug_gpu_pools():
    """Debug endpoint to check GPU pool status"""
    if 'batch_worker' in globals():
        return batch_worker.gpu_scheduler.get_pool_status()
    else:
        return {"error": "Worker not initialized"}

from fastapi import UploadFile, File
import aiofiles
@app.post("/v1/files")
async def upload_file(file: UploadFile = File(...)):
    
    if not file.filename.endswith('.jsonl'):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")
    
    try:
        file_id = str(uuid.uuid4())[:12]
        file_path = os.path.join(BATCH_DIR, f"{file_id}.jsonl")
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        line_count = 0
        async with aiofiles.open(file_path, 'r') as f:
            async for line in f:
                if line.strip():
                    try:
                        json.loads(line)
                        line_count += 1
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=400, detail="Invalid JSONL format")
        
        logger.info(f"Uploaded file {file.filename} as {file_id} with {line_count} lines")
        
        return {
            "id": file_id,
            "object": "file",
            "filename": file.filename,
            "created_at": int(time.time()),
            "purpose": "batch",
            "bytes": len(content),
            "lines": line_count
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")
     
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

@app.get("/v1/batches")
async def list_batches():
    try:
        batches = []
        batch_files = []
        
        if os.path.exists(BATCH_DIR):
            batch_files = [f for f in os.listdir(BATCH_DIR) if f.startswith("job_") and f.endswith(".json")]
        
        for batch_file in batch_files:
            job_path = os.path.join(BATCH_DIR, batch_file)
            try:
                with open(job_path, "r") as f:
                    job = json.load(f)
                    batches.append({
                        "id": job.get("id", "unknown"),
                        "object": "batch",
                        "created_at": job.get("created_at", 0),
                        "completed_at": job.get("completed_at", 0),
                        "status": job.get("status", "unknown"),
                        "total_prompts": job.get("num_prompts", 0),
                        "model": job.get("model", "unknown")
                    })
            except Exception as e:
                logger.error(f"Failed to read batch file {batch_file}: {e}")
                continue
        
        return {"object": "list", "data": batches}
        
    except Exception as e:
        logger.error(f"Failed to list batches: {e}")
        raise HTTPException(status_code=500, detail="Failed to list batches")

@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    job_path = os.path.join(BATCH_DIR, f"job_{batch_id}.json")
    
    try:
        if not os.path.exists(job_path):
            raise HTTPException(status_code=404, detail="Batch not found")
        
        with open(job_path, "r") as f:
            job = json.load(f)
        
        current_status = job.get("status", "unknown")
        
        if current_status in ["running", "completed", "failed", "cancelled"]:
            return {
                "id": job.get("id", batch_id),
                "object": "batch",
                "created_at": job.get("created_at", 0),
                "completed_at": job.get("completed_at", 0),
                "status": current_status,
                "error": f"Cannot cancel batch in '{current_status}' status"
            }
        
        job["status"] = "cancelled"
        job["completed_at"] = time.time()
        
        if 'batch_worker' in globals():        
            logger.info(f"Cancelling batch {batch_id}")
        
        with open(job_path, "w") as f:
            json.dump(job, f, indent=2)
        
        logger.info(f"Batch {batch_id} cancelled successfully")
        
        return {
            "id": job.get("id", batch_id),
            "object": "batch",
            "created_at": job.get("created_at", 0),
            "completed_at": job.get("completed_at"),
            "status": "cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel batch {batch_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel batch")
        
@app.post("/v1/batches", response_model=OpenAIBatchResponse)
async def create_openai_batch(request: OpenAIBatchRequest):
    try:
        # Handle both inline input and file_id
        if request.input_file_id:
            # Load prompts from uploaded file
            file_path = os.path.join(BATCH_DIR, f"{request.input_file_id}.jsonl")
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Input file not found")
            
            prompts = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        prompts.append(data.get("prompt", ""))
        elif request.input:
            prompts = [item.get("prompt", "") for item in request.input]
        else:
            raise HTTPException(status_code=400, detail="Either input or input_file_id must be provided")
        
        created_at = float(time.time())
        batch_id = str(uuid.uuid4())[:12]  
        
        job_data = {
            "id": batch_id,
            "model": request.model,
            "status": "queued",
            "created_at": created_at,
            "num_prompts": len(prompts),
            "input_file": f"{batch_id}_input.jsonl",
            "output_file": f"{batch_id}_output.jsonl",
            "error_file": f"{batch_id}_errors.jsonl"
        }
        
        job_path = os.path.join(BATCH_DIR, f"job_{batch_id}.json")
        with open(job_path, "w") as f:
            json.dump(job_data, f, indent=2)
        
        input_path = os.path.join(BATCH_DIR, f"{batch_id}_input.jsonl")
        with open(input_path, "w") as f:
            for prompt in prompts:
                json.dump({"prompt": prompt}, f)
                f.write("\n")
        
        priority = calculate_priority(created_at, len(prompts))
        logger.info(f"Enqueuing batch {batch_id} with priority {priority} and {len(prompts)} prompts")
        
        job_payload = {
            "job_id": batch_id,
            "input_file": input_path,
            "output_file": os.path.join(BATCH_DIR, f"{batch_id}_output.jsonl"),
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "error_file": os.path.join(BATCH_DIR, f"{batch_id}_errors.jsonl")
        }
        
        msg_id = job_queue.enqueue(job_payload, priority=priority)
        
        return OpenAIBatchResponse(
            id=batch_id,
            object="batch",
            created_at=int(created_at),
            status="queued",
            total_prompts=len(prompts),
            model=request.model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to create batch")

@app.get("/v1/batches/{batch_id}")
async def get_openai_batch(batch_id: str):
    job_path = os.path.join(BATCH_DIR, f"job_{batch_id}.json")
    try:
        with open(job_path, "r") as f:
            job = json.load(f)
        return {
            "id": job["id"],
            "model": job.get("model", "unknown"),
            "created_at": job.get("created_at", 0),
            "completed_at": job.get("completed_at", 0),
            "status": job.get("status", "unknown"),
            "total_prompts": job.get("num_prompts", 0)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Batch not found")


@app.get("/v1/batches/{batch_id}/results")
async def get_openai_batch_results(batch_id: str):
    try:
        output_file = os.path.join(BATCH_DIR, f"{batch_id}_output.jsonl")
        
        if not os.path.exists(output_file):
            raise HTTPException(status_code=404, detail="Batch results not found")
        
        results = []
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        return {"object": "list", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to load batch results")