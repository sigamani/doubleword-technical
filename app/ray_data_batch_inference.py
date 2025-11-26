#!/usr/bin/env python3
"""
Ray Data + vLLM distributed batch inference using official ray.data.llm API
Following: https://docs.ray.io/en/latest/data/batch_inference.html
"""

import os
import time
import sys
import yaml
import logging
import asyncio
import nest_asyncio
import hashlib
import json
from typing import List, Dict, Any
from dataclasses import dataclass

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
import redis
from prometheus_client import (
    Gauge,
    Histogram,
    Counter,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prometheus metrics
inference_requests = Counter(
    "ray_data_requests_total", "Total Ray Data requests", ["method", "status"]
)

inference_duration = Histogram(
    "ray_data_inference_duration_seconds",
    "Ray Data inference duration",
    ["node_id", "model_name"],
)

active_batches = Gauge("ray_data_active_batches", "Active Ray Data batches")

throughput_gauge = Gauge("ray_data_throughput_requests_per_sec", "Current throughput")

# Authentication
security = HTTPBearer(auto_error=False)

# Pydantic models
class BatchInferenceRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = Field(ge=1, le=2048, default=100)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    batch_size: int = Field(ge=1, le=1000, default=128)

class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    inference_time: float

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]
    total_time: float
    total_prompts: int
    throughput: float

class MultimodalInput(BaseModel):
    text: str = Field(default="")
    image_url: str = Field(default="")
    audio_url: str = Field(default="")
    video_url: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AuthBatchJobRequest(BaseModel):
    input_path: str
    output_path: str
    num_samples: int = Field(ge=1, le=100000, default=1000)
    max_tokens: int = Field(ge=1, le=2048, default=512)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    batch_size: int = Field(ge=1, le=1000, default=128)
    concurrency: int = Field(ge=1, le=10, default=2)
    sla_tier: str = Field(default="basic")  # free, basic, premium, enterprise
    model_type: str = Field(default="text")  # text, multimodal
    multimodal_inputs: List[MultimodalInput] = Field(default_factory=list)

class BatchJobRequest(BaseModel):
    input_path: str
    output_path: str
    num_samples: int = Field(ge=1, le=100000, default=1000)
    max_tokens: int = Field(ge=1, le=2048, default=512)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    batch_size: int = Field(ge=1, le=1000, default=128)
    concurrency: int = Field(ge=1, le=10, default=2)

class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_completion_hours: float

@dataclass
class DataArtifact:
    sha256_hash: str
    content_hash: str
    version: str
    created_at: float
    size_bytes: int

@dataclass
class BatchMetrics:
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    start_time: float = 0.0
    tokens_processed: int = 0
    artifact_id: str = ""
    
    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()
    
    def throughput_per_sec(self) -> float:
        elapsed = time.time() - self.start_time
        return self.completed_requests / elapsed if elapsed > 0 else 0
    
    def tokens_per_sec(self) -> float:
        elapsed = time.time() - self.start_time
        return self.tokens_processed / elapsed if elapsed > 0 else 0
    
    def eta_hours(self) -> float:
        throughput = self.throughput_per_sec()
        if throughput == 0:
            return float('inf')
        remaining = self.total_requests - self.completed_requests
        return (remaining / throughput) / 3600
    
    def optimal_batch_size(self, model_size_gb: float, available_memory_gb: float) -> int:
        """Calculate optimal batch size based on model size and available memory"""
        # Rule of thumb: batch_size = floor(available_memory * 0.7 / model_size_gb)
        memory_for_batch = available_memory * 0.7
        optimal_size = int(memory_for_batch / model_size_gb)
        return max(1, min(optimal_size, 512))  # Cap at 512
    
    def tokens_per_second_target(self, target_hours: float = 24.0) -> float:
        """Calculate required tokens/sec to meet SLA target"""
        if self.total_requests == 0:
            return 0
        avg_tokens_per_request = self.tokens_processed / max(1, self.completed_requests)
        required_tokens_per_sec = (self.total_requests * avg_tokens_per_request) / (target_hours * 3600)
        return required_tokens_per_sec
    
    def progress_pct(self) -> float:
        return (self.completed_requests / self.total_requests) * 100

class SLATier:
    FREE = {"name": "free", "hours": 72, "priority": 1}
    BASIC = {"name": "basic", "hours": 24, "priority": 2}
    PREMIUM = {"name": "premium", "hours": 12, "priority": 3}
    ENTERPRISE = {"name": "enterprise", "hours": 6, "priority": 4}

class InferenceMonitor:
    def __init__(self, metrics: BatchMetrics, sla_tier: dict, log_interval: int = 100):
        self.metrics = metrics
        self.sla_tier = sla_tier
        self.sla_hours = sla_tier["hours"]
        self.priority = sla_tier["priority"]
        self.log_interval = log_interval
        self.last_log = 0
    
    def update(self, batch_size: int, tokens: int):
        self.metrics.completed_requests += batch_size
        self.metrics.tokens_processed += tokens
        
        if self.metrics.completed_requests - self.last_log >= self.log_interval:
            self.log_progress()
            self.check_sla()
            self.last_log = self.metrics.completed_requests
    
    def log_progress(self):
        logger.info(
            f"Progress: {self.metrics.progress_pct():.1f}% | "
            f"Completed: {self.metrics.completed_requests}/{self.metrics.total_requests} | "
            f"Throughput: {self.metrics.throughput_per_sec():.2f} req/s | "
            f"Tokens/sec: {self.metrics.tokens_per_sec():.2f} | "
            f"ETA: {self.metrics.eta_hours():.2f}h | "
            f"Failed: {self.metrics.failed_requests}"
        )
    
    def check_sla(self) -> bool:
        eta = self.metrics.eta_hours()
        elapsed_hours = (time.time() - self.metrics.start_time) / 3600
        remaining_hours = self.sla_hours - elapsed_hours
        
        # Add small buffer to avoid false positives when ETA == remaining time
        buffer = 0.1  # 0.1 hour buffer
        if eta > remaining_hours + buffer:
            logger.warning(f"SLA AT RISK! Tier {self.sla_tier['name']} ETA {eta:.2f}h > Remaining {remaining_hours:.2f}h")
            return False
        return True

# Global variables
config: Dict[str, Any] = {}
processor = None
monitor: InferenceMonitor = None
metrics: BatchMetrics = None
redis_client = None

def load_config() -> Dict[str, Any]:
    global config
    config_path = os.environ.get("CONFIG_PATH", "/app/config/config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Use default config
        config = {
            "model": {
                "name": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_model_len": 32768,
                "tensor_parallel_size": 2
            },
            "inference": {
                "batch_size": 128,
                "concurrency": 2,
                "max_num_batched_tokens": 16384,
                "gpu_memory_utilization": 0.90,
                "temperature": 0.7,
                "max_tokens": 512
            },
            "data": {
                "input_path": "/tmp/sharegpt_sample.json",
                "output_path": "/tmp/output.json",
                "num_samples": 1000
            },
            "sla": {
                "target_hours": 24,
                "buffer_factor": 0.7,
                "alert_threshold_hours": 20
            },
            "storage": {
                "local_path": "/tmp/artifacts",
                "s3_bucket": "batch-inference-artifacts"  # Production S3 bucket
            }
        }
        logger.info("Using default configuration")
        return config

def initialize_redis():
    global redis_client
    try:
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", 6379))
        redis_password = os.environ.get("REDIS_PASSWORD", "ray123")
        
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        
        # Test connection
        redis_client.ping()
        logger.info(f"Redis connected to {redis_host}:{redis_port}")
        
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None

def initialize_ray_cluster():
    import asyncio
    import nest_asyncio
    
    # Apply nest_asyncio to allow event loop nesting
    nest_asyncio.apply()
    
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # Worker mode
        head_address = sys.argv[2] if len(sys.argv) > 2 else "localhost:6379"
        ray.init(address=head_address, _redis_password="ray123")
        logger.info(f"Worker connected to {head_address}")
    else:
        # Head mode
        ray.init(
            address="local",
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            _redis_password="ray123",
        )
        logger.info("Ray head node started")

def build_vllm_processor():
    """Build vLLM processor using official Ray Data API"""
    global processor, monitor, metrics
    
    # Setup vLLM engine config using official API
    vllm_config = vLLMEngineProcessorConfig(
        model_source=config["model"]["name"],
        concurrency=config["inference"]["concurrency"],
        batch_size=config["inference"]["batch_size"],
        engine_kwargs={
            "max_num_batched_tokens": config["inference"]["max_num_batched_tokens"],
            "max_model_len": config["model"]["max_model_len"],
            "gpu_memory_utilization": config["inference"]["gpu_memory_utilization"],
            "tensor_parallel_size": config["model"]["tensor_parallel_size"],
            # Enable chunked prefill for better memory efficiency
            "enable_chunked_prefill": config["model"].get("enable_chunked_prefill", True),
            "chunked_prefill_size": config["model"].get("chunked_prefill_size", 8192),
            # Enable speculative decoding for faster inference
            "speculative_model": config["model"].get("speculative_model"),
            "num_speculative_tokens": config["model"].get("num_speculative_tokens", 5),
            "speculative_draft_tensor_parallel_size": config["model"].get("speculative_draft_tensor_parallel_size", 1),
            # KV Cache optimization
            "kv_cache_config": {
                "enable_prefix_caching": config["model"].get("enable_prefix_caching", True),
                "max_cache_size": config["model"].get("max_kv_cache_size", 4096),
                "cache_dtype": "fp16"  # Use half-precision for cache
            },
            # Prompt caching for repeated prompts
            "prompt_cache_config": {
                "enable_prompt_cache": config["model"].get("enable_prompt_cache", True),
                "max_prompt_cache_size": config["model"].get("max_prompt_cache_size", 1024)
            }
        }
    )
    
    # Initialize monitoring
    metrics = BatchMetrics(total_requests=config["data"]["num_samples"])
    monitor = InferenceMonitor(metrics, sla_hours=config["sla"]["target_hours"])
    
    def preprocess(row):
        """Preprocess function for Ray Data"""
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": row.get("prompt", "")}
            ],
            "sampling_params": {
                "temperature": config["inference"]["temperature"],
                "max_tokens": config["inference"]["max_tokens"],
            }
        }
    
    def postprocess(row):
        """Postprocess function for Ray Data"""
        # Estimate tokens (rough approximation)
        tokens = len(row["generated_text"].split()) * 1.3
        monitor.update(batch_size=1, tokens=int(tokens))
        
        return {
            "response": row["generated_text"],
            "prompt": row.get("prompt", ""),
        }
    
    # Build processor using official API
    processor = build_llm_processor(
        vllm_config, 
        preprocess=preprocess, 
        postprocess=postprocess
    )
    
    logger.info("vLLM processor built successfully")

def run_batch_inference(prompts: List[str]) -> tuple:
    """Run batch inference using Ray Data and vLLM processor"""
    global processor, monitor, metrics
    
    if not processor:
        raise Exception("vLLM processor not initialized")
    
    try:
        # Create Ray Dataset from prompts
        ds = ray.data.from_items([{"prompt": prompt} for prompt in prompts])
        logger.info(f"Created dataset with {ds.count()} samples")
        
        # Initialize metrics for this batch
        if metrics is None:
            metrics = BatchMetrics(total_requests=len(prompts))
            monitor = InferenceMonitor(metrics, sla_hours=config["sla"]["target_hours"])
        
        # Run inference using the processor
        logger.info("Starting batch inference...")
        start_time = time.time()
        
        result_ds = processor(ds)
        results = result_ds.take_all()
        
        inference_time = time.time() - start_time
        
        logger.info(f"Batch inference completed in {inference_time:.2f} seconds")
        logger.info(f"Processed {len(results)} samples")
        
        return results, inference_time
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise


app = FastAPI(title="Ray Data vLLM Batch Inference", version="1.0.0")

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
        "service": "ray-data-vllm-official",
        "ray_version": ray.__version__,
        "model": config.get("model", {}).get("name", "unknown"),
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        return {
            "status": "healthy",
            "service": "ray-data-vllm-official",
            "ray_nodes": len(ray.nodes()),
            "model": config.get("model", {}).get("name", "unknown"),
            "processor_initialized": processor is not None,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/generate_batch", response_model=BatchInferenceResponse)
async def generate_batch(request: BatchInferenceRequest):
    """Distributed batch inference using official Ray Data API"""
    try:
        results, total_time = run_batch_inference(request.prompts)
        
        # Create response objects
        response_results = [
            InferenceResponse(
                text=result.get("response", ""),
                tokens_generated=len(result.get("response", "").split()),
                inference_time=total_time / len(request.prompts),
            )
            for result in results
        ]
        
        # Calculate metrics
        throughput = len(request.prompts) / total_time if total_time > 0 else 0
        throughput_gauge.set(throughput)
        
        return BatchInferenceResponse(
            results=response_results,
            total_time=total_time,
            total_prompts=len(request.prompts),
            throughput=throughput,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Bearer token required")
    
    # Simple token validation (in production, use proper JWT)
    token = credentials.credentials
    if not token or len(token) < 10:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

@app.post("/start", response_model=BatchJobResponse)
async def start_batch_job(request: BatchJobRequest, token: str = Depends(verify_token)):
    """Start a batch inference job with job_id and SLA tracking"""
    import uuid
    
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())[:8]
        
        # Update global config with job-specific parameters
        config["data"]["input_path"] = request.input_path
        config["data"]["output_path"] = request.output_path
        config["data"]["num_samples"] = request.num_samples
        config["inference"]["max_tokens"] = request.max_tokens
        config["inference"]["temperature"] = request.temperature
        config["inference"]["batch_size"] = request.batch_size
        config["inference"]["concurrency"] = request.concurrency
        
        # Initialize metrics for this job
        global metrics, monitor, redis_client
        sla_tier_config = getattr(SLATier, request.sla_tier.upper(), SLATier.BASIC)
        metrics = BatchMetrics(total_requests=request.num_samples)
        monitor = InferenceMonitor(metrics, sla_tier=sla_tier_config)
        
        # Create data artifact for input
        input_data = [{"prompt": f"sample_prompt_{i}"} for i in range(request.num_samples)]
        input_artifact = create_data_artifact(input_data)
        metrics.artifact_id = input_artifact.sha256_hash[:16]
        
        # Add job to Redis queue if available
        if redis_client:
            # Create data artifact for input
            input_data = [{"prompt": f"sample_prompt_{i}"} for i in range(request.num_samples)]
            input_artifact = create_data_artifact(input_data)
            
            job_data = {
                "job_id": job_id,
                "input_path": request.input_path,
                "output_path": request.output_path,
                "num_samples": request.num_samples,
                "status": "queued",
                "created_at": time.time(),
                "auth_token": token[:10] + "...",  # Store partial token for audit
                "input_artifact": input_artifact.sha256_hash,
                "artifact_version": input_artifact.version
            }
            redis_client.hset(f"job:{job_id}", mapping=job_data)
            redis_client.lpush("batch_job_queue", job_id)
            
            # Store artifact locally (S3 in production)
            storage_path = config.get("storage", {}).get("local_path", "/tmp/artifacts")
            os.makedirs(storage_path, exist_ok=True)
            store_artifact(input_artifact, storage_path)
            
            logger.info(f"Job {job_id} added to Redis queue with token validation and artifact storage")
        
        # Estimate completion time based on tier and baseline throughput
        sla_tier_config = getattr(SLATier, request.sla_tier.upper(), SLATier.BASIC)
        baseline_throughput = 10.0 * sla_tier_config["priority"]  # priority multiplier
        estimated_hours = request.num_samples / baseline_throughput / 3600
        
        logger.info(f"Started batch job {job_id} with {request.num_samples} samples")
        
        return BatchJobResponse(
            job_id=job_id,
            status="started",
            message=f"Batch job started with {request.num_samples} samples",
            estimated_completion_hours=estimated_hours
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    # Load configuration
    load_config()
    
    # Initialize Redis
    initialize_redis()
    
    # Initialize Ray cluster
    initialize_ray_cluster()
    
    # Build vLLM processor
    build_vllm_processor()
    
    # Start FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")