import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import threading
import uuid
logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

@dataclass
class JobStore():
    jobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self, data: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())[:12]  
        job_data = {
            "id": job_id,
            "status": JobStatus.QUEUED,
            "created_at": float(time.time()),
            "model": data.get("model"),
            "num_prompts": data.get("num_prompts"),
            "input_file": data.get("input_file"),
            "output_file": None,  # Will be set when completed
            "error_file": None,   # Will be set if errors occur
            "started_at": None,
            "completed_at": None,
            "processed_count": 0,
            "error_count": 0
        }

        with self.lock:
            self.jobs[job_id] = job_data
        
        return self.jobs[job_id]['id']

    def update(self, job_id: str, data: Dict[str, Any]) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(data)
                self.jobs[job_id]['updated_at'] = int(time.time())
                if data.get("status") == JobStatus.COMPLETED:
                    self.jobs[job_id]['completed_at'] = int(time.time())
                if data.get("status") == JobStatus.RUNNING:
                    self.jobs[job_id]['started_at'] = int(time.time())
                if data.get("status") == JobStatus.FAILED:
                    self.jobs[job_id]['completed_at'] = int(time.time())
            else:
                raise KeyError(f"Job ID {job_id} not found")
                        
            
    def get(self, job_id: str) -> Dict[str, Any]:
        with self.lock:
            if job_id in self.jobs:
                return self.jobs[job_id]
            else:
                raise KeyError(f"Job ID {job_id} not found")
        

    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.lock:
            if status:
                return [job for job in self.jobs.values() if job['status'] == status]
            return list(self.jobs.values())
    



class BatchInferenceRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of prompts to process")
    max_tokens: Optional[int] = Field(256, description="Maximum tokens per response")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")

class InferenceResponse(BaseModel):
    text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    inference_time: float = Field(..., description="Inference time in seconds")

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse] = Field(..., description="Batch inference results")
    total_time: float = Field(..., description="Total processing time")
    total_prompts: int = Field(..., description="Total number of prompts processed")
    throughput: float = Field(..., description="Prompts per second")

class AuthBatchJobRequest(BaseModel):
    input_path: Optional[str] = Field(None, description="Input data path")
    output_path: Optional[str] = Field(None, description="Output data path")
    num_samples: int = Field(100, description="Number of samples to process")
    batch_size: Optional[int] = Field(128, description="Batch size for processing")
    concurrency: Optional[int] = Field(2, description="Concurrency level")

class BatchJobResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
    estimated_completion_hours: float = Field(..., description="Estimated completion time")

# OpenAI-compatible batch schemas
class BatchInputItem(BaseModel):
    prompt: str = Field(..., description="Input prompt")

class OpenAIBatchCreateRequest(BaseModel):
    model: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="Model name")
    input: List[BatchInputItem] = Field(..., description="List of input items")
    max_tokens: Optional[int] = Field(256, description="Maximum tokens per response")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")

class OpenAIBatchCreateResponse(BaseModel):
    id: str = Field(..., description="Batch job ID")
    object: str = Field("batch", description="Object type")
    created_at: int = Field(..., description="Creation timestamp")
    status: JobStatus = Field(..., description="Job status")

class OpenAIBatchRetrieveResponse(BaseModel):
    id: str = Field(..., description="Batch job ID")
    object: str = Field("batch", description="Object type")
    created_at: int = Field(..., description="Creation timestamp")
    completed_at: Optional[int] = Field(None, description="Completion timestamp")
    status: JobStatus = Field(..., description="Job status")
    results_file: Optional[str] = Field(None, description="Results file path")
    error_file: Optional[str] = Field(None, description="Error file path")

class BatchResultItem(BaseModel):
    id: str = Field(..., description="Result item ID")
    input: BatchInputItem = Field(..., description="Original input")
    output_text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")

class OpenAIBatchResultsResponse(BaseModel):
    object: str = Field("list", description="Object type")
    data: List[BatchResultItem] = Field(..., description="Batch results")

class BatchCostEstimate(BaseModel):
    input_tokens: int = Field(..., description="Total input tokens")
    max_output_tokens: int = Field(..., description="Maximum output tokens")
    estimated_cost_usd: float = Field(..., description="Estimated cost in USD")

class BatchValidationRequest(BaseModel):
    json_data: Dict[str, Any] = Field(..., description="JSON data to validate")
    max_batch_size: int = Field(1000, description="Maximum allowed batch size")

class BatchValidationResponse(BaseModel):
    is_valid: bool = Field(..., description="Validation result")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    estimated_items: int = Field(..., description="Number of items that will be processed")

class BatchRetryConfig(BaseModel):
    max_retries: int = Field(3, description="Maximum retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    backoff_factor: float = Field(2.0, description="Exponential backoff factor")

class BatchInputWithIndex(BaseModel):
    index: int = Field(..., description="Input index for mapping")
    prompt: str = Field(..., description="Input prompt")

class OpenAIBatchCreateRequestWithIndex(BaseModel):
    model: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="Model name")
    input: List[BatchInputWithIndex] = Field(..., description="List of input items with indices")
    max_tokens: Optional[int] = Field(256, description="Maximum tokens per response")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    retry_config: Optional[BatchRetryConfig] = Field(None, description="Retry configuration")
    validate_input: bool = Field(True, description="Whether to validate input before processing")



@dataclass
class DataArtifact:
    """Immutable data artifact with SHA tracking"""
    sha256_hash: str
    content_hash: str
    version: str
    created_at: float
    size_bytes: int

@dataclass
class BatchMetrics:
    """Batch processing metrics with SLA tracking"""
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    start_time: float = 0.0
    tokens_processed: int = 0
    artifact_id: str = ""
    
    def progress_pct(self) -> float:
        return (self.completed_requests / self.total_requests) * 100 if self.total_requests > 0 else 0
    
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



@dataclass
class InferenceMonitor:
    """Simple monitoring without SLA tiers"""
    def __init__(self, metrics: BatchMetrics, log_interval: int = 100):
        self.metrics = metrics
        self.log_interval = log_interval
        self.last_log = 0
    
    def update(self, batch_size: int, tokens: int):
        self.metrics.completed_requests += batch_size
        self.metrics.tokens_processed += tokens
        
        if self.metrics.completed_requests - self.last_log >= self.log_interval:
            self.log_progress()
            self.last_log = self.metrics.completed_requests
    
    def log_progress(self):
        logger.info(
            f"Progress: {self.metrics.progress_pct():.1f}% | "
            f"Completed: {self.metrics.completed_requests}/{self.metrics.total_requests} | "
            f"Throughput: {self.metrics.throughput_per_sec():.2f} req/s | "
            f"Tokens/sec: {self.metrics.tokens_per_sec():.2f} | "
            f"ETA: {self.metrics.eta_hours():.2f}h"
        )
    
    def check_sla(self) -> bool:
        return True

def create_data_artifact(data: List[Dict]) -> DataArtifact:
    """Create SHA-based data artifact"""
    import json
    import hashlib
    
    content_str = json.dumps(data, sort_keys=True)
    sha256_hash = hashlib.sha256(content_str.encode()).hexdigest()
    content_hash = hashlib.md5(content_str.encode()).hexdigest()
    
    return DataArtifact(
        sha256_hash=sha256_hash,
        content_hash=content_hash,
        version=f"v{int(time.time())}",
        created_at=time.time(),
        size_bytes=len(content_str.encode())
    )

def store_artifact(artifact: DataArtifact, storage_path: str):
    """Store artifact to local storage (S3 in production)"""
    import os
    import json
    
    os.makedirs(storage_path, exist_ok=True)
    artifact_file = os.path.join(storage_path, f"artifact_{artifact.sha256_hash[:8]}.json")
    
    with open(artifact_file, 'w') as f:
        json.dump(artifact.__dict__, f, indent=2)
    
    logger.info(f"Stored artifact {artifact.sha256_hash[:8]} to {artifact_file}")
    return artifact_file