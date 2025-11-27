from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

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