"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class BatchInferenceRequest(BaseModel):
    """Request for immediate batch inference"""
    prompts: List[str]
    max_tokens: int = Field(ge=1, le=2048, default=100)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    batch_size: int = Field(ge=1, le=1000, default=128)

class InferenceResponse(BaseModel):
    """Single inference result"""
    text: str
    tokens_generated: int
    inference_time: float

class BatchInferenceResponse(BaseModel):
    """Batch inference response"""
    results: List[InferenceResponse]
    total_time: float
    total_prompts: int
    throughput: float

class AuthBatchJobRequest(BaseModel):
    """Authenticated batch job request"""
    input_path: str
    output_path: str
    num_samples: int = Field(ge=1, le=100000, default=1000)
    max_tokens: int = Field(ge=1, le=2048, default=512)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    batch_size: int = Field(ge=1, le=1000, default=128)
    concurrency: int = Field(ge=1, le=10, default=2)
    sla_tier: str = Field(default="basic")  # free, basic, premium, enterprise

class BatchJobResponse(BaseModel):
    """Batch job response"""
    job_id: str
    status: str
    message: str
    estimated_completion_hours: float

class MultimodalInput(BaseModel):
    """Multimodal input support"""
    text: str = Field(default="")
    image_url: str = Field(default="")
    audio_url: str = Field(default="")
    video_url: str = Field(default="")
    metadata: Optional[dict] = Field(default_factory=dict)

class MultimodalAuthBatchJobRequest(BaseModel):
    """Multimodal authenticated batch job request"""
    input_path: str
    output_path: str
    num_samples: int = Field(ge=1, le=100000, default=1000)
    max_tokens: int = Field(ge=1, le=2048, default=512)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    batch_size: int = Field(ge=1, le=1000, default=128)
    concurrency: int = Field(ge=1, le=10, default=2)
    sla_tier: str = Field(default="basic")
    model_type: str = Field(default="text")  # text, multimodal
    multimodal_inputs: List[MultimodalInput] = Field(default_factory=list)