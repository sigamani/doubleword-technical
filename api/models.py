""" Data models for batch processing API. ""

from enum import Enum
from typing import Any, Dict, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class priorityLevels(Enum):
    LOW = 1
    HIGH = 10

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