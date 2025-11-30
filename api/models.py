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


if __name__ == "__main__":
    logger.info("Testing API models...")
    
    logger.info(f"Priority levels: {list(priorityLevels)}")
    logger.info(f"LOW = {priorityLevels.LOW.value}")
    logger.info(f"HIGH = {priorityLevels.HIGH.value}")
    
    batch_req = BatchRequest(
        prompts=["Hello world", "What is AI?"],
        max_tokens=128,
        temperature=0.5
    )
    logger.info(f"BatchRequest created: {batch_req.prompts}")
    
    batch_resp = BatchResponse(
        results=[{"response": "test"}],
        total_time=1.5,
        total_prompts=2,
        throughput=1.33
    )
    logger.info(f"BatchResponse created: {batch_resp.throughput:.2f} prompts/sec")
    
    openai_req = OpenAIBatchRequest(
        model="test-model",
        input=[{"prompt": "test"}],
        max_tokens=64
    )
    logger.info(f"OpenAIBatchRequest created: {openai_req.model}")
    
    openai_resp = OpenAIBatchResponse(
        id="test-123",
        created_at=1234567890,
        status="completed"
    )
    logger.info(f"OpenAIBatchResponse created: {openai_resp.id}")
    
    logger.info("Models test completed!")