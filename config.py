""" The idea is to keep all configuration parameters centralised in one file."""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EnvironmentConfig:
    is_dev: bool
    is_gpu_available: bool
    
    @classmethod
    def from_env(cls):
        return cls(
            is_dev=os.getenv("ENVIRONMENT", "DEV").upper() == "DEV",
            is_gpu_available=os.getenv("GPU_AVAILABLE", "false").lower() == "true"
        )

@dataclass
class BatchConfig:
    """Configuration for batch processing and storage."""
    batch_dir: str = os.getenv("BATCH_DIR", "/tmp")
    max_queue_depth: int = int(os.getenv("MAX_QUEUE_DEPTH", "5000"))
    job_timeout_seconds: float = float(os.getenv("JOB_TIMEOUT_SECONDS", "30.0"))
    worker_poll_interval: float = float(os.getenv("WORKER_POLL_INTERVAL", "0.1"))

@dataclass
class ModelConfig:
    model_name: str
    batch_size: int
    concurrency: int
    max_model_len: int
    temperature: float
    max_tokens: int
    
    @classmethod
    def default(cls):
        return cls(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            batch_size=32,
            concurrency=1,
            max_model_len=512,
            temperature=0.7,
            max_tokens=256
        )


@dataclass
class VLLMEngineConfig:
    model_source: str
    batch_size: int
    concurrency: int
    engine_kwargs: Dict[str, Any]
    
    @classmethod
    def from_model_config(cls, config: ModelConfig):
        return cls(
            model_source=config.model_name,
            batch_size=config.batch_size,
            concurrency=config.concurrency,
            engine_kwargs={
                "max_model_len": config.max_model_len,
                "enforce_eager": True,
                "dtype": "float16",
                "gpu_memory_utilization": 0.85,
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 2048,
            }
        )