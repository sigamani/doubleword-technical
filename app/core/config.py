"""
Configuration management with environment variable support
"""

import os
from typing import Dict, Any
from pydantic import BaseSettings

logger = logging.getLogger(__name__)

class ModelConfig(BaseSettings):
    """Model configuration with validation"""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_model_len: int = 32768
    tensor_parallel_size: int = 2
    
    class Config:
        env_prefix = "MODEL_"

class InferenceConfig(BaseSettings):
    """Inference parameters with validation"""
    batch_size: int = 128
    concurrency: int = 2
    temperature: float = 0.7
    max_tokens: int = 512
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.90
    
    class Config:
        env_prefix = "INFERENCE_"

class StorageConfig(BaseSettings):
    """Storage configuration"""
    local_path: str = "/tmp/artifacts"
    s3_bucket: str = "batch-inference-artifacts"
    
    class Config:
        env_prefix = "STORAGE_"

class SLAConfig(BaseSettings):
    """SLA configuration with tier support"""
    target_hours: float = 24.0
    buffer_factor: float = 0.7
    alert_threshold_hours: float = 20.0
    
    class Config:
        env_prefix = "SLA_"

class MonitoringConfig(BaseSettings):
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    prometheus_port: int = 8001
    grafana_enabled: bool = False
    loki_enabled: bool = False
    
    class Config:
        env_prefix = "MONITORING_"

class AppConfig(BaseSettings):
    """Main application configuration"""
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    storage: StorageConfig = StorageConfig()
    sla: SLAConfig = SLAConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from YAML with validation"""
        import yaml
        
        try:
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Override with environment variables
            return cls(**config_data)
            
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return cls()  # Return defaults

def get_config() -> AppConfig:
    """Get application configuration"""
    config_path = os.getenv("CONFIG_PATH", "/app/config/config.yaml")
    return AppConfig.from_yaml(config_path)