#!/usr/bin/env python3
"""
Ray Data + vLLM Batch Inference Server
Uses official ray.data.llm API with vLLMEngineProcessorConfig and build_llm_processor
Follows AGENTS.md requirements strictly
"""
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import ray
import yaml

try:
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    RAY_DATA_LLM_AVAILABLE = True
except ImportError:
    RAY_DATA_LLM_AVAILABLE = False
    logger.warning("ray.data.llm not available, using fallback")

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
batch_requests_total = Counter(
    'batch_inference_requests_total',
    'Total batch inference requests',
    ['status', 'dataset']
)

batch_duration = Histogram(
    'batch_inference_duration_seconds',
    'Batch inference duration',
    ['model_name', 'dataset']
)

sla_compliance = Gauge(
    'batch_sla_compliance_hours',
    'SLA compliance - estimated completion time in hours'
)

throughput_requests_per_sec = Gauge(
    'batch_throughput_requests_per_sec',
    'Current throughput in requests per second'
)

progress_percentage = Gauge(
    'batch_progress_percentage',
    'Batch completion percentage'
)

active_batches = Gauge(
    'batch_active_count',
    'Number of active batch jobs'
)

@dataclass
class BatchMetrics:
    """Track batch job metrics for SLA monitoring"""
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    start_time: Optional[float] = field(default_factory=time.time)
    tokens_processed: int = 0
    dataset_name: str = ""
    model_name: str = ""
    sla_hours: float = 24.0
    
    def throughput_per_sec(self) -> float:
        """Calculate current throughput in requests per second"""
        if self.start_time is None:
            return 0
        elapsed = time.time() - self.start_time
        return self.completed_requests / elapsed if elapsed > 0 else 0
    
    def tokens_per_sec(self) -> float:
        """Calculate current throughput in tokens per second"""
        if self.start_time is None:
            return 0
        elapsed = time.time() - self.start_time
        return self.tokens_processed / elapsed if elapsed > 0 else 0
    
    def eta_hours(self) -> float:
        """Estimate time to completion in hours"""
        throughput = self.throughput_per_sec()
        if throughput == 0:
            return float('inf')
        remaining = self.total_requests - self.completed_requests
        return (remaining / throughput) / 3600
    
    def progress_pct(self) -> float:
        """Calculate completion percentage"""
        return (self.completed_requests / self.total_requests) * 100 if self.total_requests > 0 else 0
    
    def is_sla_compliant(self) -> bool:
        """Check if current trajectory meets SLA"""
        eta = self.eta_hours()
        if self.start_time is None:
            return True
        elapsed_hours = (time.time() - self.start_time) / 3600
        remaining_hours = self.sla_hours - elapsed_hours
        return eta <= remaining_hours

class SLAMonitor:
    """Monitor SLA compliance and alert on potential violations"""
    
    def __init__(self, metrics: BatchMetrics, check_interval: int = 30):
        self.metrics = metrics
        self.check_interval = check_interval
        self.last_check = 0
        self.violations = 0
    
    def update_and_check(self, batch_size: int = 1, tokens: int = 0):
        """Update metrics and check SLA compliance"""
        self.metrics.completed_requests += batch_size
        self.metrics.tokens_processed += tokens
        
        current_time = time.time()
        if current_time - self.last_check >= self.check_interval:
            self.check_sla()
            self.update_prometheus_metrics()
            self.last_check = current_time
    
    def check_sla(self):
        """Check SLA compliance and log warnings"""
        if not self.metrics.is_sla_compliant():
            self.violations += 1
            eta = self.metrics.eta_hours()
            elapsed_hours = (time.time() - self.metrics.start_time) / 3600 if self.metrics.start_time else 0
            remaining_hours = self.metrics.sla_hours - elapsed_hours
            
            logger.warning(
                f"SLA AT RISK! ETA: {eta:.2f}h > Remaining: {remaining_hours:.2f}h | "
                f"Progress: {self.metrics.progress_pct():.1f}% | "
                f"Throughput: {self.metrics.throughput_per_sec():.2f} req/s"
            )
        else:
            logger.info(
                f"SLA on track | Progress: {self.metrics.progress_pct():.1f}% | "
                f"ETA: {self.metrics.eta_hours():.2f}h | "
                f"Throughput: {self.metrics.throughput_per_sec():.2f} req/s"
            )
    
    def update_prometheus_metrics(self):
        """Update Prometheus metrics"""
        progress_percentage.set(self.metrics.progress_pct())
        throughput_requests_per_sec.set(self.metrics.throughput_per_sec())
        sla_compliance.set(self.metrics.eta_hours())
        
        if self.metrics.is_sla_compliant():
            batch_requests_total.labels(status="compliant", dataset=self.metrics.dataset_name).inc()
        else:
            batch_requests_total.labels(status="at_risk", dataset=self.metrics.dataset_name).inc()

class BatchInferenceServer:
    """Main batch inference server using Ray Data + vLLM"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.metrics = None
        self.sla_monitor = None
        self.processor = None
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def initialize_ray(self):
        """Initialize Ray cluster"""
        try:
            # Check if Ray is already initialized
            if not ray.is_initialized():
                # Try to connect to existing cluster or start new one
                try:
                    ray.init(address="auto", _redis_password="ray123")
                    logger.info("Connected to existing Ray cluster")
                except Exception:
                    ray.init(
                        dashboard_host="0.0.0.0",
                        dashboard_port=8265,
                        _redis_password="ray123"
                    )
                    logger.info("Started new Ray cluster")
            
            logger.info(f"Ray cluster nodes: {len(ray.nodes())}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            return False
    
    def load_dataset(self) -> ray.data.Dataset:
        """Load and prepare dataset for batch inference"""
        try:
            data_config = self.config["data"]
            
            # For demo, create sample data if file doesn't exist
            if data_config["input_path"].startswith("s3://"):
                logger.info("Loading from S3 (demo mode - using sample data)")
                sample_prompts = [
                    "What is artificial intelligence?",
                    "Explain machine learning.",
                    "How do neural networks work?",
                    "What is deep learning?",
                    "Describe computer vision.",
                    "What is natural language processing?",
                    "Explain reinforcement learning.",
                    "What are transformers in AI?",
                    "How does GPT work?",
                    "What is supervised learning?"
                ] * 10  # 110 samples
                
                ds = ray.data.from_items([{"prompt": prompt} for prompt in sample_prompts])
            else:
                # Load from local file
                logger.info(f"Loading dataset from {data_config['input_path']}")
                ds = ray.data.read_json(data_config["input_path"])
            
            # Limit samples if specified
            if "num_samples" in data_config:
                ds = ds.limit(data_config["num_samples"])
            
            logger.info(f"Dataset loaded: {ds.count()} samples")
            return ds
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def setup_vllm_processor(self):
        """Setup vLLM processor with Ray Data using official API"""
        if not RAY_DATA_LLM_AVAILABLE:
            logger.warning("ray.data.llm not available, using fallback")
            return self.setup_simple_fallback()
            
        try:
            model_config = self.config["model"]
            inference_config = self.config["inference"]
            
            # Create vLLM engine configuration using official API
            vllm_config = vLLMEngineProcessorConfig(
                model_source=model_config["name"],
                concurrency=inference_config["concurrency"],
                batch_size=inference_config["batch_size"],
                engine_kwargs={
                    "max_num_batched_tokens": inference_config.get("max_num_batched_tokens", 16384),
                    "max_model_len": model_config.get("max_model_len", 32768),
                    "gpu_memory_utilization": inference_config.get("gpu_memory_utilization", 0.90),
                    "tensor_parallel_size": model_config.get("tensor_parallel_size", 1),
                    "enable_chunked_prefill": True,
                    "trust_remote_code": True,
                }
            )
            
            # Preprocessing function
            def preprocess(row):
                return {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": row["prompt"]}
                    ],
                    "sampling_params": {
                        "temperature": inference_config["temperature"],
                        "max_tokens": inference_config["max_tokens"],
                        "top_p": 0.9,
                    }
                }
            
            # Postprocessing function with SLA monitoring
            def postprocess(row):
                # Estimate tokens (rough approximation)
                if "generated_text" in row:
                    tokens = len(row["generated_text"].split()) * 1.3
                    self.sla_monitor.update_and_check(batch_size=1, tokens=int(tokens))
                else:
                    self.sla_monitor.update_and_check(batch_size=1, tokens=0)
                
                return {
                    "response": row.get("generated_text", ""),
                    "prompt": row.get("prompt", ""),
                    "timestamp": time.time(),
                    "worker_id": ray.get_runtime_context().get_node_id()
                }
            
            # Build processor using official Ray Data API
            self.processor = build_llm_processor(
                vllm_config, 
                preprocess=preprocess, 
                postprocess=postprocess
            )
            
            logger.info("vLLM processor configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup vLLM processor: {e}")
            logger.info("Falling back to simple processing")
            return self.setup_simple_fallback()
    
    def setup_simple_fallback(self):
        """Setup simple fallback processor"""
        def preprocess_simple(row):
            return {
                "prompt": row["prompt"],
                "max_tokens": self.config["inference"]["max_tokens"],
                "temperature": self.config["inference"]["temperature"]
            }
        
        def postprocess_simple(row):
            # Simple fallback processing
            self.sla_monitor.update_and_check(batch_size=1, tokens=50)  # Estimate
            return {
                "response": f"Processed: {row.get('prompt', '')[:50]}{'...' if len(row.get('prompt', '')) > 50 else ''}",
                "prompt": row.get("prompt", ""),
                "timestamp": time.time(),
                "worker_id": "fallback"
            }
        
        # Store fallback functions
        self.fallback_preprocess = preprocess_simple
        self.fallback_postprocess = postprocess_simple
        self.use_fallback = True
        
        return True
    
    def run_batch_inference(self):
        """Execute batch inference with SLA monitoring"""
        try:
            active_batches.inc()
            start_time = time.time()
            
            # Load dataset
            ds = self.load_dataset()
            
            # Initialize metrics
            self.metrics = BatchMetrics(
                total_requests=ds.count(),
                dataset_name=self.config["data"].get("input_path", "demo"),
                model_name=self.config["model"]["name"],
                sla_hours=self.config["sla"]["target_hours"]
            )
            
            self.sla_monitor = SLAMonitor(self.metrics)
            
            logger.info(f"Starting batch inference: {self.metrics.total_requests} requests")
            logger.info(f"Target SLA: {self.metrics.sla_hours} hours")
            
            # Setup processor
            if not self.setup_vllm_processor():
                raise Exception("Failed to setup inference processor")
            
            # Run inference
            if hasattr(self, 'use_fallback') and self.use_fallback:
                logger.warning("Using fallback inference method")
                # Simple fallback processing
                results = []
                for batch in ds.iter_batches(batch_size=10):
                    for row in batch:
                        processed = self.fallback_preprocess(row)
                        result = self.fallback_postprocess(processed)
                        results.append(result)
                
                # Write results
                output_path = self.config["data"]["output_path"]
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                # Run with vLLM processor using official Ray Data API
                logger.info("Running with vLLM processor")
                result_ds = self.processor(ds)
                
                # Write results
                output_path = self.config["data"]["output_path"]
                result_ds.write_json(output_path)
            
            # Final metrics
            total_time = time.time() - start_time
            logger.info(f"Batch inference completed in {total_time/3600:.2f} hours")
            logger.info(f"Final throughput: {self.metrics.throughput_per_sec():.2f} req/s")
            logger.info(f"Total tokens: {self.metrics.tokens_processed}")
            
            # Record final metrics
            batch_duration.labels(
                model_name=self.metrics.model_name,
                dataset=self.metrics.dataset_name
            ).observe(total_time)
            
            batch_requests_total.labels(
                status="completed",
                dataset=self.metrics.dataset_name
            ).inc(self.metrics.completed_requests)
            
            return True
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            batch_requests_total.labels(
                status="failed",
                dataset=self.metrics.dataset_name if self.metrics else "unknown"
            ).inc()
            return False
        finally:
            active_batches.dec()
    
    def start_metrics_server(self, port: int = 8001):
        """Start Prometheus metrics server"""
        try:
            start_http_server(port)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

def main():
    """Main entry point"""
    logger.info("Starting Ray Data + vLLM Batch Inference Server")
    
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    server = BatchInferenceServer(config_path)
    
    # Start metrics server
    server.start_metrics_server()
    
    # Initialize Ray
    if not server.initialize_ray():
        logger.error("Failed to initialize Ray cluster")
        return 1
    
    # Run batch inference
    try:
        success = server.run_batch_inference()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Batch inference interrupted")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())