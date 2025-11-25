#!/usr/bin/env python3
"""
Custom Prometheus exporter for LLM inference metrics
Monitors KV cache utilization, queue depth, and inference performance
"""

import os
import time
import torch
from prometheus_client import start_http_server, Gauge, Histogram, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import queue
from typing import List


class InferenceMetrics:
    def __init__(self):
        # Prometheus metrics
        self.kv_cache_utilization = Gauge(
            "llm_kv_cache_utilization_percent",
            "KV cache utilization percentage",
            ["worker_id", "model_name"],
        )

        self.queue_depth = Gauge(
            "llm_queue_depth", "Number of requests waiting in queue", ["worker_id"]
        )

        self.inference_duration = Histogram(
            "llm_inference_duration_seconds",
            "Time spent on inference",
            ["worker_id", "model_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        )

        self.tokens_generated = Counter(
            "llm_tokens_generated_total",
            "Total number of tokens generated",
            ["worker_id", "model_name"],
        )

        self.gpu_memory_utilization = Gauge(
            "llm_gpu_memory_utilization_percent",
            "GPU memory utilization percentage",
            ["worker_id", "gpu_id"],
        )

        self.active_requests = Gauge(
            "llm_active_requests",
            "Number of currently active inference requests",
            ["worker_id"],
        )

        self.worker_id = os.environ.get("WORKER_ID", "worker-1")
        self.model_name = os.environ.get("MODEL_NAME", "Qwen2.5-0.5B")

        # Request queue for simulation
        self.request_queue = queue.Queue()
        self.active_requests_count = 0

    def update_kv_cache_metrics(self, model):
        """Update KV cache utilization metrics"""
        if hasattr(model, "config") and hasattr(
            model.config, "max_position_embeddings"
        ):
            # Estimate KV cache usage based on allocated memory
            kv_cache_memory = torch.cuda.memory_allocated() * 0.4  # Rough estimate
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            kv_utilization = (kv_cache_memory / total_gpu_memory) * 100

            self.kv_cache_utilization.labels(
                worker_id=self.worker_id, model_name=self.model_name
            ).set(kv_utilization)

    def update_queue_metrics(self):
        """Update queue depth metrics"""
        queue_size = self.request_queue.qsize()
        self.queue_depth.labels(worker_id=self.worker_id).set(queue_size)
        self.active_requests.labels(worker_id=self.worker_id).set(
            self.active_requests_count
        )

    def update_gpu_metrics(self):
        """Update GPU memory utilization"""
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            total = torch.cuda.get_device_properties(i).total_memory
            utilization = (allocated / total) * 100

            self.gpu_memory_utilization.labels(
                worker_id=self.worker_id, gpu_id=str(i)
            ).set(utilization)

    def record_inference(self, duration: float, tokens: int):
        """Record completed inference"""
        self.inference_duration.labels(
            worker_id=self.worker_id, model_name=self.model_name
        ).observe(duration)

        self.tokens_generated.labels(
            worker_id=self.worker_id, model_name=self.model_name
        ).inc(tokens)

    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        start_http_server(port)
        print(f"Metrics server started on port {port}")

        # Start background metrics update thread
        def update_metrics():
            while True:
                self.update_queue_metrics()
                self.update_gpu_metrics()
                time.sleep(5)

        metrics_thread = threading.Thread(target=update_metrics, daemon=True)
        metrics_thread.start()


class BatchInferenceWorker:
    def __init__(self, model_path: str, metrics: InferenceMetrics):
        self.metrics = metrics
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model from {self.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            dtype=torch.float16,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print("Model loaded successfully")

    def process_request(self, prompt: str, max_tokens: int = 100) -> str:
        """Process a single inference request"""
        # Simulate queue
        self.metrics.request_queue.put(prompt)

        # Update active requests
        self.metrics.active_requests_count += 1

        try:
            start_time = time.time()

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_response[len(prompt) :].strip()

            # Calculate metrics
            duration = time.time() - start_time
            tokens_generated = len(self.tokenizer.encode(generated_text))

            # Record metrics
            self.metrics.record_inference(duration, tokens_generated)
            self.metrics.update_kv_cache_metrics(self.model)

            # Remove from queue
            try:
                self.metrics.request_queue.get_nowait()
            except queue.Empty:
                pass

            return generated_text

        finally:
            self.metrics.active_requests_count -= 1

    def run_batch_inference(self, prompts: List[str]):
        """Run batch inference with monitoring"""
        print(f"Processing {len(prompts)} prompts...")

        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i + 1}/{len(prompts)}")

            # Simulate some queue depth
            time.sleep(0.5)

            response = self.process_request(prompt)
            print(f"Generated {len(self.tokenizer.encode(response))} tokens")

        print("Batch inference completed")


def main():
    # Initialize metrics
    metrics = InferenceMetrics()
    metrics.start_metrics_server(port=8000)

    # Load model
    model_path = "Qwen/Qwen2.5-0.5B"
    worker = BatchInferenceWorker(model_path, metrics)

    # Example batch prompts
    prompts = [
        "What is the future of artificial intelligence?",
        "Explain quantum computing in simple terms.",
        "How does machine learning work?",
        "What are the applications of deep learning?",
        "Describe the impact of AI on society.",
    ]

    # Wait a bit for metrics server to start
    time.sleep(2)

    # Run batch inference
    worker.run_batch_inference(prompts)

    print("\nMetrics available at http://localhost:8000/metrics")
    print("Worker ID:", metrics.worker_id)
    print("Model:", metrics.model_name)

    # Keep running to allow metrics collection
    try:
        while True:
            time.sleep(10)
            metrics.update_kv_cache_metrics(worker.model)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
