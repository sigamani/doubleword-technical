#!/usr/bin/env python3
"""
Ray Data + vLLM distributed batch inference orchestration
Following official Ray Data batch inference documentation:
https://docs.ray.io/en/latest/data/batch_inference.html
"""
import os
import time
import sys
import torch
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ray
from ray import data
from prometheus_client import start_http_server, Gauge, Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
import threading

# Prometheus metrics
inference_requests = Counter(
    'ray_data_requests_total',
    'Total Ray Data requests',
    ['method', 'status']
)

inference_duration = Histogram(
    'ray_data_inference_duration_seconds',
    'Ray Data inference duration',
    ['node_id', 'model_name']
)

active_batches = Gauge(
    'ray_data_active_batches',
    'Active Ray Data batches'
)

gpu_utilization = Gauge(
    'ray_data_gpu_utilization_percent',
    'GPU utilization per node',
    ['node_id', 'gpu_id']
)

# Pydantic models
class BatchInferenceRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 100
    temperature: float = 0.7
    batch_size: int = 4  # Ray Data batch size

class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    inference_time: float
    node_id: str

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]
    total_time: float
    total_prompts: int
    nodes_used: int
    throughput: float  # prompts per second

@ray.remote
class VLLMActor:
    """Ray actor for vLLM inference with GPU allocation"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.node_id = ray.get_runtime_context().get_node_id()
        self.load_model()
        self.start_metrics_monitoring()
    
    def load_model(self):
        """Initialize vLLM with GPU"""
        try:
            from vllm import LLM, SamplingParams
            
            print(f"Loading {self.model_name} on {self.node_id}...")
            
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                trust_remote_code=True
            )
            
            self.use_vllm = True
            print(f"✅ vLLM model loaded on {self.node_id}")
            
        except Exception as e:
            print(f"⚠ vLLM failed, falling back to transformers: {e}")
            self.load_transformers_fallback()
    
    def load_transformers_fallback(self):
        """Fallback to transformers"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading {self.model_name} with transformers on {self.node_id}...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                dtype=torch.float16,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.use_vllm = False
            print(f"✅ Transformers model loaded on {self.node_id}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def start_metrics_monitoring(self):
        """Start GPU monitoring"""
        def monitor_gpu():
            while True:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        utilization = torch.cuda.utilization(i)
                        gpu_utilization.labels(
                            node_id=self.node_id,
                            gpu_id=str(i)
                        ).set(utilization)
                time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
        monitor_thread.start()
    
    def infer_batch(self, prompts: List[str], max_tokens: int, temperature: float) -> List[Dict]:
        """Process a batch of prompts"""
        start_time = time.time()
        
        if self.use_vllm:
            results = self._infer_with_vllm(prompts, max_tokens, temperature)
        else:
            results = self._infer_with_transformers(prompts, max_tokens, temperature)
        
        inference_time = time.time() - start_time
        
        # Record metrics
        inference_duration.labels(
            node_id=self.node_id,
            model_name=self.model_name
        ).observe(inference_time)
        
        # Add node info to results
        for result in results:
            result['node_id'] = self.node_id
            result['inference_time'] = inference_time / len(prompts)  # Average time per prompt
        
        return results
    
    def _infer_with_vllm(self, prompts: List[str], max_tokens: int, temperature: float) -> List[Dict]:
        """Inference using vLLM"""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append({
                'text': generated_text,
                'tokens_generated': len(output.outputs[0].token_ids),
                'inference_time': 0  # Will be set by caller
            })
        
        return results
    
    def _infer_with_transformers(self, prompts: List[str], max_tokens: int, temperature: float) -> List[Dict]:
        """Inference using transformers"""
        results = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_response[len(prompt):].strip()
            
            results.append({
                'text': generated_text,
                'tokens_generated': len(self.tokenizer.encode(generated_text)),
                'inference_time': 0  # Will be set by caller
            })
        
        return results

# FastAPI app
app = FastAPI(title="Ray Data vLLM Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Ray actors pool
vllm_actors = []

def initialize_ray_cluster():
    """Initialize Ray cluster and create actors"""
    global vllm_actors
    
    # Initialize Ray
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # Worker mode
        head_address = sys.argv[2] if len(sys.argv) > 2 else "localhost:6379"
        ray.init(address=head_address, _redis_password="ray123")
        print(f"✅ Worker connected to {head_address}")
    else:
        # Head mode
        ray.init(
            address="local",
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            _redis_password="ray123"
        )
        print("🚀 Ray head node started")
    
    # Create vLLM actors (one per GPU node)
    model_name = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
    
    # Get available GPU nodes
    nodes = ray.nodes()
    gpu_nodes = [node for node in nodes if node.get('Resources', {}).get('GPU', 0) > 0]
    
    print(f"Found {len(gpu_nodes)} GPU nodes")
    
    # Create actors on GPU nodes
    for i, node in enumerate(gpu_nodes):
        actor = VLLMActor.options(
            num_gpus=1,
            resources={f"node:{node['NodeID']}": 1}
        ).remote(model_name)
        vllm_actors.append(actor)
        print(f"Created VLLM actor on node {node['NodeID']}")
    
    print(f"✅ Created {len(vllm_actors)} VLLM actors")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ray-data-vllm",
        "ray_version": ray.__version__,
        "actors_created": len(vllm_actors),
        "ray_nodes": len(ray.nodes())
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        nodes = ray.nodes()
        gpu_nodes = [node for node in nodes if node.get('Resources', {}).get('GPU', 0) > 0]
        
        return {
            "status": "healthy",
            "service": "ray-data-vllm",
            "ray_nodes": len(nodes),
            "gpu_nodes": len(gpu_nodes),
            "vllm_actors": len(vllm_actors),
            "model": os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def vllm_inference(batch):
    """Ray Data map_batches function - runs on distributed actors"""
    # batch is a Ray DataBatch, extract prompts
    # Convert batch to list of prompts
    prompts = list(batch)
    
    # Get current actor (this runs on actor itself)
    current_actor = ray.get_runtime_context().current_actor
    
    # Use the actor to process the batch
    return ray.get(current_actor.infer_batch.remote(prompts, 100, 0.7))

@app.post("/generate_batch", response_model=BatchInferenceResponse)
async def generate_batch(request: BatchInferenceRequest):
    """Distributed batch inference using Ray Data map_batches"""
    if not vllm_actors:
        raise HTTPException(status_code=503, detail="VLLM actors not initialized")
    
    active_batches.inc()
    
    try:
        start_time = time.time()
        
        # Create Ray Dataset from prompts
        ds = data.from_items(request.prompts)
        
        # Use map_batches for distributed processing
        # This automatically distributes batches across available actors
        batch_results = ds.map_batches(
            vllm_inference,
            batch_size=request.batch_size,
            num_gpus=1,  # Each actor gets 1 GPU
            concurrency=2   # 2 parallel actors (one per node)
        )
        
        # Collect results
        results = []
        for batch in batch_results.iter_batches():
            results.extend(batch)
        
        total_time = time.time() - start_time
        
        # Create response objects
        response_results = [
            InferenceResponse(
                text=result['text'],
                tokens_generated=result['tokens_generated'],
                inference_time=result['inference_time'],
                node_id=result['node_id']
            )
            for result in results
        ]
        
        # Calculate metrics
        throughput = len(request.prompts) / total_time if total_time > 0 else 0
        nodes_used = len(set(result['node_id'] for result in results))
        
        # Record metrics
        inference_requests.labels(
            method="generate_batch",
            status="success"
        ).inc(len(request.prompts))
        
        return BatchInferenceResponse(
            results=response_results,
            total_time=total_time,
            total_prompts=len(request.prompts),
            nodes_used=nodes_used,
            throughput=throughput
        )
        
    except Exception as e:
        inference_requests.labels(
            method="generate_batch",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_batches.dec()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    # Initialize Ray cluster and actors
    initialize_ray_cluster()
    
    # Start FastAPI server
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )