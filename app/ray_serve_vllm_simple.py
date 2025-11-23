#!/usr/bin/env python3
"""
Simplified Ray Serve deployment for vLLM batch inference
"""
import os
import time
import torch
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ray
from ray import serve
from prometheus_client import start_http_server, Gauge, Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
import threading

# Prometheus metrics
inference_requests = Counter(
    'ray_serve_requests_total',
    'Total Ray Serve requests',
    ['deployment', 'method', 'status']
)

inference_duration = Histogram(
    'ray_serve_inference_duration_seconds',
    'Ray Serve inference duration',
    ['deployment', 'worker_id', 'model_name']
)

active_requests = Gauge(
    'ray_serve_active_requests',
    'Active Ray Serve requests'
)

# Pydantic models
class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

class BatchInferenceRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 100
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    inference_time: float
    worker_id: str
    deployment: str

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]
    total_time: float
    worker_id: str
    deployment: str

@serve.deployment(
    name="vllm_deployment",
    num_replicas=1,
    ray_actor_options={"num_gpus": 1} if torch.cuda.is_available() else {}
)
class VLLMInference:
    def __init__(self):
        self.model_name = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
        self.worker_id = os.environ.get('WORKER_ID', f'worker-{ray.get_runtime_context().get_node_id()}')
        self.deployment = "vllm_deployment"
        self.load_model()
        self.start_metrics_monitoring()
    
    def load_model(self):
        """Initialize model with vLLM or transformers fallback"""
        try:
            # Try vLLM first
            from vllm import LLM, SamplingParams
            
            print(f"Loading {self.model_name} with vLLM on {self.worker_id}...")
            
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                trust_remote_code=True
            )
            
            self.use_vllm = True
            print(f"✅ vLLM model loaded on {self.worker_id}")
            
        except Exception as e:
            print(f"⚠ vLLM failed, falling back to transformers: {e}")
            self.load_transformers_fallback()
    
    def load_transformers_fallback(self):
        """Fallback to transformers"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading {self.model_name} with transformers on {self.worker_id}...")
            
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
            print(f"✅ Transformers model loaded on {self.worker_id}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def start_metrics_monitoring(self):
        """Start background metrics monitoring"""
        def monitor_gpu():
            while True:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        utilization = torch.cuda.utilization(i)
                        # Simple GPU monitoring
                time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
        monitor_thread.start()
    
    def generate_with_vllm(self, prompts: List[str], max_tokens: int, temperature: float) -> List[Dict]:
        """Generate using vLLM"""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        inference_time = time.time() - start_time
        
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            results.append({
                'text': generated_text,
                'tokens_generated': len(output.outputs[0].token_ids),
                'inference_time': inference_time / len(prompts)
            })
        
        return results
    
    def generate_with_transformers(self, prompts: List[str], max_tokens: int, temperature: float) -> List[Dict]:
        """Generate using transformers"""
        results = []
        
        for i, prompt in enumerate(prompts):
            start_time = time.time()
            
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
            
            inference_time = time.time() - start_time
            tokens_generated = len(self.tokenizer.encode(generated_text))
            
            results.append({
                'text': generated_text,
                'tokens_generated': tokens_generated,
                'inference_time': inference_time
            })
        
        return results
    
    async def generate_batch(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """Handle batch inference request"""
        active_requests.inc()
        
        try:
            start_time = time.time()
            
            # Choose generation method
            if hasattr(self, 'use_vllm') and self.use_vllm:
                results = self.generate_with_vllm(
                    request.prompts, 
                    request.max_tokens, 
                    request.temperature
                )
            else:
                results = self.generate_with_transformers(
                    request.prompts, 
                    request.max_tokens, 
                    request.temperature
                )
            
            total_time = time.time() - start_time
            
            # Record metrics
            inference_requests.labels(
                deployment=self.deployment,
                method="generate_batch",
                status="success"
            ).inc(len(request.prompts))
            
            for result in results:
                inference_duration.labels(
                    deployment=self.deployment,
                    worker_id=self.worker_id,
                    model_name=self.model_name
                ).observe(result['inference_time'])
            
            # Create response objects
            response_results = [
                InferenceResponse(
                    text=result['text'],
                    tokens_generated=result['tokens_generated'],
                    inference_time=result['inference_time'],
                    worker_id=self.worker_id,
                    deployment=self.deployment
                )
                for result in results
            ]
            
            return BatchInferenceResponse(
                results=response_results,
                total_time=total_time,
                worker_id=self.worker_id,
                deployment=self.deployment
            )
            
        except Exception as e:
            inference_requests.labels(
                deployment=self.deployment,
                method="generate_batch",
                status="error"
            ).inc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            active_requests.dec()
    
    async def generate_single(self, request: InferenceRequest) -> InferenceResponse:
        """Handle single inference request"""
        batch_request = BatchInferenceRequest(
            prompts=[request.prompt],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        batch_response = await self.generate_batch(batch_request)
        
        return InferenceResponse(
            text=batch_response.results[0].text,
            tokens_generated=batch_response.results[0].tokens_generated,
            inference_time=batch_response.results[0].inference_time,
            worker_id=batch_response.worker_id,
            deployment=batch_response.deployment
        )

# FastAPI app for Ray Serve HTTP adapter
app = FastAPI(title="Ray Serve vLLM Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ray-serve-vllm",
        "ray_version": ray.__version__,
        "worker_id": os.environ.get('WORKER_ID', 'unknown')
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        return {
            "status": "healthy",
            "service": "ray-serve-vllm",
            "ray_nodes": len(ray.nodes()),
            "worker_id": os.environ.get('WORKER_ID', 'unknown'),
            "model": os.environ.get('MODEL_NAME', 'unknown')
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Single text generation endpoint"""
    try:
        # Get deployment handle
        handle = serve.get_deployment("vllm_deployment").get_handle()
        return await handle.generate_single.remote(request)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Deployment not available: {e}")

@app.post("/generate_batch", response_model=BatchInferenceResponse)
async def generate_batch(request: BatchInferenceRequest):
    """Batch text generation endpoint"""
    try:
        # Get deployment handle
        handle = serve.get_deployment("vllm_deployment").get_handle()
        return await handle.generate_batch.remote(request)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Deployment not available: {e}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def deploy_ray_serve():
    """Deploy Ray Serve with vLLM"""
    # Start Ray Serve
    serve.start(
        http_options={
            "host": "0.0.0.0",
            "port": 8000
        }
    )
    
    # Deploy vLLM inference
    VLLMInference.deploy()
    
    # Deploy FastAPI HTTP adapter
    serve.run(app)
    
    print("🚀 Ray Serve with vLLM deployed successfully")
    print(f"📍 API: http://localhost:8000")
    print(f"📊 Ray Dashboard: http://localhost:8265")
    print(f"🔍 Metrics: http://localhost:8000/metrics")

if __name__ == "__main__":
    import sys
    
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
    
    # Deploy Ray Serve
    deploy_ray_serve()
    
    try:
        while True:
            time.sleep(10)
            # Print status every 10 seconds
            print(f"Ray nodes: {len(ray.nodes())}")
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
        serve.shutdown()
        ray.shutdown()