#!/usr/bin/env python3
"""
Ray cluster configuration for multi-node setup
"""
import ray
import time
import os
from ray import serve
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def start_head_node():
    """Start Ray head node"""
    # Stop existing Ray
    os.system("ray stop")
    
    # Start head node with correct parameters
    ray.init(
        address="local",
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
        _redis_password="ray123"
    )
    
    print(f"🚀 Ray Head Node Started")
    print(f"📍 Dashboard: http://localhost:8265")
    # Get dashboard URL safely
    try:
        dashboard_url = ray.util.client.ray.get_dashboard_url()
    except:
        dashboard_url = "http://localhost:8265"
    print(f"🔗 Head address: {dashboard_url}")
    
    # Get the actual head address for workers to connect
    import socket
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    head_address = f"{host_ip}:6379"
    
    print(f"🔗 Worker connection: ray start --address='{head_address}' --redis-password='ray123'")
    
    return head_address

@serve.deployment(num_replicas=2)
class DistributedLLM:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        print(f"Loading {self.model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Model loaded on node: {ray.get_runtime_context().get_node_id()}")
    
    def __call__(self, request):
        """Generate text"""
        import json
        data = json.loads(request.body)
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 100)
        temperature = data.get("temperature", 0.7)
        
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt):].strip()
        
        inference_time = time.time() - start_time
        tokens_generated = len(self.tokenizer.encode(generated_text))
        
        return {
            'text': generated_text,
            'tokens_generated': tokens_generated,
            'inference_time': inference_time,
            'node_id': ray.get_runtime_context().get_node_id(),
            'worker_id': ray.get_runtime_context().get_worker_id()
        }

def setup_distributed_serve():
    """Setup Ray Serve with distributed deployment"""
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    
    # Deploy distributed LLM service
    DistributedLLM.deploy()
    
    print("\n🌐 Distributed LLM Service Ready")
    print(f"📍 API: http://localhost:8000")
    print(f"📊 Dashboard: http://localhost:8265")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # Connect as worker
        head_address = sys.argv[2] if len(sys.argv) > 2 else "localhost:6379"
        ray.init(address=head_address, _redis_password="ray123")
        print(f"✅ Worker connected to cluster at {head_address}")
    else:
        # Start head node
        head_address = start_head_node()
        setup_distributed_serve()