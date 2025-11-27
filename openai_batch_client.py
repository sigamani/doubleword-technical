#!/usr/bin/env python3
"""
OpenAI-Compatible Batch Client for Ray Data + vLLM Server
Provides OpenAI-like interface for batch processing
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class BatchClient:
    """OpenAI-compatible batch client"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def create(self, model: str, input: List[Dict[str, str]], 
               max_tokens: Optional[int] = None, 
               temperature: Optional[float] = None) -> Dict[str, Any]:
        """Create a batch job (OpenAI-compatible)"""
        url = f"{self.base_url}/v1/batches"
        
        payload = {
            "model": model,
            "input": [{"prompt": item.get("prompt", "")} for item in input],
            "max_tokens": max_tokens or 256,
            "temperature": temperature or 0.7
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def retrieve(self, id: str) -> Dict[str, Any]:
        """Retrieve batch job status"""
        url = f"{self.base_url}/v1/batches/{id}"
        
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_result(self, id: str) -> List[Dict[str, Any]]:
        """Get batch results"""
        url = f"{self.base_url}/v1/batches/{id}/results"
        
        response = self.session.get(url)
        response.raise_for_status()
        result = response.json()
        return result.get("data", [])
    
    def wait_for_completion(self, id: str, poll_interval: float = 5.0, 
                          timeout: float = 3600.0) -> Dict[str, Any]:
        """Wait for batch completion with polling"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.retrieve(id)
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                return status
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Batch {id} did not complete within {timeout} seconds")
    
    def validate_input(self, json_data: Dict[str, Any], 
                     max_batch_size: int = 1000) -> Dict[str, Any]:
        """Validate batch input JSON"""
        url = f"{self.base_url}/v1/batches/validate"
        
        payload = {
            "json_data": json_data,
            "max_batch_size": max_batch_size
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def estimate_cost(self, model: str, input: List[Dict[str, str]], 
                   max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None) -> Dict[str, Any]:
        """Estimate batch processing cost"""
        url = f"{self.base_url}/v1/batches/cost-estimate"
        
        payload = {
            "model": model,
            "input": [{"prompt": item.get("prompt", "")} for item in input],
            "max_tokens": max_tokens or 256,
            "temperature": temperature or 0.7
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()

# Convenience functions for OpenAI-like usage
def create_batch(model: str, input: List[Dict[str, str]], 
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Create batch job (OpenAI-compatible function)"""
    client = BatchClient(base_url)
    return client.create(model, input, max_tokens, temperature)

def retrieve_batch(id: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Retrieve batch status (OpenAI-compatible function)"""
    client = BatchClient(base_url)
    return client.retrieve(id)

def get_batch_results(id: str, base_url: str = "http://localhost:8000") -> List[Dict[str, Any]]:
    """Get batch results (OpenAI-compatible function)"""
    client = BatchClient(base_url)
    return client.get_result(id)

# Example usage
if __name__ == "__main__":
    # Example 1: Create batch job
    print("Creating batch job...")
    response = create_batch(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        input=[
            {"prompt": "What is AI?"},
            {"prompt": "Explain ML basics"},
            {"prompt": "How do neural networks work?"}
        ],
        max_tokens=100,
        temperature=0.7
    )
    
    job_id = response["id"]
    print(f"Batch job created with ID: {job_id}")
    print(f"Status: {response['status']}")
    
    # Example 2: Wait for completion
    print("\nWaiting for completion...")
    try:
        final_status = BatchClient().wait_for_completion(job_id, poll_interval=2.0)
        print(f"Job completed with status: {final_status['status']}")
        
        # Example 3: Get results
        if final_status["status"] == "completed":
            print("\nRetrieving results...")
            results = get_batch_results(job_id)
            
            for i, item in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Input: {item['input']['prompt']}")
                print(f"Output: {item['output_text']}")
                print(f"Tokens: {item['tokens_generated']}")
        
    except TimeoutError as e:
        print(f"Timeout: {e}")
    
    # Example 4: Validate input
    print("\nValidating input...")
    validation_response = BatchClient().validate_input({
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "input": [
            {"prompt": "What is AI?"},
            {"prompt": "Explain ML basics"}
        ],
        "max_tokens": 100
    })
    
    print(f"Validation result: {validation_response['is_valid']}")
    if validation_response['errors']:
        print(f"Errors: {validation_response['errors']}")
    if validation_response['warnings']:
        print(f"Warnings: {validation_response['warnings']}")
    
    # Example 5: Cost estimation
    print("\nEstimating cost...")
    cost_response = BatchClient().estimate_cost(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        input=[
            {"prompt": "What is AI?"},
            {"prompt": "Explain ML basics"},
            {"prompt": "How do neural networks work?"}
        ],
        max_tokens=100
    )
    
    print(f"Estimated cost: ${cost_response['estimated_cost_usd']:.6f}")
    print(f"Input tokens: {cost_response['input_tokens']}")
    print(f"Max output tokens: {cost_response['max_output_tokens']}")