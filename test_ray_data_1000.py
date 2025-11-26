#!/usr/bin/env python3
"""
Lean Ray Data test with 1000 examples (CPU only)
"""

import json
import logging
import time
import os
from typing import List, Dict

import ray
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ray_data_1000_test():
    """Run 1000 sample Ray Data test without vLLM"""
    
    config = {
        "model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_model_len": 32768,
            "tensor_parallel_size": 1,
        },
        "inference": {
            "batch_size": 128,
            "concurrency": 2,
            "temperature": 0.7,
            "max_tokens": 128,
        },
        "data": {
            "input_path": "/tmp/test_input.json",
            "output_path": "/tmp/test_output",
            "num_samples": 1000,
        },
        "sla": {
            "target_hours": 1.0,
            "buffer_factor": 0.7,
        }
    }
    
    try:
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Check resources
        resources = ray.cluster_resources()
        logger.info(f"Available resources: {resources}")
        
        # Create 1000 test prompts
        base_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning.",
            "How do neural networks work?",
            "What is deep learning?",
            "Describe computer vision.",
            "What is natural language processing?",
            "How does reinforcement learning work?",
            "What are transformers in AI?",
            "Explain convolutional neural networks.",
            "What is supervised learning?"
        ]
        
        # Create 1000 prompts
        prompts = base_prompts * 100  # 1000 prompts
        prompts = prompts[:1000]
        
        logger.info(f"Created {len(prompts)} test prompts")
        
        # Create Ray Dataset
        ds = ray.data.from_items([{"prompt": prompt} for prompt in prompts])
        logger.info(f"Created dataset with {ds.count()} samples")
        
        # Simple preprocessing function
        def preprocess_batch(batch):
            """Simple preprocessing"""
            processed = []
            for item in batch:
                # Handle both dict and string cases
                if isinstance(item, dict):
                    prompt = item.get("prompt", str(item))
                else:
                    prompt = str(item)
                    
                processed.append({
                    "prompt": prompt,
                    "processed_text": f"Processed: {prompt[:50]}...",
                    "timestamp": time.time()
                })
            return {"preprocessed": processed}
        
        # Simple postprocessing function  
        def postprocess_batch(batch):
            """Simple postprocessing"""
            processed = []
            for item in batch["preprocessed"]:
                # Mock inference result
                mock_response = f"This is a mock response for: {item['prompt'][:30]}..."
                processed.append({
                    "prompt": item.get("prompt", ""),
                    "response": mock_response,
                    "tokens": len(mock_response.split()),
                    "processing_time": 0.001
                })
            return {"results": processed}
        
        # Run Ray Data processing pipeline
        logger.info("Starting Ray Data processing...")
        start_time = time.time()
        
        # Apply preprocessing
        preprocessed_ds = ds.map_batches(
            preprocess_batch,
            batch_size=config["inference"]["batch_size"]
        )
        
        # Apply postprocessing
        result_ds = preprocessed_ds.map_batches(
            postprocess_batch,
            batch_size=config["inference"]["batch_size"]
        )
        
        # Get results
        results = result_ds.take_all()
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        total_tokens = sum(r.get("tokens", 0) for r in results)
        throughput = len(results) / processing_time if processing_time > 0 else 0
        tokens_per_sec = total_tokens / processing_time if processing_time > 0 else 0
        
        # Output results
        result = {
            "success": True,
            "throughput": throughput,
            "tokens_per_sec": tokens_per_sec,
            "total_time": processing_time,
            "num_results": len(results),
            "total_tokens": total_tokens,
            "config": {
                "model": config["model"]["name"],
                "batch_size": config["inference"]["batch_size"],
                "concurrency": config["inference"]["concurrency"],
                "num_samples": config["data"]["num_samples"],
                "test_type": "ray_data_cpu_only"
            }
        }
        
        logger.info("1000 sample Ray Data test completed successfully!")
        logger.info(f"Total time: {processing_time:.2f} seconds")
        logger.info(f"Throughput: {throughput:.2f} req/s")
        logger.info(f"Tokens/sec: {tokens_per_sec:.2f}")
        logger.info(f"Processed {len(results)} samples")
        
        print(json.dumps(result, indent=2))
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "config": {
                "model": config["model"]["name"],
                "batch_size": config["inference"]["batch_size"],
                "concurrency": config["inference"]["concurrency"],
                "num_samples": config["data"]["num_samples"],
                "test_type": "ray_data_cpu_only"
            }
        }
        print(json.dumps(error_result, indent=2))
        return error_result

if __name__ == "__main__":
    run_ray_data_1000_test()