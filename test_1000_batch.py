#!/usr/bin/env python3
"""
Simple 1000 example batch inference test using Ray Data directly
"""

import json
import logging
import time
import os
import sys
from typing import Dict

import ray
import yaml
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_1000_sample_test():
    """Run 1000 sample batch inference test"""
    
    config = {
        "model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_model_len": 32768,
            "tensor_parallel_size": 1,
        },
        "inference": {
            "batch_size": 128,
            "concurrency": 2,
            "gpu_memory_utilization": 0.85,
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
        
        # Preprocess function
        def preprocess_row(row: Dict, cfg: Dict) -> Dict:
            return {
                "prompt": row["prompt"],
                "sampling_params": {
                    "temperature": cfg["inference"]["temperature"],
                    "max_tokens": cfg["inference"]["max_tokens"],
                    "top_p": 0.9,
                },
            }
        
        # Postprocess function
        def postprocess_row(row: Dict) -> Dict:
            generated_text = row.get("generated_text", "")
            tokens = len(generated_text.split()) * 1.3
            return {
                "prompt": row.get("prompt", ""),
                "response": generated_text,
                "tokens": int(tokens),
            }
        
        # Configure vLLM processor
        vllm_config = vLLMEngineProcessorConfig(
            model_source=config["model"]["name"],
            concurrency=config["inference"]["concurrency"],
            batch_size=config["inference"]["batch_size"],
            engine_kwargs={
                "max_model_len": config["model"]["max_model_len"],
                "gpu_memory_utilization": config["inference"]["gpu_memory_utilization"],
                "tensor_parallel_size": config["model"]["tensor_parallel_size"],
                "trust_remote_code": True,
                "enable_chunked_prefill": True,
            },
        )
        
        # Create processor
        logger.info("Building vLLM processor...")
        preprocess_fn = lambda row: preprocess_row(row, config)
        postprocess_fn = lambda row: postprocess_row(row)
        
        processor = build_llm_processor(
            vllm_config,
            preprocess=preprocess_fn,
            postprocess=postprocess_fn,
        )
        
        # Run inference
        logger.info("Starting 1000 sample inference...")
        start_time = time.time()
        
        result_ds = processor(ds)
        results = result_ds.take_all()
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        total_tokens = sum(r.get("tokens", 0) for r in results)
        throughput = len(results) / inference_time if inference_time > 0 else 0
        tokens_per_sec = total_tokens / inference_time if inference_time > 0 else 0
        
        # Output results
        result = {
            "success": True,
            "throughput": throughput,
            "tokens_per_sec": tokens_per_sec,
            "total_time": inference_time,
            "num_results": len(results),
            "total_tokens": total_tokens,
            "config": {
                "model": config["model"]["name"],
                "batch_size": config["inference"]["batch_size"],
                "concurrency": config["inference"]["concurrency"],
                "num_samples": config["data"]["num_samples"],
            }
        }
        
        logger.info("1000 sample batch inference completed successfully!")
        logger.info(f"Total time: {inference_time:.2f} seconds")
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
            }
        }
        print(json.dumps(error_result, indent=2))
        return error_result

if __name__ == "__main__":
    run_1000_sample_test()