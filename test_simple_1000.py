#!/usr/bin/env python3
"""
Lean 1000 sample test using CPU only
"""

import json
import logging
import time
import os
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simple_1000_test():
    """Run simple 1000 sample test without vLLM"""
    
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
    
    logger.info(f"Starting simple 1000 sample test")
    logger.info(f"Created {len(prompts)} test prompts")
    
    # Simulate processing
    start_time = time.time()
    
    results = []
    for i, prompt in enumerate(prompts):
        # Simple mock response
        response = f"Mock response for: {prompt[:50]}..."
        
        results.append({
            "prompt": prompt,
            "response": response,
            "tokens": len(response.split()),
            "processing_time": 0.001  # 1ms mock
        })
        
        # Progress update every 100 samples
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/1000 samples")
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    total_tokens = sum(r["tokens"] for r in results)
    throughput = len(results) / total_time if total_time > 0 else 0
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    # Output results
    result = {
        "success": True,
        "throughput": throughput,
        "tokens_per_sec": tokens_per_sec,
        "total_time": total_time,
        "num_results": len(results),
        "total_tokens": total_tokens,
        "test_type": "simple_cpu_mock",
        "samples_processed": len(prompts)
    }
    
    logger.info("1000 sample test completed successfully!")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Throughput: {throughput:.2f} req/s")
    logger.info(f"Tokens/sec: {tokens_per_sec:.2f}")
    logger.info(f"Processed {len(results)} samples")
    
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    run_simple_1000_test()