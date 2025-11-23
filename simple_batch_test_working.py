#!/usr/bin/env python3
"""
Working Batch Inference Test for Ray Data LLM
Fixed variable scoping and Ray initialization issues
"""

import json
import logging
import time
from typing import Dict, List

import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test_config(model_name: str, batch_size: int, concurrency: int, expected_impact: str, num_samples: int = 20):
    """Run single test configuration"""
    config = {
        "model": {
            "name": model_name,
            "max_model_len": 32768 if "0.5B" in model_name else 8192,
            "tensor_parallel_size": 1,
        },
        "inference": {
            "batch_size": batch_size,
            "concurrency": concurrency,
            "gpu_memory_utilization": 0.8,
            "temperature": 0.7,
            "max_tokens": 64,  # Shorter for faster testing
        },
        "data": {
            "input_path": "/tmp/test_input.json",
            "output_path": "/tmp/test_output",
            "num_samples": num_samples,
        },
        "sla": {
            "target_hours": 1.0,
            "buffer_factor": 0.7,
        }
    }
    
    result = {
        "model": model_name,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "expected_impact": expected_impact,
        "success": False,
        "throughput": 0.0,
        "tokens_per_sec": 0.0,
        "total_time": 0.0,
        "error": ""
    }
    
    try:
        # Check if Ray is already initialized
        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)
        
        # Check GPU resources
        resources = ray.cluster_resources()
        gpu_count = resources.get("GPU", 0)
        logger.info(f"GPU resources available: {gpu_count}")
        
        if gpu_count == 0:
            result["error"] = "No GPUs available"
            return result
        
        # Create test data with correct message format
        prompts = [
            "What is AI?",
            "Explain ML.",
            "How do NNs work?",
            "What is DL?",
            "Describe CV.",
        ] * (num_samples // 5 + 1)
        
        prompts = prompts[:num_samples]
        
        # Convert to message format expected by Ray Data LLM
        def create_message_data(prompt: str) -> Dict:
            return {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
        
        message_data = [create_message_data(p) for p in prompts]
        ds = ray.data.from_items(message_data)
        
        logger.info(f"Created dataset with {ds.count()} samples")
        
        # Configure vLLM processor (no preprocess needed since we provide messages)
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
        
        # Simple postprocess function
        def postprocess_row(row: Dict) -> Dict:
            generated_text = row.get("generated_text", "")
            tokens = len(generated_text.split()) * 1.3
            return {
                "messages": row.get("messages", []),
                "response": generated_text,
                "tokens": int(tokens),
            }
        
        # Build processor
        processor = build_llm_processor(
            vllm_config,
            postprocess=postprocess_row,
        )
        
        # Run inference
        logger.info("Starting inference...")
        start_time = time.time()
        
        result_ds = processor(ds)
        results = result_ds.take_all()
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        total_tokens = sum(r.get("tokens", 0) for r in results)
        throughput = len(results) / inference_time if inference_time > 0 else 0
        tokens_per_sec = total_tokens / inference_time if inference_time > 0 else 0
        
        result.update({
            "success": True,
            "throughput": throughput,
            "tokens_per_sec": tokens_per_sec,
            "total_time": inference_time,
            "num_results": len(results),
            "total_tokens": total_tokens
        })
        
        logger.info(f"Test completed successfully! Throughput: {throughput:.2f} req/s")
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Test failed: {e}")
    
    return result


def main():
    """Run test matrix"""
    print("🚀 Starting Batch Inference Test Matrix")
    
    test_configs = [
        ("Qwen/Qwen2.5-0.5B-Instruct", 128, 2, "Baseline"),
        ("Qwen/Qwen2.5-0.5B-Instruct", 256, 2, "+30-50% throughput"),
        ("Qwen/Qwen2.5-0.5B-Instruct", 128, 4, "Test scaling"),
        ("Qwen/Qwen2.5-7B-Instruct", 64, 2, "Validate larger model"),
        ("Qwen/Qwen2.5-13B-Instruct", 32, 2, "Test memory limits"),
    ]
    
    results = []
    
    for i, (model, batch_size, concurrency, expected) in enumerate(test_configs, 1):
        print(f"\\n📊 Test {i}/{len(test_configs)}")
        print(f"Model: {model}")
        print(f"Batch Size: {batch_size}, Concurrency: {concurrency}")
        print(f"Expected: {expected}")
        
        result = run_test_config(model, batch_size, concurrency, expected)
        results.append(result)
        
        print(f"Result: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        if result['success']:
            print(f"  Throughput: {result['throughput']:.2f} req/s")
            print(f"  Tokens/sec: {result['tokens_per_sec']:.2f}")
        else:
            print(f"  Error: {result['error']}")
        
        # Small delay between tests
        time.sleep(2)
    
    # Generate report
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    report = {
        "summary": {
            "total_tests": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
        },
        "results": results
    }
    
    # Save report
    with open("/tmp/batch_inference_test_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(successful) / len(results) * 100 if results else 0:.1f}%")
    
    if successful:
        best_test = max(successful, key=lambda x: x['throughput'])
        print(f"\\nBest performance:")
        print(f"  Model: {best_test['model']}")
        print(f"  Batch Size: {best_test['batch_size']}")
        print(f"  Concurrency: {best_test['concurrency']}")
        print(f"  Throughput: {best_test['throughput']:.2f} req/s")
    
    if failed:
        print(f"\\nFailed tests:")
        for test in failed:
            print(f"  {test['model']}: {test['error']}")
    
    print(f"\\n📄 Detailed results saved to: /tmp/batch_inference_test_results.json")
    
    return report


if __name__ == "__main__":
    main()