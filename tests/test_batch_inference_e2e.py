#!/usr/bin/env python3
"""
End-to-End Batch Inference Test Suite
Tests Ray Data + vLLM configurations systematically
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import ray
import yaml
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig


@dataclass
class TestConfig:
    """Test configuration parameters"""
    model_name: str
    batch_size: int
    concurrency: int
    expected_impact: str
    max_model_len: int = 32768
    tensor_parallel_size: int = 1
    num_samples: int = 50  # Small for testing


@dataclass
class TestResult:
    """Test execution results"""
    config: TestConfig
    throughput: float
    tokens_per_sec: float
    total_time: float
    success: bool
    error_message: str = ""
    gpu_memory_mb: float = 0.0


class BatchInferenceTester:
    """End-to-end batch inference tester"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.temp_dir = tempfile.mkdtemp()
        
    def create_test_config(self, config: TestConfig) -> Dict:
        """Create configuration dict for test"""
        return {
            "model": {
                "name": config.model_name,
                "max_model_len": config.max_model_len,
                "tensor_parallel_size": config.tensor_parallel_size,
            },
            "inference": {
                "batch_size": config.batch_size,
                "concurrency": config.concurrency,
                "gpu_memory_utilization": 0.85,  # Conservative for testing
                "temperature": 0.7,
                "max_tokens": 128,  # Shorter for faster testing
            },
            "data": {
                "input_path": os.path.join(self.temp_dir, "test_input.json"),
                "output_path": os.path.join(self.temp_dir, "test_output"),
                "num_samples": config.num_samples,
            },
            "sla": {
                "target_hours": 1.0,  # Short SLA for testing
                "buffer_factor": 0.7,
            }
        }
    
    def create_test_data(self, config: TestConfig) -> ray.data.Dataset:
        """Create test dataset"""
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning.",
            "How do neural networks work?",
            "What is deep learning?",
            "Describe computer vision.",
            "What is natural language processing?",
            "Explain reinforcement learning.",
            "What are transformers in AI?",
            "How does GPT work?",
            "What is supervised learning?",
        ] * (config.num_samples // 10 + 1)
        
        # Limit to exact num_samples
        prompts = prompts[:config.num_samples]
        
        # Create Ray dataset
        ds = ray.data.from_items([{"prompt": prompt} for prompt in prompts])
        return ds
    
    def preprocess_row(self, row: Dict, config: Dict) -> Dict:
        """Preprocess function for Ray Data"""
        return {
            "prompt": row["prompt"],
            "sampling_params": {
                "temperature": config["inference"]["temperature"],
                "max_tokens": config["inference"]["max_tokens"],
                "top_p": 0.9,
            },
        }
    
    def postprocess_row(self, row: Dict) -> Dict:
        """Postprocess function for Ray Data"""
        generated_text = row.get("generated_text", "")
        tokens = len(generated_text.split()) * 1.3
        
        return {
            "prompt": row.get("prompt", ""),
            "response": generated_text,
            "tokens": int(tokens),
        }
    
    def run_single_test(self, test_config: TestConfig) -> TestResult:
        """Run a single test configuration"""
        print(f"\n{'='*60}")
        print(f"Testing: {test_config.model_name}")
        print(f"Batch Size: {test_config.batch_size}, Concurrency: {test_config.concurrency}")
        print(f"Expected: {test_config.expected_impact}")
        print(f"{'='*60}")
        
        config = self.create_test_config(test_config)
        start_time = time.time()
        
        try:
            # Initialize Ray if not already running
            if not ray.is_initialized():
                ray.init(num_gpus=1, ignore_reinit_error=True)
            
            # Check GPU resources
            resources = ray.cluster_resources()
            gpu_count = resources.get("GPU", 0)
            if gpu_count == 0:
                return TestResult(
                    config=test_config,
                    throughput=0.0,
                    tokens_per_sec=0.0,
                    total_time=0.0,
                    success=False,
                    error_message="No GPUs available"
                )
            
            # Create dataset
            ds = self.create_test_data(test_config)
            print(f"Created dataset with {ds.count()} samples")
            
            # Configure vLLM processor
            vllm_config = vLLMEngineProcessorConfig(
                model_source=test_config.model_name,
                concurrency=test_config.concurrency,
                batch_size=test_config.batch_size,
                engine_kwargs={
                    "max_model_len": test_config.max_model_len,
                    "gpu_memory_utilization": config["inference"]["gpu_memory_utilization"],
                    "tensor_parallel_size": test_config.tensor_parallel_size,
                    "trust_remote_code": True,
                    "enable_chunked_prefill": True,
                },
            )
            
            # Create processor
            preprocess_fn = lambda row: self.preprocess_row(row, config)
            postprocess_fn = lambda row: self.postprocess_row(row)
            
            processor = build_llm_processor(
                vllm_config,
                preprocess=preprocess_fn,
                postprocess=postprocess_fn,
            )
            
            print("Starting inference...")
            inference_start = time.time()
            
            # Run inference
            result_ds = processor(ds)
            
            # Collect results
            results = result_ds.take_all()
            inference_time = time.time() - inference_start
            
            # Calculate metrics
            total_tokens = sum(r.get("tokens", 0) for r in results)
            throughput = len(results) / inference_time if inference_time > 0 else 0
            tokens_per_sec = total_tokens / inference_time if inference_time > 0 else 0
            
            total_time = time.time() - start_time
            
            print(f"✅ Test completed successfully!")
            print(f"   Throughput: {throughput:.2f} req/s")
            print(f"   Tokens/sec: {tokens_per_sec:.2f}")
            print(f"   Total time: {total_time:.2f}s")
            
            return TestResult(
                config=test_config,
                throughput=throughput,
                tokens_per_sec=tokens_per_sec,
                total_time=total_time,
                success=True
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = str(e)
            print(f"❌ Test failed: {error_msg}")
            
            return TestResult(
                config=test_config,
                throughput=0.0,
                tokens_per_sec=0.0,
                total_time=total_time,
                success=False,
                error_message=error_msg
            )
    
    def run_test_matrix(self) -> List[TestResult]:
        """Run the complete test matrix"""
        print("🚀 Starting End-to-End Batch Inference Test Matrix")
        
        # Test matrix from requirements
        test_configs = [
            TestConfig(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=128,
                concurrency=2,
                expected_impact="Baseline"
            ),
            TestConfig(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=256,
                concurrency=2,
                expected_impact="+30-50% throughput"
            ),
            TestConfig(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=128,
                concurrency=4,
                expected_impact="Test scaling"
            ),
            TestConfig(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                batch_size=64,
                concurrency=2,
                expected_impact="Validate larger model",
                max_model_len=8192,  # Smaller for larger model
                tensor_parallel_size=1,
            ),
            TestConfig(
                model_name="Qwen/Qwen2.5-13B-Instruct",
                batch_size=32,
                concurrency=2,
                expected_impact="Test memory limits",
                max_model_len=4096,  # Even smaller for 13B model
                tensor_parallel_size=1,
            ),
        ]
        
        results = []
        for i, test_config in enumerate(test_configs, 1):
            print(f"\n📊 Test {i}/{len(test_configs)}")
            result = self.run_single_test(test_config)
            results.append(result)
            
            # Small delay between tests
            time.sleep(2)
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        if not self.results:
            return {"error": "No test results available"}
        
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.results) * 100,
            },
            "results": [],
            "performance_analysis": {},
            "recommendations": []
        }
        
        # Add individual test results
        for result in self.results:
            result_dict = {
                "model": result.config.model_name,
                "batch_size": result.config.batch_size,
                "concurrency": result.config.concurrency,
                "expected_impact": result.config.expected_impact,
                "throughput_req_per_sec": result.throughput,
                "tokens_per_sec": result.tokens_per_sec,
                "total_time_sec": result.total_time,
                "success": result.success,
                "error_message": result.error_message,
            }
            report["results"].append(result_dict)
        
        # Performance analysis
        if successful_tests:
            baseline_throughput = None
            for result in successful_tests:
                if "Baseline" in result.config.expected_impact:
                    baseline_throughput = result.throughput
                    break
            
            if baseline_throughput:
                improvements = []
                for result in successful_tests:
                    if "Baseline" not in result.config.expected_impact:
                        improvement = (result.throughput - baseline_throughput) / baseline_throughput * 100
                        improvements.append({
                            "config": f"{result.config.batch_size}/{result.config.concurrency}",
                            "improvement_percent": improvement
                        })
                report["performance_analysis"]["baseline_throughput"] = baseline_throughput
                report["performance_analysis"]["improvements"] = improvements
        
        # Recommendations
        if failed_tests:
            report["recommendations"].append("Some tests failed - check GPU memory and model availability")
        
        if successful_tests:
            best_throughput = max(successful_tests, key=lambda x: x.throughput)
            report["recommendations"].append(
                f"Best performance: {best_throughput.config.model_name} with "
                f"batch_size={best_throughput.config.batch_size}, concurrency={best_throughput.config.concurrency}"
            )
        
        return report
    
    def save_report(self, filepath: str):
        """Save test report to file"""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Test report saved to: {filepath}")


# Pytest test functions
@pytest.fixture(scope="module")
def batch_tester():
    """Create batch inference tester"""
    return BatchInferenceTester()


@pytest.mark.slow
@pytest.mark.e2e
def test_batch_inference_matrix(batch_tester):
    """Test the complete batch inference matrix"""
    # Run test matrix
    results = batch_tester.run_test_matrix()
    
    # Verify we have results for all test configurations
    assert len(results) == 5, f"Expected 5 test results, got {len(results)}"
    
    # Check that at least baseline test succeeded
    baseline_results = [r for r in results if "Baseline" in r.config.expected_impact and r.success]
    assert len(baseline_results) > 0, "Baseline test must succeed"
    
    # Generate and save report
    report_path = os.path.join(batch_tester.temp_dir, "test_report.json")
    batch_tester.save_report(report_path)
    
    # Verify report was created
    assert os.path.exists(report_path), "Test report should be created"
    
    # Return results for further analysis
    return results


@pytest.mark.slow
@pytest.mark.e2e
def test_performance_scaling(batch_tester):
    """Test performance scaling across configurations"""
    results = batch_tester.run_test_matrix()
    
    # Extract successful tests
    successful_tests = [r for r in results if r.success]
    
    if len(successful_tests) < 2:
        pytest.skip("Need at least 2 successful tests for scaling analysis")
    
    # Find baseline
    baseline = None
    for test in successful_tests:
        if "Baseline" in test.config.expected_impact:
            baseline = test
            break
    
    if not baseline:
        pytest.skip("Baseline test not successful")
    
    # Check scaling expectations
    for test in successful_tests:
        if test.config.batch_size > baseline.config.batch_size:
            # Higher batch size should generally improve throughput
            assert test.throughput >= baseline.throughput * 0.8, \
                f"Higher batch size should not significantly reduce throughput"
        
        if test.config.concurrency > baseline.config.concurrency:
            # Higher concurrency should generally improve throughput
            assert test.throughput >= baseline.throughput * 0.8, \
                f"Higher concurrency should not significantly reduce throughput"


if __name__ == "__main__":
    # Run tests manually
    tester = BatchInferenceTester()
    results = tester.run_test_matrix()
    
    # Generate and save report
    report_path = "batch_inference_test_report.json"
    tester.save_report(report_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        best_test = max(successful, key=lambda x: x.throughput)
        print(f"\nBest performance:")
        print(f"  Model: {best_test.config.model_name}")
        print(f"  Batch Size: {best_test.config.batch_size}")
        print(f"  Concurrency: {best_test.config.concurrency}")
        print(f"  Throughput: {best_test.throughput:.2f} req/s")
    
    if failed:
        print(f"\nFailed tests:")
        for test in failed:
            print(f"  {test.config.model_name}: {test.error_message}")