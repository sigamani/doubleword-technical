#!/usr/bin/env python3
"""
Configuration Matrix Testing Script
Tests different Ray Data + vLLM configurations for performance analysis
"""

import os
import sys
import time
import json
import yaml
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import ray
from ray import data
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Test configuration parameters"""
    name: str
    model_size: str
    batch_size: int
    concurrency: int
    gpu_memory_utilization: float
    temperature: float
    max_tokens: int
    num_samples: int


@dataclass
class TestResult:
    """Test result metrics"""
    config_name: str
    model_size: str
    batch_size: int
    concurrency: int
    throughput: float
    tokens_per_sec: float
    gpu_memory_peak: float
    success: bool
    error_message: str = ""
    total_time: float = 0.0
    num_requests: int = 0


class ConfigurationMatrix:
    """Configuration matrix testing framework"""
    
    def __init__(self):
        self.configs = self._define_test_configs()
        self.results = []
    
    def _define_test_configs(self) -> List[TestConfig]:
        """Define all test configurations"""
        return [
            # Baseline configuration
            TestConfig(
                name="baseline",
                model_size="0.5B",
                batch_size=128,
                concurrency=2,
                gpu_memory_utilization=0.90,
                temperature=0.7,
                max_tokens=512,
                num_samples=1000
            ),
            
            # Large batch size test
            TestConfig(
                name="large_batch",
                model_size="0.5B", 
                batch_size=256,
                concurrency=2,
                gpu_memory_utilization=0.90,
                temperature=0.7,
                max_tokens=512,
                num_samples=1000
            ),
            
            # High concurrency test
            TestConfig(
                name="high_concurrency",
                model_size="0.5B",
                batch_size=128,
                concurrency=4,
                gpu_memory_utilization=0.90,
                temperature=0.7,
                max_tokens=512,
                num_samples=1000
            ),
            
            # 7B model test
            TestConfig(
                name="7b_model",
                model_size="7B",
                batch_size=64,
                concurrency=2,
                gpu_memory_utilization=0.85,
                temperature=0.7,
                max_tokens=512,
                num_samples=500  # Fewer samples for larger model
            ),
            
            # 13B model test
            TestConfig(
                name="13b_model",
                model_size="13B",
                batch_size=32,
                concurrency=2,
                gpu_memory_utilization=0.80,
                temperature=0.7,
                max_tokens=512,
                num_samples=200  # Even fewer samples for large model
            ),
        ]
    
    def run_test(self, config: TestConfig) -> TestResult:
        """Run a single test configuration"""
        logger.info(f"Running test: {config.name}")
        
        try:
            # Initialize Ray
            ray.init(address="local", _redis_password="ray123")
            
            # Create test dataset
            ds = data.from_items([
                {"prompt": f"Test prompt {i} for batch inference testing."}
                for i in range(config.num_samples)
            ])
            
            # Configure vLLM processor
            model_name = {
                "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
                "7B": "Qwen/Qwen2.5-7B-Instruct", 
                "13B": "Qwen/Qwen2.5-13B-Instruct"
            }[config.model_size]
            
            vllm_config = vLLMEngineProcessorConfig(
                model_source=model_name,
                batch_size=config.batch_size,
                concurrency=config.concurrency,
                engine_kwargs={
                    "max_model_len": 16384,
                    "gpu_memory_utilization": config.gpu_memory_utilization,
                    "tensor_parallel_size": 1,
                    "trust_remote_code": True,
                    "enable_chunked_prefill": True,
                }
            )
            
            # Preprocess and postprocess functions
            def preprocess(row):
                return {
                    "prompt": row["prompt"],
                    "sampling_params": {
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "top_p": 0.9,
                    }
                }
            
            def postprocess(row):
                generated_text = row.get("generated_text", "")
                tokens = len(generated_text.split()) * 1.3
                return {
                    "prompt": row.get("prompt", ""),
                    "response": generated_text,
                    "tokens": int(tokens),
                }
            
            # Build processor
            processor = build_llm_processor(
                vllm_config,
                preprocess=preprocess,
                postprocess=postprocess
            )
            
            # Run inference and measure
            start_time = time.time()
            result_ds = processor(ds)
            
            # Collect results and calculate metrics
            results = []
            total_tokens = 0
            
            for batch in result_ds.iter_batches():
                for row in batch:
                    results.append(row)
                    total_tokens += row.get("tokens", 0)
            
            total_time = time.time() - start_time
            throughput = config.num_samples / total_time if total_time > 0 else 0
            tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            
            # Estimate GPU memory usage
            gpu_memory_peak = config.batch_size * config.concurrency * 0.001  # Rough estimate
            
            success = True
            
            logger.info(f"Test {config.name} completed successfully")
            
            return TestResult(
                config_name=config.name,
                model_size=config.model_size,
                batch_size=config.batch_size,
                concurrency=config.concurrency,
                throughput=throughput,
                tokens_per_sec=tokens_per_sec,
                gpu_memory_peak=gpu_memory_peak,
                success=success,
                total_time=total_time,
                num_requests=config.num_samples
            )
            
        except Exception as e:
            logger.error(f"Test {config.name} failed: {e}")
            return TestResult(
                config_name=config.name,
                model_size=config.model_size,
                batch_size=config.batch_size,
                concurrency=config.concurrency,
                throughput=0.0,
                tokens_per_sec=0.0,
                gpu_memory_peak=0.0,
                success=False,
                error_message=str(e),
                total_time=0.0,
                num_requests=config.num_samples
            )
        
        finally:
            ray.shutdown()
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all test configurations"""
        logger.info("Starting configuration matrix testing")
        results = []
        
        for config in self.configs:
            result = self.run_test(config)
            results.append(result)
            
            # Brief pause between tests
            time.sleep(2)
        
        logger.info(f"Completed {len(results)} tests")
        return results
    
    def save_results(self, results: List[TestResult], output_dir: str = "matrix_results"):
        """Save test results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        results_data = [asdict(r) for r in results]
        with open(f"{output_dir}/detailed_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Generate HTML report
        html_content = self._generate_html_report(results)
        with open(f"{output_dir}/matrix_report.html", "w") as f:
            f.write(html_content)
        
        logger.info(f"Results saved to {output_dir}/")
    
    def _generate_html_report(self, results: List[TestResult]) -> str:
        """Generate HTML report for test results"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Configuration Matrix Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; color: white; padding: 20px; border-radius: 5px; }
        .test-result { margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { border-left: 5px solid #28a745; }
        .failure { border-left: 5px solid #dc3545; }
        .metrics { margin-top: 10px; }
        .metric { display: inline-block; margin: 5px; padding: 5px; background-color: #f8f9fa; border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .better { color: #28a745; font-weight: bold; }
        .worse { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Configuration Matrix Test Results</h1>
        <p>Tests completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        # Results table
        html += """
    <table>
        <tr>
            <th>Configuration</th>
            <th>Model Size</th>
            <th>Batch Size</th>
            <th>Concurrency</th>
            <th>Throughput (req/s)</th>
            <th>Tokens/sec</th>
            <th>GPU Memory (GB)</th>
            <th>Status</th>
            <th>Time (s)</th>
        </tr>
"""
        
        baseline_result = next((r for r in results if r.config_name == "baseline"), None)
        
        for result in results:
            status_class = "success" if result.success else "failure"
            
            # Compare with baseline
            throughput_class = ""
            if baseline_result and result.throughput > 0:
                if result.throughput > baseline_result.throughput * 1.1:
                    throughput_class = "better"
                elif result.throughput < baseline_result.throughput * 0.9:
                    throughput_class = "worse"
            
            html += f"""
        <tr class="{status_class}">
            <td>{result.config_name}</td>
            <td>{result.model_size}</td>
            <td>{result.batch_size}</td>
            <td>{result.concurrency}</td>
            <td>{result.throughput:.2f}</td>
            <td>{result.tokens_per_sec:.2f}</td>
            <td>{result.gpu_memory_peak:.2f}</td>
            <td>{result.success}</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
        
        html += """
    </table>
    
    <div class="metrics">
        <h2>Performance Analysis</h2>
"""
        
        if baseline_result:
            html += f"""
        <div class="metric">
            <h3>Baseline Performance</h3>
            <p>Configuration: {baseline_result.config_name}</p>
            <p>Throughput: {baseline_result.throughput:.2f} req/s</p>
            <p>Tokens/sec: {baseline_result.tokens_per_sec:.2f}</p>
        </div>
"""
        
        # Find best configuration
        best_result = max((r for r in results if r.success), key=lambda x: x.throughput)
        if best_result:
            html += f"""
        <div class="metric">
            <h3>Best Performance</h3>
            <p>Configuration: {best_result.config_name}</p>
            <p>Throughput: {best_result.throughput:.2f} req/s</p>
            <p>Tokens/sec: {best_result.tokens_per_sec:.2f}</p>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        return html


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python run_matrix_tests.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    # Create matrix tester and run tests
    matrix = ConfigurationMatrix()
    results = matrix.run_all_tests()
    
    # Save results
    matrix.save_results(results, output_dir)
    
    # Print summary
    successful_tests = [r for r in results if r.success]
    print(f"\nCompleted {len(successful_tests)}/{len(results)} tests successfully")
    
    if successful_tests:
        best_test = max(successful_tests, key=lambda x: x.throughput)
        print(f"Best configuration: {best_test.config_name} ({best_test.throughput:.2f} req/s)")
    
    failed_tests = [r for r in results if not r.success]
    if failed_tests:
        print(f"Failed tests: {[r.config_name for r in failed_tests]}")


if __name__ == "__main__":
    main()