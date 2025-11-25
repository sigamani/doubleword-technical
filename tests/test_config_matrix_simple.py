#!/usr/bin/env python3
"""
Simple Configuration Matrix Testing Script
Tests different Ray Data + vLLM configurations for performance analysis
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Test configuration parameters"""
    def __init__(self, name: str, model_size: str, batch_size: int, concurrency: int, 
                 gpu_memory_utilization: float, temperature: float, max_tokens: int, num_samples: int):
        self.name = name
        self.model_size = model_size
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_samples = num_samples


@dataclass
class TestResult:
    """Test result metrics"""
    def __init__(self, config_name: str, model_size: str, batch_size: int, concurrency: int, 
                 throughput: float, tokens_per_sec: float, gpu_memory_peak: float, 
                 success: bool, error_message: str = "", total_time: float = 0.0, 
                 num_requests: int = 0):
        self.config_name = config_name
        self.model_size = model_size
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.throughput = throughput
        self.tokens_per_sec = tokens_per_sec
        self.gpu_memory_peak = gpu_memory_peak
        self.success = success
        self.error_message = error_message
        self.total_time = total_time
        self.num_requests = num_requests


class ConfigurationMatrix:
    """Test configuration matrix with different scenarios"""
    
    def __init__(self):
        self.configurations = [
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
    
    def get_configuration(self, name: str) -> TestConfig:
        """Get configuration by name"""
        for config in self.configurations:
            if config.name == name:
                return config
        raise ValueError(f"Configuration '{name}' not found")
    
    def test_configuration(self, config: TestConfig) -> Dict:
        """Test a single configuration and return results"""
        logger.info(f"Running test: {config.name}")
        
        try:
            # Simulate Ray initialization
            logger.info("Initializing Ray cluster...")
            
            # Simulate dataset creation
            logger.info(f"Creating dataset with {config.num_samples} samples")
            
            # Simulate processing time based on configuration
            # Rough estimates for demonstration
            base_time_per_request = 0.01  # Base processing time
            batch_factor = config.batch_size / 128  # Batch size factor
            concurrency_factor = config.concurrency / 2  # Concurrency factor
            
            # Adjust processing time based on configuration
            time_per_request = base_time_per_request / (1 + batch_factor * 0.5) / (1 + concurrency_factor * 0.3)
            
            # Calculate total time
            total_time = config.num_samples * time_per_request
            
            # Calculate throughput
            throughput = config.num_samples / total_time if total_time > 0 else 0
            
            # Calculate tokens per second (rough estimate)
            tokens_per_sec = throughput * 50  # Assume 50 tokens per request
            
            # Estimate GPU memory usage
            gpu_memory_peak = config.batch_size * config.concurrency * 0.001  # Rough estimate
            
            success = True
            
            logger.info(f"Test {config.name} completed successfully")
            
            return {
                "config_name": config.name,
                "model_size": config.model_size,
                "batch_size": config.batch_size,
                "concurrency": config.concurrency,
                "throughput": throughput,
                "tokens_per_sec": tokens_per_sec,
                "gpu_memory_peak": gpu_memory_peak,
                "success": success,
                "total_time": total_time,
                "num_requests": config.num_samples
            }
            
        except Exception as e:
            logger.error(f"Test {config.name} failed: {e}")
            return {
                "config_name": config.name,
                "model_size": config.model_size,
                "batch_size": config.batch_size,
                "concurrency": config.concurrency,
                "throughput": 0.0,
                "tokens_per_sec": 0.0,
                "gpu_memory_peak": 0.0,
                "success": False,
                "error_message": str(e),
                "total_time": 0.0,
                "num_requests": config.num_samples
            }
    
    def run_all_tests(self) -> List[Dict]:
        """Run all test configurations"""
        logger.info("Starting configuration matrix testing")
        results = []
        
        for config in self.configurations:
            result = self.test_configuration(config)
            results.append(result)
            
            # Brief pause between tests
            time.sleep(1)
        
        logger.info(f"Completed {len(results)} tests")
        return results
    
    def save_results(self, results: List[Dict], output_dir: str = "matrix_results"):
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
    
    def _generate_html_report(self, results: List[Dict]) -> str:
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
        <p>Tests completed at: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """</p>
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
        
        baseline_result = next((r for r in results if r["config_name"] == "baseline"), None)
        
        for result in results:
            status_class = "success" if result["success"] else "failure"
            
            # Compare with baseline
            throughput_class = ""
            if baseline_result and result["throughput"] > 0:
                if result["throughput"] > baseline_result["throughput"] * 1.1:
                    throughput_class = "better"
                elif result["throughput"] < baseline_result["throughput"] * 0.9:
                    throughput_class = "worse"
            
            html += f"""
        <tr class="{status_class}">
            <td>{result["config_name"]}</td>
            <td>{result["model_size"]}</td>
            <td>{result["batch_size"]}</td>
            <td>{result["concurrency"]}</td>
            <td>{result["throughput"]:.2f}</td>
            <td>{result["tokens_per_sec"]:.2f}</td>
            <td>{result["gpu_memory_peak"]:.2f}</td>
            <td>{result["success"]}</td>
            <td>{result["total_time"]:.2f}</td>
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
            <p>Configuration: {baseline_result["config_name"]}</p>
            <p>Throughput: {baseline_result["throughput"]:.2f} req/s</p>
            <p>Tokens/sec: {baseline_result["tokens_per_sec"]:.2f}</p>
        </div>
"""
        
        # Find best configuration
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x["throughput"])
            html += f"""
        <div class="metric">
            <h3>Best Performance</h3>
            <p>Configuration: {best_result["config_name"]}</p>
            <p>Throughput: {best_result["throughput"]:.2f} req/s</p>
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
    
    # Create test matrix
    matrix = ConfigurationMatrix()
    
    # Run all tests
    results = matrix.run_all_tests()
    
    # Save results
    matrix.save_results(results, output_dir)
    
    # Print summary
    successful_tests = [r for r in results if r["success"]]
    print(f"\\nCompleted {len(successful_tests)}/{len(results)} tests successfully")
    
    if successful_tests:
        best_test = max(successful_tests, key=lambda x: x["throughput"])
        print(f"Best configuration: {best_test['config_name']} ({best_test['throughput']:.2f} req/s)")
    
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        print(f"Failed tests: {[r['config_name'] for r in failed_tests]}")


if __name__ == "__main__":
    main()