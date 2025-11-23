#!/usr/bin/env python3
"""
Batch Inference Test Runner
Creates test configurations and runs them in Docker environment
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class TestConfig:
    """Test configuration parameters"""
    model_name: str
    batch_size: int
    concurrency: int
    expected_impact: str
    max_model_len: int = 32768
    tensor_parallel_size: int = 1
    num_samples: int = 50


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


class BatchInferenceTestRunner:
    """Manages batch inference testing in Docker"""
    
    def __init__(self, docker_image="michaelsigamani/proj-grounded-telescopes:0.1.0"):
        self.docker_image = docker_image
        self.temp_dir = tempfile.mkdtemp()
        self.results: List[TestResult] = []
        
    def create_test_configs(self) -> List[TestConfig]:
        """Create test matrix configurations"""
        return [
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
                max_model_len=8192,
            ),
            TestConfig(
                model_name="Qwen/Qwen2.5-13B-Instruct",
                batch_size=32,
                concurrency=2,
                expected_impact="Test memory limits",
                max_model_len=4096,
            ),
        ]
    
    def create_docker_test_script(self, config: TestConfig) -> str:
        """Create Python script to run inside Docker container"""
        script_content = f'''#!/usr/bin/env python3
"""
Docker test script for configuration: {config.model_name}
Batch Size: {config.batch_size}, Concurrency: {config.concurrency}
"""

import json
import logging
import os
import time
from typing import Dict

import ray
import yaml
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    """Run single test configuration"""
    config = {{
        "model": {{
            "name": "{config.model_name}",
            "max_model_len": {config.max_model_len},
            "tensor_parallel_size": {config.tensor_parallel_size},
        }},
        "inference": {{
            "batch_size": {config.batch_size},
            "concurrency": {config.concurrency},
            "gpu_memory_utilization": 0.85,
            "temperature": 0.7,
            "max_tokens": 128,
        }},
        "data": {{
            "input_path": "/tmp/test_input.json",
            "output_path": "/tmp/test_output",
            "num_samples": {config.num_samples},
        }},
        "sla": {{
            "target_hours": 1.0,
            "buffer_factor": 0.7,
        }}
    }}
    
    try:
        # Initialize Ray
        ray.init(num_gpus=1, ignore_reinit_error=True)
        
        # Check GPU resources
        resources = ray.cluster_resources()
        gpu_count = resources.get("GPU", 0)
        if gpu_count == 0:
            raise Exception("No GPUs available")
        
        logger.info(f"GPU resources available: {{gpu_count}}")
        
        # Create test data
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning.",
            "How do neural networks work?",
            "What is deep learning?",
            "Describe computer vision.",
        ] * ({config.num_samples} // 5 + 1)
        
        prompts = prompts[:{config.num_samples}]
        ds = ray.data.from_items([{{"prompt": prompt}} for prompt in prompts])
        
        logger.info(f"Created dataset with {{ds.count()}} samples")
        
        # Preprocess function
        def preprocess_row(row: Dict, cfg: Dict) -> Dict:
            return {{
                "prompt": row["prompt"],
                "sampling_params": {{
                    "temperature": cfg["inference"]["temperature"],
                    "max_tokens": cfg["inference"]["max_tokens"],
                    "top_p": 0.9,
                }},
            }}
        
        # Postprocess function
        def postprocess_row(row: Dict) -> Dict:
            generated_text = row.get("generated_text", "")
            tokens = len(generated_text.split()) * 1.3
            return {{
                "prompt": row.get("prompt", ""),
                "response": generated_text,
                "tokens": int(tokens),
            }}
        
        # Configure vLLM processor
        vllm_config = vLLMEngineProcessorConfig(
            model_source=config["model"]["name"],
            concurrency=config["inference"]["concurrency"],
            batch_size=config["inference"]["batch_size"],
            engine_kwargs={{
                "max_model_len": config["model"]["max_model_len"],
                "gpu_memory_utilization": config["inference"]["gpu_memory_utilization"],
                "tensor_parallel_size": config["model"]["tensor_parallel_size"],
                "trust_remote_code": True,
                "enable_chunked_prefill": True,
            }},
        )
        
        # Create processor
        preprocess_fn = lambda row: preprocess_row(row, config)
        postprocess_fn = lambda row: postprocess_row(row)
        
        processor = build_llm_processor(
            vllm_config,
            preprocess=preprocess_fn,
            postprocess=postprocess_fn,
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
        
        # Output results as JSON
        result = {{
            "success": True,
            "throughput": throughput,
            "tokens_per_sec": tokens_per_sec,
            "total_time": inference_time,
            "num_results": len(results),
            "total_tokens": total_tokens,
            "config": {{
                "model": "{config.model_name}",
                "batch_size": {config.batch_size},
                "concurrency": {config.concurrency},
                "expected_impact": "{config.expected_impact}",
            }}
        }}
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {{
            "success": False,
            "error": str(e),
            "config": {{
                "model": "{config.model_name}",
                "batch_size": {config.batch_size},
                "concurrency": {config.concurrency},
                "expected_impact": "{config.expected_impact}",
            }}
        }}
        print(json.dumps(error_result))

if __name__ == "__main__":
    run_test()
'''
        
        script_path = os.path.join(self.temp_dir, f"test_{config.batch_size}_{config.concurrency}.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def run_docker_test(self, config: TestConfig) -> TestResult:
        """Run single test in Docker container"""
        print(f"\\n{'='*60}")
        print(f"Testing: {config.model_name}")
        print(f"Batch Size: {config.batch_size}, Concurrency: {config.concurrency}")
        print(f"Expected: {config.expected_impact}")
        print(f"{'='*60}")
        
        # Create test script
        script_path = self.create_docker_test_script(config)
        
        try:
            # Run Docker command
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{script_path}:/tmp/test_script.py",
                self.docker_image,
                "python", "/tmp/test_script.py"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            total_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse JSON output
                try:
                    output = json.loads(result.stdout.strip())
                    if output.get("success", False):
                        print(f"✅ Test completed successfully!")
                        print(f"   Throughput: {output['throughput']:.2f} req/s")
                        print(f"   Tokens/sec: {output['tokens_per_sec']:.2f}")
                        print(f"   Total time: {output['total_time']:.2f}s")
                        
                        return TestResult(
                            config=config,
                            throughput=output["throughput"],
                            tokens_per_sec=output["tokens_per_sec"],
                            total_time=output["total_time"],
                            success=True
                        )
                    else:
                        error_msg = output.get("error", "Unknown error")
                        print(f"❌ Test failed: {error_msg}")
                        return TestResult(
                            config=config,
                            throughput=0.0,
                            tokens_per_sec=0.0,
                            total_time=total_time,
                            success=False,
                            error_message=error_msg
                        )
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse output: {e}")
                    print(f"Output: {result.stdout}")
                    return TestResult(
                        config=config,
                        throughput=0.0,
                        tokens_per_sec=0.0,
                        total_time=total_time,
                        success=False,
                        error_message=f"JSON parse error: {e}"
                    )
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                print(f"❌ Docker command failed: {error_msg}")
                return TestResult(
                    config=config,
                    throughput=0.0,
                    tokens_per_sec=0.0,
                    total_time=total_time,
                    success=False,
                    error_message=error_msg
                )
                
        except subprocess.TimeoutExpired:
            print(f"❌ Test timed out after 10 minutes")
            return TestResult(
                config=config,
                throughput=0.0,
                tokens_per_sec=0.0,
                total_time=600.0,
                success=False,
                error_message="Test timed out after 10 minutes"
            )
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return TestResult(
                config=config,
                throughput=0.0,
                tokens_per_sec=0.0,
                total_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def run_test_matrix(self) -> List[TestResult]:
        """Run complete test matrix"""
        print("🚀 Starting Batch Inference Test Matrix in Docker")
        
        configs = self.create_test_configs()
        results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\\n📊 Test {i}/{len(configs)}")
            result = self.run_docker_test(config)
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
                "success_rate": len(successful_tests) / len(self.results) * 100 if self.results else 0,
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
        print(f"\\n📄 Test report saved to: {filepath}")


if __name__ == "__main__":
    # Run test matrix
    runner = BatchInferenceTestRunner()
    results = runner.run_test_matrix()
    
    # Generate and save report
    report_path = "batch_inference_test_report.json"
    runner.save_report(report_path)
    
    # Print summary
    print(f"\\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        best_test = max(successful, key=lambda x: x.throughput)
        print(f"\\nBest performance:")
        print(f"  Model: {best_test.config.model_name}")
        print(f"  Batch Size: {best_test.config.batch_size}")
        print(f"  Concurrency: {best_test.config.concurrency}")
        print(f"  Throughput: {best_test.throughput:.2f} req/s")
    
    if failed:
        print(f"\\nFailed tests:")
        for test in failed:
            print(f"  {test.config.model_name}: {test.error_message}")