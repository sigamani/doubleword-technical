#!/usr/bin/env python3
"""
Testing-Symbiote Subagent for sigamani/doubleword-technical
Symbiotic testing agent that runs and repairs the full test matrix after Troels completes his tasks.
"""

import os
import sys
import json
import time
import logging
import subprocess
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_remove_emojis(file_path: str) -> bool:
    """
    Strict emoji check using ruff to detect and remove emojis from source code
    Returns True if emojis were found and removed, False otherwise
    """
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Define emoji pattern (comprehensive)
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U00002702-\U000027B0\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]'
        )
        
        # Find all emojis
        emojis_found = emoji_pattern.findall(content)
        
        if emojis_found:
            logger.warning(f"EMOJI VIOLATION DETECTED: Found {len(emojis_found)} emojis in {file_path}")
            logger.warning("Removing emojis to comply with AGENTS.md cardinal rule...")
            
            # Remove all emojis
            clean_content = emoji_pattern.sub('', content)
            
            # Write back clean content
            with open(file_path, 'w') as f:
                f.write(clean_content)
            
            logger.info(f"Successfully removed {len(emojis_found)} emojis from {file_path}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking emojis in {file_path}: {e}")
        return False

def run_ruff_emoji_check() -> bool:
    """
    Use ruff to check for emoji violations in the current file
    """
    try:
        # Run ruff on this file to check for any issues
        result = subprocess.run(
            ["ruff", "check", "--select=ALL", __file__],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.warning("Ruff found issues in testing_symbiote.py")
            logger.warning(result.stdout)
            return False
        
        logger.info("Ruff check passed - no issues found")
        return True
        
    except FileNotFoundError:
        logger.warning("Ruff not available - skipping ruff check")
        return True
    except Exception as e:
        logger.error(f"Error running ruff check: {e}")
        return False

# Perform strict emoji check on startup
if __name__ != "__main__":
    # Only run checks when imported as module, not when executed directly
    current_file = __file__
    if check_and_remove_emojis(current_file):
        logger.warning("EMOJIS REMOVED - Restarting without emojis")
        # Exit and let the process restart with clean code
        sys.exit(1)
    
    # Run ruff check
    run_ruff_emoji_check()

@dataclass
class TestMatrixConfig:
    """Test matrix configuration for batch size × concurrency × model-size permutations"""
    name: str
    model_name: str
    batch_size: int
    concurrency: int
    max_tokens: int
    num_samples: int
    expected_impact: str
    max_model_len: int = 32768
    tensor_parallel_size: int = 2

class TestingSymbiote:
    """
    Testing Symbiote - Dedicated quality-assurance agent for sigamani/doubleword-technical
    
    Mission: Every time you are invoked, perform the following operations in strict sequence:
    1. Run the complete test matrix (batch size × concurrency × model-size permutations)
    2. If any tests fail:
       - Diagnose which failure modes were introduced
       - Apply targeted fixes to restore functionality
       - Re-run the test matrix to verify repairs
    3. Generate a comprehensive report with performance metrics and recommendations
    """
    
    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or os.getcwd()
        self.test_results: List[Dict[str, Any]] = []
        self.failure_modes: List[Dict[str, Any]] = []
        
    def get_test_matrix_configurations(self) -> List[TestMatrixConfig]:
        """Get the complete testing matrix for batch size × concurrency × model-size permutations"""
        return [
            # Baseline configurations
            TestMatrixConfig(
                name="baseline_0.5b_small",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=64,
                concurrency=1,
                max_tokens=256,
                num_samples=25,
                expected_impact="Baseline small batch test"
            ),
            TestMatrixConfig(
                name="baseline_0.5b_medium", 
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=128,
                concurrency=2,
                max_tokens=512,
                num_samples=50,
                expected_impact="Baseline medium batch test"
            ),
            TestMatrixConfig(
                name="baseline_0.5b_large",
                model_name="Qwen/Qwen2.5-0.5B-Instruct", 
                batch_size=256,
                concurrency=2,
                max_tokens=512,
                num_samples=100,
                expected_impact="Baseline large batch test"
            ),
            
            # High concurrency tests
            TestMatrixConfig(
                name="high_concurrency_0.5b",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=128,
                concurrency=4,
                max_tokens=512,
                num_samples=50,
                expected_impact="Test scaling with higher concurrency"
            ),
            TestMatrixConfig(
                name="max_concurrency_0.5b",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=64,
                concurrency=8,
                max_tokens=256,
                num_samples=25,
                expected_impact="Maximum concurrency stress test"
            ),
            
            # Large model tests
            TestMatrixConfig(
                name="baseline_7b",
                model_name="Qwen/Qwen2.5-7B-Instruct",
                batch_size=32,
                concurrency=1,
                max_tokens=256,
                num_samples=20,
                expected_impact="Validate larger model baseline performance",
                max_model_len=8192,
                tensor_parallel_size=1
            ),
            TestMatrixConfig(
                name="medium_7b",
                model_name="Qwen/Qwen2.5-7B-Instruct",
                batch_size=64,
                concurrency=2,
                max_tokens=256,
                num_samples=25,
                expected_impact="Test larger model with medium batches",
                max_model_len=8192,
                tensor_parallel_size=1
            ),
            
            # Stress tests
            TestMatrixConfig(
                name="stress_batch_size",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=512,
                concurrency=2,
                max_tokens=1024,
                num_samples=100,
                expected_impact="Stress test with maximum batch size"
            ),
            TestMatrixConfig(
                name="stress_concurrency",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=128,
                concurrency=16,
                max_tokens=512,
                num_samples=50,
                expected_impact="Stress test with maximum concurrency"
            ),
            
            # SLA validation tests
            TestMatrixConfig(
                name="sla_validation_24h",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=256,
                concurrency=4,
                max_tokens=512,
                num_samples=1000,
                expected_impact="24-hour SLA validation test"
            )
        ]
    
    def run_test_configuration(self, config: TestMatrixConfig) -> Dict[str, Any]:
        """Run a single test configuration and return results"""
        logger.info(f"Running test configuration: {config.name}")
        start_time = time.time()
        
        try:
            # Create temporary config file for this test
            test_config = {
                "model": {
                    "name": config.model_name,
                    "max_model_len": config.max_model_len,
                    "tensor_parallel_size": config.tensor_parallel_size
                },
                "inference": {
                    "batch_size": config.batch_size,
                    "concurrency": config.concurrency,
                    "max_tokens": config.max_tokens,
                    "temperature": 0.7,
                    "gpu_memory_utilization": 0.90
                },
                "data": {
                    "num_samples": config.num_samples,
                    "output_path": f"/tmp/test_output_{config.name}"
                },
                "sla": {
                    "target_hours": 24,
                    "buffer_factor": 0.7
                }
            }
            
            # Run the batch inference test
            test_script = os.path.join(self.repo_root, "tests", "run_batch_inference_tests.py")
            if not os.path.exists(test_script):
                # Fallback to main test runner
                test_script = os.path.join(self.repo_root, "run_tests.py")
            
            cmd = ["python3", str(test_script), "--config", json.dumps(test_config)]
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Parse output for metrics
            metrics = self._parse_test_output(result.stdout, result.stderr)
            
            return {
                "config_name": config.name,
                "config": config,
                "success": success,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "metrics": metrics,
                "expected_impact": config.expected_impact
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "config_name": config.name,
                "config": config,
                "success": False,
                "duration": duration,
                "error": "Test timed out after 30 minutes",
                "expected_impact": config.expected_impact
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "config_name": config.name,
                "config": config,
                "success": False,
                "duration": duration,
                "error": str(e),
                "expected_impact": config.expected_impact
            }
    
    def _parse_test_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse test output to extract performance metrics"""
        metrics = {
            "tokens_processed": 0,
            "throughput_tokens_per_sec": 0.0,
            "batch_processing_time": 0.0,
            "memory_usage_mb": 0.0,
            "gpu_utilization": 0.0,
            "errors": []
        }
        
        # Parse stdout for metrics
        lines = stdout.split('\n') + stderr.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for token counts
            if "tokens processed" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i > 0:
                            metrics["tokens_processed"] = int(part)
                            break
                except:
                    pass
            
            # Look for throughput
            if "tokens/sec" in line.lower() or "tokens per second" in line.lower():
                try:
                    import re
                    match = re.search(r'(\d+\.?\d*)\s*tokens?/?sec', line.lower())
                    if match:
                        metrics["throughput_tokens_per_sec"] = float(match.group(1))
                except:
                    pass
            
            # Look for memory usage
            if "memory" in line.lower() and "mb" in line.lower():
                try:
                    import re
                    match = re.search(r'(\d+\.?\d*)\s*mb', line.lower())
                    if match:
                        metrics["memory_usage_mb"] = float(match.group(1))
                except:
                    pass
        
        return metrics
    
    def diagnose_failure_modes(self, failed_tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Diagnose failure modes from failed tests"""
        failure_modes = []
        
        for test in failed_tests:
            error_msg = test.get("error", "") or test.get("stderr", "")
            
            # Categorize failure modes
            if "timeout" in error_msg.lower():
                failure_modes.append({
                    "type": "performance_timeout",
                    "severity": "high",
                    "description": f"Test {test['config_name']} timed out",
                    "affected_configs": [test["config_name"]],
                    "recommended_fix": "Reduce batch size or increase timeout"
                })
            elif "memory" in error_msg.lower() or "cuda out of memory" in error_msg.lower():
                failure_modes.append({
                    "type": "memory_exhaustion",
                    "severity": "high", 
                    "description": f"GPU memory exhaustion in {test['config_name']}",
                    "affected_configs": [test["config_name"]],
                    "recommended_fix": "Reduce batch size or model size"
                })
            elif "import" in error_msg.lower() or "module" in error_msg.lower():
                failure_modes.append({
                    "type": "dependency_issue",
                    "severity": "medium",
                    "description": f"Import error in {test['config_name']}",
                    "affected_configs": [test["config_name"]],
                    "recommended_fix": "Install missing dependencies"
                })
            elif "ray" in error_msg.lower() and "connection" in error_msg.lower():
                failure_modes.append({
                    "type": "ray_cluster_issue",
                    "severity": "high",
                    "description": f"Ray cluster connection issue in {test['config_name']}",
                    "affected_configs": [test["config_name"]],
                    "recommended_fix": "Check Ray cluster status and connectivity"
                })
            else:
                failure_modes.append({
                    "type": "unknown_error",
                    "severity": "medium",
                    "description": f"Unknown error in {test['config_name']}: {error_msg[:100]}",
                    "affected_configs": [test["config_name"]],
                    "recommended_fix": "Manual investigation required"
                })
        
        return failure_modes
    
    def apply_targeted_fixes(self, failure_modes: List[Dict[str, Any]]) -> List[str]:
        """Apply targeted fixes based on failure modes"""
        fixes_applied = []
        
        for failure_mode in failure_modes:
            fix_type = failure_mode["type"]
            
            if fix_type == "dependency_issue":
                # Try to install common missing dependencies
                common_deps = ["ray[data]", "vllm", "torch", "transformers", "pyyaml"]
                for dep in common_deps:
                    try:
                        subprocess.run(["pip", "install", dep], capture_output=True, check=True)
                        fixes_applied.append(f"Installed {dep}")
                    except:
                        pass
                        
            elif fix_type == "memory_exhaustion":
                # Create optimized config for memory-constrained tests
                optimized_config = {
                    "inference": {
                        "batch_size": 32,
                        "gpu_memory_utilization": 0.7,
                        "max_model_len": 4096
                    }
                }
                config_path = os.path.join(self.repo_root, "config", "optimized_memory.yaml")
                try:
                    import yaml
                    with open(config_path, 'w') as f:
                        yaml.dump(optimized_config, f)
                    fixes_applied.append("Created memory-optimized configuration")
                except:
                    fixes_applied.append("Failed to create memory-optimized config")
            
            elif fix_type == "ray_cluster_issue":
                # Try to restart Ray cluster
                try:
                    subprocess.run(["ray", "stop", "--force"], capture_output=True)
                    subprocess.run(["ray", "start", "--head"], capture_output=True)
                    fixes_applied.append("Restarted Ray cluster")
                except:
                    fixes_applied.append("Failed to restart Ray cluster")
        
        return fixes_applied
    
    def run_complete_test_matrix(self) -> Dict[str, Any]:
        """Run the complete test matrix following the strict sequence"""
        logger.info("Starting Testing Symbiote - Complete Test Matrix Execution")
        
        # Step 1: Run the complete test matrix
        configurations = self.get_test_matrix_configurations()
        all_results = []
        
        for config in configurations:
            result = self.run_test_configuration(config)
            all_results.append(result)
            self.test_results.append(result)
            
            status = "PASS" if result["success"] else "FAIL"
            logger.info(f"Configuration {config.name}: {status} ({result['duration']:.2f}s)")
        
        # Step 2: Diagnose failures if any
        failed_tests = [r for r in all_results if not r["success"]]
        passed_tests = [r for r in all_results if r["success"]]
        
        if failed_tests:
            logger.warning(f"Found {len(failed_tests)} failed tests, diagnosing failure modes...")
            self.failure_modes = self.diagnose_failure_modes(failed_tests)
            
            # Apply targeted fixes
            logger.info("Applying targeted fixes...")
            fixes_applied = self.apply_targeted_fixes(self.failure_modes)
            
            if fixes_applied:
                logger.info(f"Applied {len(fixes_applied)} fixes: {', '.join(fixes_applied)}")
                
                # Re-run failed tests to verify repairs
                logger.info("Re-running failed tests to verify repairs...")
                for failed_test in failed_tests:
                    retry_result = self.run_test_configuration(failed_test["config"])
                    retry_result["original_failure"] = failed_test
                    all_results.append(retry_result)
                    
                    if retry_result["success"]:
                        logger.info(f" Successfully repaired {failed_test['config_name']}")
                    else:
                        logger.error(f" Failed to repair {failed_test['config_name']}")
        
        # Step 3: Generate comprehensive report
        report = self.generate_comprehensive_report(all_results)
        return report
    
    def generate_comprehensive_report(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive report with performance metrics and recommendations"""
        
        # Separate original and retry results
        original_results = [r for r in all_results if "original_failure" not in r]
        retry_results = [r for r in all_results if "original_failure" in r]
        
        passed_original = [r for r in original_results if r["success"]]
        failed_original = [r for r in original_results if not r["success"]]
        
        # Calculate performance metrics
        total_tokens = sum(r.get("metrics", {}).get("tokens_processed", 0) for r in passed_original)
        avg_throughput = sum(r.get("metrics", {}).get("throughput_tokens_per_sec", 0) for r in passed_original) / len(passed_original) if passed_original else 0
        total_duration = sum(r["duration"] for r in original_results)
        
        report = {
            "testing_symbiote_report": {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "repository": "sigamani/doubleword-technical",
                "test_matrix_summary": {
                    "total_configurations": len(original_results),
                    "passed_initial": len(passed_original),
                    "failed_initial": len(failed_original),
                    "retry_attempts": len(retry_results),
                    "success_rate": len(passed_original) / len(original_results) * 100 if original_results else 0,
                    "total_duration": total_duration
                },
                "performance_metrics": {
                    "total_tokens_processed": total_tokens,
                    "average_throughput_tokens_per_sec": avg_throughput,
                    "best_throughput": max([r.get("metrics", {}).get("throughput_tokens_per_sec", 0) for r in passed_original], default=0),
                    "worst_throughput": min([r.get("metrics", {}).get("throughput_tokens_per_sec", float('inf')) for r in passed_original], default=0)
                },
                "failure_analysis": {
                    "failure_modes": self.failure_modes,
                    "failure_count_by_type": {}
                },
                "test_results": original_results,
                "retry_results": retry_results,
                "recommendations": []
            }
        }
        
        # Count failure types
        for failure_mode in self.failure_modes:
            failure_type = failure_mode["type"]
            report["failure_analysis"]["failure_count_by_type"][failure_type] = \
                report["failure_analysis"]["failure_count_by_type"].get(failure_type, 0) + 1
        
        # Generate recommendations
        if len(passed_original) == len(original_results):
            report["testing_symbiote_report"]["recommendations"].append("All test configurations passed - repository is stable")
        else:
            report["testing_symbiote_report"]["recommendations"].append(f"{len(failed_original)} test configurations failed - review failure modes")
        
        if avg_throughput < 100:
            report["testing_symbiote_report"]["recommendations"].append("Low average throughput detected - consider optimization")
        
        if len(self.failure_modes) > 3:
            report["testing_symbiote_report"]["recommendations"].append("Multiple failure modes detected - systematic review needed")
        
        # Performance recommendations
        best_config = max(passed_original, key=lambda x: x.get("metrics", {}).get("throughput_tokens_per_sec", 0), default=None)
        if best_config:
            report["testing_symbiote_report"]["recommendations"].append(f"Best performing configuration: {best_config['config_name']} with {best_config.get('metrics', {}).get('throughput_tokens_per_sec', 0):.1f} tokens/sec")
        
        return report
    
    def save_report(self, report: Dict[str, Any], filepath: Optional[str] = None) -> None:
        """Save comprehensive report to file"""
        if filepath is None:
            filepath = os.path.join(self.repo_root, "testing_symbiote_report.json")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Testing Symbiote report saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

def main():
    """Main entry point for Testing Symbiote"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Testing Symbiote for sigamani/doubleword-technical")
    parser.add_argument("--repo-root", help="Repository root directory", default=os.getcwd())
    parser.add_argument("--report-file", help="Report output file", default="testing_symbiote_report.json")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize Testing Symbiote
    symbiote = TestingSymbiote(args.repo_root)
    
    # Run complete test matrix
    logger.info("Testing Symbiote activated - Starting comprehensive test matrix...")
    report = symbiote.run_complete_test_matrix()
    
    # Save report
    symbiote.save_report(report, args.report_file)
    
    # Print summary
    summary = report["testing_symbiote_report"]["test_matrix_summary"]
    logger.info(f"{'='*60}")
    logger.info("TESTING SYMBIOTE REPORT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total Configurations: {summary['total_configurations']}")
    logger.info(f"Passed Initially: {summary['passed_initial']}")
    logger.info(f"Failed Initially: {summary['failed_initial']}")
    logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
    logger.info(f"Total Duration: {summary['total_duration']:.2f}s")
    
    if report["testing_symbiote_report"]["recommendations"]:
        logger.info("Recommendations:")
        for rec in report["testing_symbiote_report"]["recommendations"]:
            logger.info(f"  {rec}")
    
    logger.info(f"{'='*60}")
    
    # Return exit code based on results
    return 1 if summary["failed_initial"] > 0 else 0

if __name__ == "__main__":
    sys.exit(main())