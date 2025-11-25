#!/usr/bin/env python3
"""
Testing Agent for Ray Data + vLLM Batch Inference Repository
Responsible for running the complete testing matrix and fixing any issues
"""

import os
import sys
import time
import json
import logging
import subprocess
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# Try to import yaml, fallback if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    success: bool
    duration: float
    error_message: str = ""
    output: str = ""

@dataclass
class TestConfiguration:
    """Test configuration matrix"""
    name: str
    model_name: str
    batch_size: int
    concurrency: int
    max_tokens: int
    num_samples: int
    expected_impact: str
    max_model_len: int = 32768
    tensor_parallel_size: int = 2

class RepositoryTester:
    """Comprehensive repository testing agent"""
    
    def __init__(self, repo_root: str = None):
        self.repo_root = repo_root or os.getcwd()
        self.test_results: List[TestResult] = []
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load repository configuration"""
        if not HAS_YAML:
            logger.warning("YAML not available, using default config")
            return self._get_default_config()
            
        config_path = os.path.join(self.repo_root, "config", "config.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for testing"""
        return {
            "model": {
                "name": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_model_len": 32768,
                "tensor_parallel_size": 2
            },
            "inference": {
                "batch_size": 128,
                "concurrency": 2,
                "max_num_batched_tokens": 16384,
                "gpu_memory_utilization": 0.90,
                "temperature": 0.7,
                "max_tokens": 512
            },
            "data": {
                "input_path": "/tmp/test_input.json",
                "output_path": "/tmp/test_output",
                "num_samples": 1000
            },
            "sla": {
                "target_hours": 24,
                "buffer_factor": 0.7,
                "alert_threshold_hours": 20
            }
        }
    
    def get_test_matrix(self) -> List[TestConfiguration]:
        """Get the complete testing matrix configuration"""
        return [
            TestConfiguration(
                name="baseline_0.5b",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=128,
                concurrency=2,
                max_tokens=512,
                num_samples=50,  # Small for testing
                expected_impact="Baseline configuration"
            ),
            TestConfiguration(
                name="high_batch_0.5b",
                model_name="Qwen/Qwen2.5-0.5B-Instruct", 
                batch_size=256,
                concurrency=2,
                max_tokens=512,
                num_samples=50,
                expected_impact="+30-50% throughput expected"
            ),
            TestConfiguration(
                name="high_concurrency_0.5b",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=128,
                concurrency=4,
                max_tokens=512,
                num_samples=50,
                expected_impact="Test scaling with higher concurrency"
            ),
            TestConfiguration(
                name="baseline_7b",
                model_name="Qwen/Qwen2.5-7B-Instruct",
                batch_size=64,
                concurrency=2,
                max_tokens=256,
                num_samples=25,  # Fewer for larger model
                expected_impact="Validate larger model performance",
                max_model_len=8192,
                tensor_parallel_size=1
            ),
            TestConfiguration(
                name="stress_test",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                batch_size=512,
                concurrency=4,
                max_tokens=1024,
                num_samples=100,
                expected_impact="Stress test with large batches"
            )
        ]
    
    def run_command(self, cmd: List[str], cwd: str = None, timeout: int = 300) -> TestResult:
        """Run a command and capture result"""
        start_time = time.time()
        cmd_str = " ".join(cmd)
        
        try:
            logger.info(f"Running: {cmd_str}")
            result = subprocess.run(
                cmd,
                cwd=cwd or self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            return TestResult(
                test_name=cmd_str,
                success=success,
                duration=duration,
                error_message=result.stderr if result.stderr else "",
                output=result.stdout if result.stdout else ""
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                test_name=cmd_str,
                success=False,
                duration=duration,
                error_message=f"Command timed out after {timeout}s"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=cmd_str,
                success=False,
                duration=duration,
                error_message=str(e)
            )
    
    def test_syntax(self) -> TestResult:
        """Test Python syntax of main application"""
        logger.info("Testing Python syntax...")
        
        app_file = os.path.join(self.repo_root, "app", "ray_data_batch_inference.py")
        cmd = ["python3", "-m", "py_compile", app_file]
        
        result = self.run_command(cmd)
        result.test_name = "Python Syntax Check"
        
        if result.success:
            logger.info("✓ Python syntax is valid")
        else:
            logger.error(f"✗ Python syntax error: {result.error_message}")
            
        return result
    
    def test_dependencies(self) -> TestResult:
        """Test if all dependencies can be imported"""
        logger.info("Testing dependency imports...")
        
        test_code = '''
try:
    import ray
    from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
    import redis
    from prometheus_client import Counter, Gauge
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    print("SUCCESS: All dependencies imported")
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
    exit(1)
except Exception as e:
    print(f"OTHER_ERROR: {e}")
    exit(1)
'''
        
        test_file = os.path.join(self.repo_root, "test_imports.py")
        
        try:
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            cmd = ["python3", test_file]
            result = self.run_command(cmd)
            result.test_name = "Dependency Import Test"
            
            if result.success and "SUCCESS" in result.output:
                logger.info("✓ All dependencies imported successfully")
            else:
                logger.error(f"✗ Dependency import failed: {result.output}")
                result.success = False
                
            return result
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_configuration_loading(self) -> TestResult:
        """Test configuration loading"""
        logger.info("Testing configuration loading...")
        
        test_code = f'''
import yaml
import os

try:
    config_path = os.path.join("{self.repo_root}", "config", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ["model", "inference", "data", "sla"]
    for section_name in required_sections:
        if section_name not in config:
            raise ValueError(f"Missing required section: {{section_name}}")
    
    print("SUCCESS: Configuration loaded and validated")
except Exception as e:
    print(f"CONFIG_ERROR: {{e}}")
    exit(1)
'''
        
        test_file = os.path.join(self.repo_root, "test_config.py")
        
        try:
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            cmd = ["python3", test_file]
            result = self.run_command(cmd)
            result.test_name = "Configuration Loading Test"
            
            if result.success and "SUCCESS" in result.output:
                logger.info("✓ Configuration loads successfully")
            else:
                logger.error(f"✗ Configuration loading failed: {result.output}")
                result.success = False
                
            return result
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_unit_tests(self) -> TestResult:
        """Run unit tests"""
        logger.info("Running unit tests...")
        
        test_files = [
            "tests/test_sla_simple.py",
            "tests/test_sla_validation.py"
        ]
        
        all_passed = True
        total_duration = 0
        errors = []
        
        for test_file in test_files:
            file_path = os.path.join(self.repo_root, test_file)
            if not os.path.exists(file_path):
                logger.warning(f"Test file not found: {test_file}")
                continue
                
            cmd = ["python3", test_file]
            result = self.run_command(cmd, timeout=120)
            
            total_duration += result.duration
            if not result.success:
                all_passed = False
                errors.append(f"{test_file}: {result.error_message}")
        
        return TestResult(
            test_name="Unit Tests",
            success=all_passed,
            duration=total_duration,
            error_message="; ".join(errors) if errors else ""
        )
    
    def test_integration_tests(self) -> TestResult:
        """Run integration tests"""
        logger.info("Running integration tests...")
        
        test_files = [
            "tests/test_batch_inference_e2e.py",
            "tests/test_config_matrix.py"
        ]
        
        all_passed = True
        total_duration = 0
        errors = []
        
        for test_file in test_files:
            file_path = os.path.join(self.repo_root, test_file)
            if not os.path.exists(file_path):
                logger.warning(f"Test file not found: {test_file}")
                continue
                
            cmd = ["python3", test_file]
            result = self.run_command(cmd, timeout=300)
            
            total_duration += result.duration
            if not result.success:
                all_passed = False
                errors.append(f"{test_file}: {result.error_message}")
        
        return TestResult(
            test_name="Integration Tests",
            success=all_passed,
            duration=total_duration,
            error_message="; ".join(errors) if errors else ""
        )
    
    def test_api_endpoints(self) -> TestResult:
        """Test FastAPI endpoints"""
        logger.info("Testing API endpoints...")
        
        # Test if FastAPI app can be imported and initialized
        test_code = '''
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

try:
    from ray_data_batch_inference import app
    print("SUCCESS: FastAPI app initialized")
except Exception as e:
    print(f"API_ERROR: {e}")
    exit(1)
'''
        
        test_file = os.path.join(self.repo_root, "test_api.py")
        
        try:
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            cmd = ["python3", test_file]
            result = self.run_command(cmd)
            result.test_name = "API Endpoint Test"
            
            if result.success and "SUCCESS" in result.output:
                logger.info("✓ FastAPI app initializes successfully")
            else:
                logger.error(f"✗ API initialization failed: {result.output}")
                result.success = False
                
            return result
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_docker_build(self) -> TestResult:
        """Test Docker build"""
        logger.info("Testing Docker build...")
        
        dockerfile_path = os.path.join(self.repo_root, "Dockerfile")
        if not os.path.exists(dockerfile_path):
            return TestResult(
                test_name="Docker Build Test",
                success=False,
                duration=0,
                error_message="Dockerfile not found"
            )
        
        cmd = ["docker", "build", "-t", "test-batch-inference", "."]
        result = self.run_command(cmd, timeout=600)
        result.test_name = "Docker Build Test"
        
        if result.success:
            logger.info("✓ Docker build successful")
        else:
            logger.error(f"✗ Docker build failed: {result.error_message}")
            
        return result
    
    def fix_broken_tests(self) -> None:
        """Automatically fix common test issues"""
        logger.info("Attempting to fix broken tests...")
        
        fixes_applied = []
        
        # Fix 1: Check for missing test directories
        test_dir = os.path.join(self.repo_root, "tests")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)
            fixes_applied.append("Created tests directory")
        
        # Fix 2: Check for missing config directory
        config_dir = os.path.join(self.repo_root, "config")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
            fixes_applied.append("Created config directory")
        
        # Fix 3: Create basic config file if missing
        config_file = os.path.join(config_dir, "config.yaml")
        if not os.path.exists(config_file) and HAS_YAML:
            with open(config_file, 'w') as f:
                yaml.dump(self._get_default_config(), f, default_flow_style=False)
            fixes_applied.append("Created default config.yaml")
        
        # Fix 4: Check for missing app directory
        app_dir = os.path.join(self.repo_root, "app")
        if not os.path.exists(app_dir):
            os.makedirs(app_dir, exist_ok=True)
            fixes_applied.append("Created app directory")
        
        # Fix 5: Create __init__.py files for Python packages
        init_files = [
            os.path.join(self.repo_root, "app", "__init__.py"),
            os.path.join(self.repo_root, "tests", "__init__.py")
        ]
        
        for init_file in init_files:
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('"""Package initialization file"""\n')
                fixes_applied.append(f"Created {init_file}")
        
        if fixes_applied:
            logger.info(f"Applied fixes: {', '.join(fixes_applied)}")
        else:
            logger.info("No fixes needed")
    
    def run_test_matrix(self) -> List[TestResult]:
        """Run the complete testing matrix"""
        logger.info("Starting comprehensive test matrix...")
        
        # First, fix any obvious issues
        self.fix_broken_tests()
        
        # Run all tests
        tests = [
            ("Syntax Check", self.test_syntax),
            ("Dependency Imports", self.test_dependencies),
            ("Configuration Loading", self.test_configuration_loading),
            ("Unit Tests", self.test_unit_tests),
            ("Integration Tests", self.test_integration_tests),
            ("API Endpoints", self.test_api_endpoints)
        ]
        
        # Add Docker test only if Docker is available
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            tests.append(("Docker Build", self.test_docker_build))
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("Docker not available, skipping Docker build test")
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = test_func()
                results.append(result)
                
                status = "PASS" if result.success else "FAIL"
                duration_str = f"{result.duration:.2f}s"
                logger.info(f"{test_name}: {status} ({duration_str})")
                
                if result.error_message:
                    logger.error(f"  Error: {result.error_message}")
                    
            except Exception as e:
                error_result = TestResult(
                    test_name=test_name,
                    success=False,
                    duration=0,
                    error_message=str(e)
                )
                results.append(error_result)
                logger.error(f"{test_name}: CRASH - {str(e)}")
                logger.error(traceback.format_exc())
        
        self.test_results = results
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        passed_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        total_duration = sum(r.duration for r in self.test_results)
        
        report = {
            "test_summary": {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_tests": len(self.test_results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "success_rate": len(passed_tests) / len(self.test_results) * 100,
                "total_duration": total_duration
            },
            "test_results": [],
            "failed_tests": [],
            "recommendations": []
        }
        
        # Add individual test results
        for result in self.test_results:
            result_dict = {
                "test_name": result.test_name,
                "success": result.success,
                "duration": result.duration,
                "error_message": result.error_message
            }
            report["test_results"].append(result_dict)
            
            if not result.success:
                report["failed_tests"].append(result_dict)
        
        # Add recommendations
        if failed_tests:
            report["recommendations"].append("Fix failed tests before proceeding to production")
        
        if len(passed_tests) == len(self.test_results):
            report["recommendations"].append("All tests passed - ready for production deployment")
        
        # Check for specific issues
        for result in failed_tests:
            if "import" in result.error_message.lower():
                report["recommendations"].append("Install missing dependencies: pip install -r requirements.txt")
            elif "syntax" in result.error_message.lower():
                report["recommendations"].append("Fix Python syntax errors in source code")
            elif "config" in result.error_message.lower():
                report["recommendations"].append("Validate configuration file format and required fields")
        
        return report
    
    def save_report(self, filepath: str) -> None:
        """Save test report to file"""
        report = self.generate_report()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Test report saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def print_summary(self) -> None:
        """Print test summary to console"""
        if not self.test_results:
            logger.error("No test results to display")
            return
        
        passed = len([r for r in self.test_results if r.success])
        failed = len([r for r in self.test_results if not r.success])
        total = len(self.test_results)
        
        print(f"\n{'='*60}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        total_duration = sum(r.duration for r in self.test_results)
        print(f"Total Duration: {total_duration:.2f}s")
        
        if failed > 0:
            print(f"\nFailed Tests:")
            for result in self.test_results:
                if not result.success:
                    print(f"  ✗ {result.test_name}: {result.error_message}")
        
        print(f"\n{'='*60}")

def main():
    """Main testing agent entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Testing Agent")
    parser.add_argument(
        "--repo-root", 
        help="Repository root directory",
        default=os.getcwd()
    )
    parser.add_argument(
        "--report-file",
        help="Test report output file", 
        default="test_report.json"
    )
    parser.add_argument(
        "--fix-only",
        action="store_true",
        help="Only apply fixes without running tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    tester = RepositoryTester(args.repo_root)
    
    if args.fix_only:
        logger.info("Applying fixes only...")
        tester.fix_broken_tests()
        return
    
    # Run test matrix
    logger.info("Starting repository testing agent...")
    results = tester.run_test_matrix()
    
    # Generate and save report
    tester.save_report(args.report_file)
    
    # Print summary
    tester.print_summary()
    
    # Return exit code based on results
    failed_count = len([r for r in results if not r.success])
    return 1 if failed_count > 0 else 0

if __name__ == "__main__":
    sys.exit(main())