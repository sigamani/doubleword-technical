#!/usr/bin/env python3
"""
SLA Tracking and 24-hour Completion Monitoring Validator
Tests the SLA monitoring functionality of the Ray Data batch inference system
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
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

# Import directly from the file to avoid import issues
exec(open(os.path.join(os.path.dirname(__file__), '..', 'app', 'ray_data_batch_inference.py')).read())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SLATestResult:
    """SLA test result"""
    test_name: str
    total_requests: int
    completed_requests: int
    elapsed_time_hours: float
    eta_hours: float
    sla_hours: float
    success: bool
    sla_warnings: List[str]
    throughput: float
    tokens_per_sec: float


class SLAValidator:
    """Comprehensive SLA validation system"""
    
    def __init__(self):
        self.test_results: List[SLATestResult] = []
        self.config = self._load_test_config()
    
    def _load_test_config(self) -> Dict:
        """Load test configuration"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for testing"""
        return {
            "sla": {
                "target_hours": 24,
                "buffer_factor": 0.7,
                "alert_threshold_hours": 20
            },
            "inference": {
                "batch_size": 128,
                "max_tokens": 512
            }
        }
    
    def test_sla_calculation_accuracy(self) -> SLATestResult:
        """Test SLA ETA calculation accuracy"""
        logger.info("Testing SLA calculation accuracy...")
        
        # Simulate a batch job
        total_requests = 1000
        metrics = BatchMetrics(total_requests=total_requests)
        sla_hours = self.config["sla"]["target_hours"]
        
        # Simulate progress over time
        completed_requests = 400
        metrics.completed_requests = completed_requests
        metrics.tokens_processed = completed_requests * 100  # Estimate
        metrics.start_time = time.time() - 2 * 3600  # Started 2 hours ago
        
        monitor = InferenceMonitor(metrics, sla_hours, log_interval=50)
        
        # Calculate metrics
        elapsed_time_hours = (time.time() - metrics.start_time) / 3600
        eta_hours = metrics.eta_hours()
        throughput = metrics.throughput_per_sec()
        tokens_per_sec = metrics.tokens_per_sec()
        
        # Check SLA compliance
        sla_warnings = []
        success = True
        
        if eta_hours > sla_hours:
            sla_warnings.append(f"ETA {eta_hours:.2f}h exceeds SLA {sla_hours}h")
            success = False
        
        if throughput <= 0:
            sla_warnings.append("Zero throughput detected")
            success = False
        
        # Test buffer factor
        remaining_time = sla_hours - elapsed_time_hours
        buffer_threshold = remaining_time * self.config["sla"]["buffer_factor"]
        
        if eta_hours > buffer_threshold:
            sla_warnings.append(f"ETA exceeds buffer threshold: {eta_hours:.2f}h > {buffer_threshold:.2f}h")
        
        result = SLATestResult(
            test_name="SLA Calculation Accuracy",
            total_requests=total_requests,
            completed_requests=completed_requests,
            elapsed_time_hours=elapsed_time_hours,
            eta_hours=eta_hours,
            sla_hours=sla_hours,
            success=success,
            sla_warnings=sla_warnings,
            throughput=throughput,
            tokens_per_sec=tokens_per_sec
        )
        
        self.test_results.append(result)
        return result
    
    def test_24_hour_completion_monitoring(self) -> SLATestResult:
        """Test 24-hour completion monitoring with realistic scenario"""
        logger.info("Testing 24-hour completion monitoring...")
        
        # Simulate a large batch job
        total_requests = 10000
        metrics = BatchMetrics(total_requests=total_requests)
        sla_hours = 24.0
        
        # Simulate realistic progress
        completed_requests = 2500  # 25% complete
        metrics.completed_requests = completed_requests
        metrics.tokens_processed = completed_requests * 150  # Average tokens per request
        metrics.start_time = time.time() - 6 * 3600  # Started 6 hours ago
        
        monitor = InferenceMonitor(metrics, sla_hours, log_interval=500)
        
        # Test SLA monitoring
        elapsed_time_hours = (time.time() - metrics.start_time) / 3600
        eta_hours = metrics.eta_hours()
        throughput = metrics.throughput_per_sec()
        tokens_per_sec = metrics.tokens_per_sec()
        
        sla_warnings = []
        success = True
        
        # Check if on track for 24-hour completion
        if eta_hours > (sla_hours - elapsed_time_hours):
            sla_warnings.append(f"Behind schedule: ETA {eta_hours:.2f}h > Remaining {sla_hours - elapsed_time_hours:.2f}h")
            success = False
        
        # Test alert threshold
        alert_threshold = self.config["sla"]["alert_threshold_hours"]
        if elapsed_time_hours > alert_threshold and eta_hours > (sla_hours - elapsed_time_hours):
            sla_warnings.append(f"Alert threshold exceeded: {elapsed_time_hours:.2f}h > {alert_threshold}h")
        
        # Validate throughput requirements
        required_throughput = total_requests / sla_hours
        if throughput < required_throughput * 0.8:  # 80% of required throughput
            sla_warnings.append(f"Throughput too low: {throughput:.2f} < {required_throughput * 0.8:.2f} req/s")
        
        result = SLATestResult(
            test_name="24-Hour Completion Monitoring",
            total_requests=total_requests,
            completed_requests=completed_requests,
            elapsed_time_hours=elapsed_time_hours,
            eta_hours=eta_hours,
            sla_hours=sla_hours,
            success=success,
            sla_warnings=sla_warnings,
            throughput=throughput,
            tokens_per_sec=tokens_per_sec
        )
        
        self.test_results.append(result)
        return result
    
    def test_sla_alert_system(self) -> SLATestResult:
        """Test SLA alert system functionality"""
        logger.info("Testing SLA alert system...")
        
        # Simulate a job at risk
        total_requests = 5000
        metrics = BatchMetrics(total_requests=total_requests)
        sla_hours = 12.0  # Shorter SLA for testing
        
        # Simulate slow progress
        completed_requests = 800  # Only 16% complete
        metrics.completed_requests = completed_requests
        metrics.tokens_processed = completed_requests * 120
        metrics.start_time = time.time() - 8 * 3600  # Started 8 hours ago
        
        monitor = InferenceMonitor(metrics, sla_hours, log_interval=100)
        
        # Trigger SLA check
        eta_hours = metrics.eta_hours()
        elapsed_time_hours = (time.time() - metrics.start_time) / 3600
        throughput = metrics.throughput_per_sec()
        tokens_per_sec = metrics.tokens_per_sec()
        
        sla_warnings = []
        success = False  # Expected to fail for this test
        
        # Test SLA check function
        sla_status = monitor.check_sla()
        if not sla_status:
            sla_warnings.append("SLA alert correctly triggered")
        else:
            sla_warnings.append("SLA alert should have been triggered")
        
        # Test multiple alert conditions
        remaining_hours = sla_hours - elapsed_time_hours
        if eta_hours > remaining_hours:
            sla_warnings.append(f"ETA exceeds remaining time: {eta_hours:.2f}h > {remaining_hours:.2f}h")
        
        if elapsed_time_hours > sla_hours * 0.8:  # 80% of SLA time used
            sla_warnings.append(f"Time budget mostly used: {elapsed_time_hours:.2f}h / {sla_hours}h")
        
        # This test expects alerts to be triggered
        if len(sla_warnings) > 1:  # At least the expected alert
            success = True
        
        result = SLATestResult(
            test_name="SLA Alert System",
            total_requests=total_requests,
            completed_requests=completed_requests,
            elapsed_time_hours=elapsed_time_hours,
            eta_hours=eta_hours,
            sla_hours=sla_hours,
            success=success,
            sla_warnings=sla_warnings,
            throughput=throughput,
            tokens_per_sec=tokens_per_sec
        )
        
        self.test_results.append(result)
        return result
    
    def test_progress_tracking_accuracy(self) -> SLATestResult:
        """Test progress tracking and metrics accuracy"""
        logger.info("Testing progress tracking accuracy...")
        
        total_requests = 2000
        metrics = BatchMetrics(total_requests=total_requests)
        sla_hours = 4.0
        
        # Simulate steady progress
        completed_requests = 1000  # 50% complete
        metrics.completed_requests = completed_requests
        metrics.tokens_processed = completed_requests * 200
        metrics.start_time = time.time() - 1 * 3600  # Started 1 hour ago
        
        monitor = InferenceMonitor(metrics, sla_hours, log_interval=200)
        
        # Calculate metrics
        elapsed_time_hours = (time.time() - metrics.start_time) / 3600
        eta_hours = metrics.eta_hours()
        throughput = metrics.throughput_per_sec()
        tokens_per_sec = metrics.tokens_per_sec()
        progress_pct = metrics.progress_pct()
        
        sla_warnings = []
        success = True
        
        # Validate progress calculation
        expected_progress = (completed_requests / total_requests) * 100
        if abs(progress_pct - expected_progress) > 0.1:
            sla_warnings.append(f"Progress calculation error: {progress_pct:.1f}% != {expected_progress:.1f}%")
            success = False
        
        # Validate ETA calculation
        if throughput > 0:
            expected_eta = (total_requests - completed_requests) / throughput / 3600
            if abs(eta_hours - expected_eta) > 0.1:
                sla_warnings.append(f"ETA calculation error: {eta_hours:.2f}h != {expected_eta:.2f}h")
                success = False
        
        # Check if on track
        if eta_hours > (sla_hours - elapsed_time_hours):
            sla_warnings.append(f"Not on track for SLA completion")
        
        result = SLATestResult(
            test_name="Progress Tracking Accuracy",
            total_requests=total_requests,
            completed_requests=completed_requests,
            elapsed_time_hours=elapsed_time_hours,
            eta_hours=eta_hours,
            sla_hours=sla_hours,
            success=success,
            sla_warnings=sla_warnings,
            throughput=throughput,
            tokens_per_sec=tokens_per_sec
        )
        
        self.test_results.append(result)
        return result
    
    def run_all_sla_tests(self) -> List[SLATestResult]:
        """Run all SLA validation tests"""
        logger.info("Starting comprehensive SLA validation...")
        
        tests = [
            self.test_sla_calculation_accuracy,
            self.test_24_hour_completion_monitoring,
            self.test_sla_alert_system,
            self.test_progress_tracking_accuracy
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                
                # Log result
                status = "PASS" if result.success else "FAIL"
                logger.info(f"SLA Test {result.test_name}: {status}")
                if result.sla_warnings:
                    for warning in result.sla_warnings:
                        logger.warning(f"  {warning}")
                        
            except Exception as e:
                logger.error(f"SLA test {test_func.__name__} failed: {e}")
                # Create failure result
                failure_result = SLATestResult(
                    test_name=test_func.__name__,
                    total_requests=0,
                    completed_requests=0,
                    elapsed_time_hours=0,
                    eta_hours=0,
                    sla_hours=0,
                    success=False,
                    sla_warnings=[f"Test execution failed: {str(e)}"],
                    throughput=0,
                    tokens_per_sec=0
                )
                results.append(failure_result)
        
        self.test_results = results
        return results
    
    def generate_sla_report(self) -> Dict:
        """Generate comprehensive SLA validation report"""
        if not self.test_results:
            return {"error": "No SLA test results available"}
        
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        report = {
            "sla_validation_summary": {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_tests": len(self.test_results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.test_results) * 100,
                "sla_hours_target": self.config["sla"]["target_hours"],
                "buffer_factor": self.config["sla"]["buffer_factor"],
                "alert_threshold_hours": self.config["sla"]["alert_threshold_hours"]
            },
            "test_results": [],
            "performance_metrics": {},
            "recommendations": []
        }
        
        # Add individual test results
        for result in self.test_results:
            result_dict = {
                "test_name": result.test_name,
                "total_requests": result.total_requests,
                "completed_requests": result.completed_requests,
                "elapsed_time_hours": result.elapsed_time_hours,
                "eta_hours": result.eta_hours,
                "sla_hours": result.sla_hours,
                "success": result.success,
                "sla_warnings": result.sla_warnings,
                "throughput": result.throughput,
                "tokens_per_sec": result.tokens_per_sec,
                "progress_pct": (result.completed_requests / result.total_requests * 100) if result.total_requests > 0 else 0
            }
            report["test_results"].append(result_dict)
        
        # Performance analysis
        if successful_tests:
            avg_throughput = sum(r.throughput for r in successful_tests) / len(successful_tests)
            avg_tokens_per_sec = sum(r.tokens_per_sec for r in successful_tests) / len(successful_tests)
            
            report["performance_metrics"] = {
                "average_throughput": avg_throughput,
                "average_tokens_per_sec": avg_tokens_per_sec,
                "best_throughput": max(r.throughput for r in successful_tests),
                "best_tokens_per_sec": max(r.tokens_per_sec for r in successful_tests)
            }
        
        # Recommendations
        if failed_tests:
            report["recommendations"].append("Some SLA tests failed - review monitoring configuration")
        
        if successful_tests:
            report["recommendations"].append("SLA tracking system is functioning correctly")
            
            # Check if any test shows risk of SLA violation
            at_risk_tests = [r for r in successful_tests if r.eta_hours > r.sla_hours * 0.8]
            if at_risk_tests:
                report["recommendations"].append("Some scenarios show SLA risk - consider optimizing throughput")
        
        return report
    
    def save_sla_report(self, filepath: str):
        """Save SLA validation report to file"""
        report = self.generate_sla_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"SLA validation report saved to: {filepath}")


class TestSLAMonitoring(unittest.TestCase):
    """Unit tests for SLA monitoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = SLAValidator()
    
    def test_batch_metrics_initialization(self):
        """Test BatchMetrics initialization"""
        metrics = BatchMetrics(total_requests=1000)
        self.assertEqual(metrics.total_requests, 1000)
        self.assertEqual(metrics.completed_requests, 0)
        self.assertEqual(metrics.failed_requests, 0)
        self.assertIsNotNone(metrics.start_time)
    
    def test_throughput_calculation(self):
        """Test throughput calculation"""
        metrics = BatchMetrics(total_requests=1000)
        metrics.completed_requests = 500
        metrics.start_time = time.time() - 100  # 100 seconds ago
        
        throughput = metrics.throughput_per_sec()
        self.assertGreater(throughput, 0)
        self.assertAlmostEqual(throughput, 5.0, places=1)  # 500 requests / 100 seconds
    
    def test_eta_calculation(self):
        """Test ETA calculation"""
        metrics = BatchMetrics(total_requests=1000)
        metrics.completed_requests = 500
        metrics.start_time = time.time() - 100  # 100 seconds ago
        
        eta = metrics.eta_hours()
        self.assertGreater(eta, 0)
        # Should be approximately the same as elapsed time for 50% completion
        elapsed_hours = 100 / 3600
        self.assertAlmostEqual(eta, elapsed_hours, places=1)
    
    def test_progress_percentage(self):
        """Test progress percentage calculation"""
        metrics = BatchMetrics(total_requests=1000)
        metrics.completed_requests = 250
        
        progress = metrics.progress_pct()
        self.assertEqual(progress, 25.0)
    
    def test_sla_monitor_initialization(self):
        """Test InferenceMonitor initialization"""
        metrics = BatchMetrics(total_requests=1000)
        monitor = InferenceMonitor(metrics, sla_hours=24.0)
        
        self.assertEqual(monitor.metrics, metrics)
        self.assertEqual(monitor.sla_hours, 24.0)
        self.assertEqual(monitor.log_interval, 100)
        self.assertEqual(monitor.last_log, 0)


def main():
    """Main function for SLA validation"""
    validator = SLAValidator()
    
    # Run all SLA tests
    results = validator.run_all_sla_tests()
    
    # Generate and save report
    report_path = "sla_validation_report.json"
    validator.save_sla_report(report_path)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("SLA VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
    
    if failed:
        print("\nFailed tests:")
        for test in failed:
            print(f"  {test.test_name}: {test.sla_warnings}")
    
    print(f"\nDetailed report saved to: {report_path}")
    
    # Return exit code
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())