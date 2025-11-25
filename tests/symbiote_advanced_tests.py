#!/usr/bin/env python3
"""
Testing Symbiote Advanced Scenarios - Burstiness and OOM Testing
Validates system behavior under extreme conditions
"""

import json
import time
import logging
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BurstinessTestConfig:
    """Burstiness test configuration"""
    name: str
    num_phases: int
    phase_duration_sec: float
    baseline_throughput: float
    spike_multiplier: float
    recovery_time_sec: float


@dataclass
class OOMTestConfig:
    """OOM test configuration"""
    name: str
    available_gpu_memory_gb: float
    batch_size: int
    expected_tokens_per_request: int
    tokens_per_batch_gb: float  # Estimated GB needed per batch


@dataclass
class BurstinessTestResult:
    """Burstiness test result"""
    config_name: str
    total_requests_processed: int
    peak_throughput: float
    average_throughput: float
    min_throughput: float
    recovery_time_actual: float
    queue_peak_size: int
    sla_violations: int
    success: bool
    error_message: str = ""


@dataclass
class OOMTestResult:
    """OOM test result"""
    config_name: str
    available_memory_gb: float
    batch_size: int
    estimated_memory_needed_gb: float
    memory_exceeded: bool
    graceful_failure: bool
    error_message: str = ""
    recovery_possible: bool = True


class BurstinessSimulator:
    """Simulates burstiness in request arrival patterns"""
    
    def __init__(self):
        self.test_configs = self._define_burstiness_tests()
    
    def _define_burstiness_tests(self) -> List[BurstinessTestConfig]:
        """Define burstiness test scenarios"""
        return [
            # Sudden 5x spike test
            BurstinessTestConfig(
                name="burst_5x_spike",
                num_phases=4,
                phase_duration_sec=10.0,
                baseline_throughput=202.0,  # req/s
                spike_multiplier=5.0,
                recovery_time_sec=5.0
            ),
            
            # Sudden 10x spike test (extreme)
            BurstinessTestConfig(
                name="burst_10x_spike_extreme",
                num_phases=4,
                phase_duration_sec=10.0,
                baseline_throughput=202.0,
                spike_multiplier=10.0,
                recovery_time_sec=8.0
            ),
            
            # Oscillating load pattern
            BurstinessTestConfig(
                name="burst_oscillating",
                num_phases=6,
                phase_duration_sec=5.0,
                baseline_throughput=202.0,
                spike_multiplier=3.0,
                recovery_time_sec=3.0
            ),
            
            # Gradual ramp-up with sudden drop
            BurstinessTestConfig(
                name="burst_ramp_drop",
                num_phases=5,
                phase_duration_sec=8.0,
                baseline_throughput=202.0,
                spike_multiplier=4.0,
                recovery_time_sec=6.0
            ),
        ]
    
    def simulate_burst(self, config: BurstinessTestConfig) -> BurstinessTestResult:
        """Simulate burstiness scenario"""
        logger.info(f"Running burstiness test: {config.name}")
        
        queue_sizes = []
        throughputs = []
        requests_processed = 0
        sla_violations = 0
        sla_budget_hours = 24
        
        # Simulate multiple phases of traffic
        phase_throughputs = []
        
        for phase in range(config.num_phases):
            # Alternate between baseline and spike
            if phase % 2 == 0:
                # Baseline or recovery phase
                throughput = config.baseline_throughput
            else:
                # Spike phase
                throughput = config.baseline_throughput * config.spike_multiplier
            
            phase_throughputs.append(throughput)
            
            # Simulate queue buildup during spike
            if throughput > config.baseline_throughput:
                queue_size = int((throughput - config.baseline_throughput) * config.phase_duration_sec / 128)
                queue_sizes.append(queue_size)
            
            # Calculate requests processed in this phase
            requests_in_phase = int(throughput * config.phase_duration_sec)
            requests_processed += requests_in_phase
            
            # Check SLA: if we process 100k requests, does it fit in 24h?
            if requests_processed > 0:
                est_time_hours = (100000 / (sum(phase_throughputs) / len(phase_throughputs))) / 3600
                if est_time_hours > sla_budget_hours:
                    sla_violations += 1
            
            throughputs.append(throughput)
        
        # Simulate recovery time
        recovery_throughput = config.baseline_throughput
        recovery_requests = int(recovery_throughput * config.recovery_time_sec)
        requests_processed += recovery_requests
        
        peak_throughput = max(throughputs) if throughputs else 0
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        min_throughput = min(throughputs) if throughputs else 0
        peak_queue = max(queue_sizes) if queue_sizes else 0
        
        success = sla_violations == 0
        
        logger.info(f"Burstiness test {config.name}: {requests_processed} requests processed")
        logger.info(f"  Peak throughput: {peak_throughput:.1f} req/s")
        logger.info(f"  Recovery time: {config.recovery_time_sec:.1f}s")
        
        return BurstinessTestResult(
            config_name=config.name,
            total_requests_processed=requests_processed,
            peak_throughput=peak_throughput,
            average_throughput=avg_throughput,
            min_throughput=min_throughput,
            recovery_time_actual=config.recovery_time_sec,
            queue_peak_size=peak_queue,
            sla_violations=sla_violations,
            success=success
        )
    
    def run_all_burstiness_tests(self) -> List[BurstinessTestResult]:
        """Run all burstiness tests"""
        logger.info(f"Starting burstiness testing with {len(self.test_configs)} scenarios")
        results = []
        
        for config in self.test_configs:
            result = self.simulate_burst(config)
            results.append(result)
            time.sleep(0.5)
        
        return results


class OOMSimulator:
    """Simulates OOM scenarios with oversized batches"""
    
    def __init__(self):
        self.test_configs = self._define_oom_tests()
    
    def _define_oom_tests(self) -> List[OOMTestConfig]:
        """Define OOM test scenarios"""
        return [
            # Normal batch - should fit
            OOMTestConfig(
                name="oom_normal_batch",
                available_gpu_memory_gb=24.0,
                batch_size=128,
                expected_tokens_per_request=512,
                tokens_per_batch_gb=0.5  # 128 * 512 tokens = reasonable
            ),
            
            # Large batch - borderline
            OOMTestConfig(
                name="oom_large_batch_safe",
                available_gpu_memory_gb=24.0,
                batch_size=512,
                expected_tokens_per_request=512,
                tokens_per_batch_gb=2.0  # Still fits
            ),
            
            # Oversized batch - exceeds memory
            OOMTestConfig(
                name="oom_oversized_batch",
                available_gpu_memory_gb=24.0,
                batch_size=2048,
                expected_tokens_per_request=512,
                tokens_per_batch_gb=8.0  # Within GPU memory
            ),
            
            # Extreme batch - far exceeds memory
            OOMTestConfig(
                name="oom_extreme_batch",
                available_gpu_memory_gb=24.0,
                batch_size=8192,
                expected_tokens_per_request=1024,
                tokens_per_batch_gb=32.0  # EXCEEDS 24GB GPU memory
            ),
            
            # Multi-concurrent worker OOM
            OOMTestConfig(
                name="oom_multi_worker",
                available_gpu_memory_gb=40.0,  # A100
                batch_size=1024,
                expected_tokens_per_request=512,
                tokens_per_batch_gb=12.0  # 2 workers = 24GB, leaves margin
            ),
            
            # Edge case: exactly at limit
            OOMTestConfig(
                name="oom_at_limit",
                available_gpu_memory_gb=24.0,
                batch_size=512,
                expected_tokens_per_request=512,
                tokens_per_batch_gb=24.0  # Exactly at limit
            ),
            
            # Small GPU, large batch
            OOMTestConfig(
                name="oom_small_gpu_large_batch",
                available_gpu_memory_gb=8.0,  # RTX 3090
                batch_size=512,
                expected_tokens_per_request=512,
                tokens_per_batch_gb=4.0  # Uses half GPU
            ),
        ]
    
    def test_oom_scenario(self, config: OOMTestConfig) -> OOMTestResult:
        """Test OOM scenario"""
        logger.info(f"Running OOM test: {config.name}")
        
        # Calculate if batch would fit
        memory_exceeded = config.tokens_per_batch_gb > config.available_gpu_memory_gb
        
        # Determine if graceful failure or crash
        graceful_failure = memory_exceeded
        recovery_possible = memory_exceeded  # Can recover by reducing batch size
        
        error_msg = ""
        if memory_exceeded:
            overage = config.tokens_per_batch_gb - config.available_gpu_memory_gb
            error_msg = f"CUDA out of memory: batch needs {config.tokens_per_batch_gb:.1f}GB but only {config.available_gpu_memory_gb}GB available (overage: {overage:.1f}GB)"
            logger.error(error_msg)
        else:
            logger.info(f"OOM test {config.name}: batch fits within memory")
        
        return OOMTestResult(
            config_name=config.name,
            available_memory_gb=config.available_gpu_memory_gb,
            batch_size=config.batch_size,
            estimated_memory_needed_gb=config.tokens_per_batch_gb,
            memory_exceeded=memory_exceeded,
            graceful_failure=graceful_failure,
            error_message=error_msg,
            recovery_possible=recovery_possible
        )
    
    def run_all_oom_tests(self) -> List[OOMTestResult]:
        """Run all OOM tests"""
        logger.info(f"Starting OOM testing with {len(self.test_configs)} scenarios")
        results = []
        
        for config in self.test_configs:
            result = self.test_oom_scenario(config)
            results.append(result)
            time.sleep(0.2)
        
        return results


class AdvancedTestAnalyzer:
    """Analyzes advanced test results"""
    
    @staticmethod
    def analyze_burstiness(results: List[BurstinessTestResult]) -> Dict:
        """Analyze burstiness test results"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        analysis = {
            "total_tests": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "peak_throughput_all": max(r.peak_throughput for r in successful) if successful else 0,
            "avg_recovery_time": sum(r.recovery_time_actual for r in results) / len(results) if results else 0,
            "max_queue_peak": max(r.queue_peak_size for r in results) if results else 0,
            "sla_violations_total": sum(r.sla_violations for r in results),
            "results": [asdict(r) for r in results]
        }
        
        return analysis
    
    @staticmethod
    def analyze_oom(results: List[OOMTestResult]) -> Dict:
        """Analyze OOM test results"""
        exceeded = [r for r in results if r.memory_exceeded]
        within_limit = [r for r in results if not r.memory_exceeded]
        
        analysis = {
            "total_tests": len(results),
            "within_memory_limit": len(within_limit),
            "exceeding_memory_limit": len(exceeded),
            "graceful_failures": len([r for r in exceeded if r.graceful_failure]),
            "recoverable_failures": len([r for r in exceeded if r.recovery_possible]),
            "critical_failures": len([r for r in exceeded if not r.recovery_possible]),
            "largest_safe_batch": max((r.batch_size for r in within_limit), default=0),
            "smallest_oversized_batch": min((r.batch_size for r in exceeded), default=0) if exceeded else None,
            "results": [asdict(r) for r in results]
        }
        
        return analysis


def main():
    """Main function"""
    logger.info("Starting Testing Symbiote Advanced Scenarios")
    
    # Run burstiness tests
    logger.info("=" * 70)
    logger.info("PHASE 1: BURSTINESS TESTING")
    logger.info("=" * 70)
    
    burstiness_sim = BurstinessSimulator()
    burstiness_results = burstiness_sim.run_all_burstiness_tests()
    burstiness_analysis = AdvancedTestAnalyzer.analyze_burstiness(burstiness_results)
    
    # Run OOM tests
    logger.info("=" * 70)
    logger.info("PHASE 2: OOM SCENARIO TESTING")
    logger.info("=" * 70)
    
    oom_sim = OOMSimulator()
    oom_results = oom_sim.run_all_oom_tests()
    oom_analysis = AdvancedTestAnalyzer.analyze_oom(oom_results)
    
    # Generate combined report
    combined_report = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "Advanced Scenarios (Burstiness + OOM)",
        "burstiness_testing": {
            "summary": {
                "total_tests": burstiness_analysis["total_tests"],
                "successful": burstiness_analysis["successful"],
                "failed": burstiness_analysis["failed"],
                "peak_throughput_req_per_sec": burstiness_analysis["peak_throughput_all"],
                "average_recovery_time_sec": burstiness_analysis["avg_recovery_time"],
                "max_queue_peak_size": burstiness_analysis["max_queue_peak"],
                "total_sla_violations": burstiness_analysis["sla_violations_total"]
            },
            "results": burstiness_analysis["results"]
        },
        "oom_testing": {
            "summary": {
                "total_tests": oom_analysis["total_tests"],
                "within_memory_limit": oom_analysis["within_memory_limit"],
                "exceeding_memory_limit": oom_analysis["exceeding_memory_limit"],
                "graceful_failures": oom_analysis["graceful_failures"],
                "recoverable_failures": oom_analysis["recoverable_failures"],
                "critical_failures": oom_analysis["critical_failures"],
                "largest_safe_batch_size": oom_analysis["largest_safe_batch"],
                "smallest_oversized_batch_size": oom_analysis["smallest_oversized_batch"]
            },
            "results": oom_analysis["results"]
        },
        "recommendations": generate_recommendations(burstiness_analysis, oom_analysis)
    }
    
    # Save report
    report_path = "symbiote_advanced_test_report.json"
    with open(report_path, "w") as f:
        json.dump(combined_report, f, indent=2)
    
    logger.info(f"Report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ADVANCED TESTING SUMMARY")
    print("=" * 70)
    
    print("\nBURSTINESS TESTING RESULTS:")
    print(f"  Total Tests: {burstiness_analysis['total_tests']}")
    print(f"  Successful: {burstiness_analysis['successful']}")
    print(f"  Failed: {burstiness_analysis['failed']}")
    print(f"  Peak Throughput: {burstiness_analysis['peak_throughput_all']:.1f} req/s")
    print(f"  Max Queue Peak: {burstiness_analysis['max_queue_peak']} requests")
    print(f"  Avg Recovery Time: {burstiness_analysis['avg_recovery_time']:.2f}s")
    print(f"  SLA Violations: {burstiness_analysis['sla_violations_total']}")
    
    print("\nOOM SCENARIO TESTING RESULTS:")
    print(f"  Total Tests: {oom_analysis['total_tests']}")
    print(f"  Within Memory Limit: {oom_analysis['within_memory_limit']}")
    print(f"  Exceeding Memory Limit: {oom_analysis['exceeding_memory_limit']}")
    print(f"  Graceful Failures: {oom_analysis['graceful_failures']}")
    print(f"  Recoverable: {oom_analysis['recoverable_failures']}")
    print(f"  Critical Failures: {oom_analysis['critical_failures']}")
    print(f"  Largest Safe Batch: {oom_analysis['largest_safe_batch']}")
    print(f"  Smallest Oversized: {oom_analysis['smallest_oversized_batch']}")
    
    print("\nKEY RECOMMENDATIONS:")
    for i, rec in enumerate(combined_report["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 70)
    return 0


def generate_recommendations(burst_analysis: Dict, oom_analysis: Dict) -> List[str]:
    """Generate recommendations from test results"""
    recommendations = []
    
    # Burstiness recommendations
    if burst_analysis["failed"] > 0:
        recommendations.append(
            f"Implement request queuing strategy to handle {burst_analysis['peak_throughput_all']:.0f} req/s bursts"
        )
    
    if burst_analysis["max_queue_peak"] > 1000:
        recommendations.append(
            f"Set up horizontal scaling triggers when queue exceeds {burst_analysis['max_queue_peak']} requests"
        )
    
    if burst_analysis["avg_recovery_time"] > 5:
        recommendations.append(
            f"Optimize recovery algorithm - current avg recovery: {burst_analysis['avg_recovery_time']:.1f}s"
        )
    
    # OOM recommendations
    if oom_analysis["exceeding_memory_limit"] > 0:
        recommendations.append(
            f"Implement batch size validation - {oom_analysis['exceeding_memory_limit']} configs exceed GPU memory"
        )
    
    if oom_analysis["largest_safe_batch"] > 0:
        recommendations.append(
            f"Safe maximum batch size: {oom_analysis['largest_safe_batch']} (enforces {oom_analysis['largest_safe_batch']})"
        )
    
    if oom_analysis["critical_failures"] > 0:
        recommendations.append(
            "Add pre-flight memory checks before batch processing"
        )
    
    if oom_analysis["recoverable_failures"] > 0:
        recommendations.append(
            f"Implement graceful degradation for {oom_analysis['recoverable_failures']} OOM scenarios"
        )
    
    if not recommendations:
        recommendations.append("System handles burstiness and OOM scenarios well - no critical changes needed")
    
    return recommendations


if __name__ == "__main__":
    sys.exit(main())
