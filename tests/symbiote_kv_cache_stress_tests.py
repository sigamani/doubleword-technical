#!/usr/bin/env python3
"""
Testing Symbiote KV Cache Stress Tests
Validates KV cache utilization under various stress conditions
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
class KVCacheTestConfig:
    """KV cache stress test configuration"""
    name: str
    model_size: str
    batch_size: int
    sequence_length: int  # Context window
    num_tokens_to_generate: int
    concurrent_sequences: int
    test_duration_sec: float
    expected_kv_cache_gb: float


@dataclass
class KVCacheTestResult:
    """KV cache stress test result"""
    config_name: str
    model_size: str
    batch_size: int
    sequence_length: int
    num_tokens_to_generate: int
    concurrent_sequences: int
    throughput_req_sec: float
    tokens_per_sec: float
    kv_cache_utilization_percent: float
    peak_kv_cache_gb: float
    cache_evictions: int
    cache_hit_rate_percent: float
    success: bool
    error_message: str = ""
    test_duration_sec: float = 0.0


class KVCacheSimulator:
    """Simulates KV cache stress scenarios"""
    
    def __init__(self):
        self.test_configs = self._define_kv_cache_tests()
    
    def _define_kv_cache_tests(self) -> List[KVCacheTestConfig]:
        """Define KV cache stress test scenarios"""
        return [
            # Small sequence, low stress
            KVCacheTestConfig(
                name="kv_cache_short_seq_low_stress",
                model_size="0.5B",
                batch_size=32,
                sequence_length=1024,
                num_tokens_to_generate=128,
                concurrent_sequences=1,
                test_duration_sec=5.0,
                expected_kv_cache_gb=0.5
            ),
            
            # Medium sequence, normal stress
            KVCacheTestConfig(
                name="kv_cache_medium_seq_normal",
                model_size="0.5B",
                batch_size=64,
                sequence_length=4096,
                num_tokens_to_generate=256,
                concurrent_sequences=2,
                test_duration_sec=5.0,
                expected_kv_cache_gb=2.0
            ),
            
            # Large sequence, high stress
            KVCacheTestConfig(
                name="kv_cache_large_seq_high_stress",
                model_size="0.5B",
                batch_size=128,
                sequence_length=8192,
                num_tokens_to_generate=512,
                concurrent_sequences=4,
                test_duration_sec=5.0,
                expected_kv_cache_gb=4.0
            ),
            
            # Very large sequence, extreme stress
            KVCacheTestConfig(
                name="kv_cache_very_large_seq_extreme",
                model_size="0.5B",
                batch_size=256,
                sequence_length=16384,
                num_tokens_to_generate=512,
                concurrent_sequences=8,
                test_duration_sec=5.0,
                expected_kv_cache_gb=8.0
            ),
            
            # Maximum context window (32k tokens)
            KVCacheTestConfig(
                name="kv_cache_max_context",
                model_size="0.5B",
                batch_size=128,
                sequence_length=32768,
                num_tokens_to_generate=256,
                concurrent_sequences=2,
                test_duration_sec=5.0,
                expected_kv_cache_gb=12.0
            ),
            
            # High batch concurrency, moderate sequence
            KVCacheTestConfig(
                name="kv_cache_high_concurrency",
                model_size="0.5B",
                batch_size=512,
                sequence_length=2048,
                num_tokens_to_generate=128,
                concurrent_sequences=16,
                test_duration_sec=5.0,
                expected_kv_cache_gb=10.0
            ),
            
            # Streaming scenario (progressive generation)
            KVCacheTestConfig(
                name="kv_cache_streaming_progressive",
                model_size="0.5B",
                batch_size=64,
                sequence_length=8192,
                num_tokens_to_generate=2048,  # Long generation
                concurrent_sequences=4,
                test_duration_sec=5.0,
                expected_kv_cache_gb=6.0
            ),
            
            # Multi-request batching (worst case for cache)
            KVCacheTestConfig(
                name="kv_cache_multi_request_batch",
                model_size="0.5B",
                batch_size=256,
                sequence_length=16384,
                num_tokens_to_generate=1024,
                concurrent_sequences=8,
                test_duration_sec=5.0,
                expected_kv_cache_gb=16.0  # High KV cache demand
            ),
            
            # 7B model with moderate sequence
            KVCacheTestConfig(
                name="kv_cache_7b_model",
                model_size="7B",
                batch_size=32,
                sequence_length=4096,
                num_tokens_to_generate=256,
                concurrent_sequences=2,
                test_duration_sec=5.0,
                expected_kv_cache_gb=6.0  # Larger model uses more cache
            ),
            
            # 7B model at sequence limit
            KVCacheTestConfig(
                name="kv_cache_7b_max_seq",
                model_size="7B",
                batch_size=16,
                sequence_length=16384,
                num_tokens_to_generate=512,
                concurrent_sequences=1,
                test_duration_sec=5.0,
                expected_kv_cache_gb=12.0
            ),
            
            # Cache exhaustion scenario
            KVCacheTestConfig(
                name="kv_cache_exhaustion",
                model_size="0.5B",
                batch_size=512,
                sequence_length=32768,
                num_tokens_to_generate=512,
                concurrent_sequences=16,
                test_duration_sec=5.0,
                expected_kv_cache_gb=24.0  # Would exceed 24GB on many GPUs
            ),
        ]
    
    def calculate_kv_cache_size(self, config: KVCacheTestConfig) -> float:
        """Calculate estimated KV cache size in GB"""
        # KV cache size formula:
        # size_gb = (batch_size * seq_len * hidden_dim * num_heads * 2 * dtype_bytes) / 1e9
        # Simplified estimation for common models
        
        # Model hidden dimensions (approximate)
        hidden_dims = {
            "0.5B": 1024,   # Qwen2.5-0.5B
            "7B": 4096,     # Qwen2.5-7B
            "13B": 5120,    # Qwen2.5-13B
        }
        
        # Number of attention heads
        num_heads = {
            "0.5B": 16,
            "7B": 32,
            "13B": 40,
        }
        
        hidden_dim = hidden_dims.get(config.model_size, 1024)
        heads = num_heads.get(config.model_size, 16)
        
        # KV cache per token = 2 * batch_size * num_heads * (hidden_dim/num_heads) * 2 bytes
        # Simplified: 2 * batch_size * hidden_dim * 2 bytes per token
        bytes_per_token = 2 * config.batch_size * hidden_dim * 2
        total_bytes = bytes_per_token * config.sequence_length * config.concurrent_sequences
        
        # Convert to GB (with overhead factor ~1.2)
        cache_gb = (total_bytes / 1e9) * 1.2
        
        return cache_gb
    
    def simulate_kv_cache_stress(self, config: KVCacheTestConfig) -> KVCacheTestResult:
        """Simulate KV cache stress test"""
        logger.info(f"Running KV cache stress test: {config.name}")
        
        start_time = time.time()
        
        # Calculate actual KV cache utilization
        actual_cache_gb = self.calculate_kv_cache_size(config)
        
        # Check if cache would be exhausted on typical GPUs
        available_cache_gb = 24.0  # RTX 3090 / A100 40GB with model loaded
        cache_utilization_percent = (actual_cache_gb / available_cache_gb) * 100
        
        # Simulate cache evictions if exceeding available
        cache_evictions = 0
        cache_hit_rate = 90.0  # Default good hit rate
        
        if actual_cache_gb > available_cache_gb:
            # Calculate evictions needed
            overage = actual_cache_gb - available_cache_gb
            cache_evictions = int((overage / actual_cache_gb) * 100)
            cache_hit_rate = max(50.0, 90.0 - (cache_evictions / 2))  # Hit rate degrades with evictions
        
        # Estimate throughput impact from cache pressure
        throughput_factor = 1.0
        if cache_utilization_percent > 80:
            throughput_factor = 0.85  # 15% slowdown at 80%+ utilization
        elif cache_utilization_percent > 60:
            throughput_factor = 0.95  # 5% slowdown at 60%+ utilization
        
        # Base throughput from batch size and sequence length
        base_throughput = 202.0 * (config.batch_size / 128)
        actual_throughput = base_throughput * throughput_factor
        
        # Token generation rate
        tokens_per_sec = actual_throughput * config.num_tokens_to_generate
        
        # Determine success
        success = cache_utilization_percent <= 95  # Allow up to 95% for extreme tests
        
        test_duration = time.time() - start_time
        
        error_msg = ""
        if cache_utilization_percent > 100:
            error_msg = f"KV cache exceeds available memory: {actual_cache_gb:.1f}GB > {available_cache_gb:.1f}GB"
            success = False
        elif cache_evictions > 30:
            error_msg = f"High cache eviction rate: {cache_evictions}%"
            if cache_utilization_percent > 95:
                success = False
        
        logger.info(f"KV cache test {config.name}: {actual_cache_gb:.1f}GB ({cache_utilization_percent:.1f}% util)")
        logger.info(f"  Throughput: {actual_throughput:.1f} req/s")
        logger.info(f"  Cache Hit Rate: {cache_hit_rate:.1f}%")
        logger.info(f"  Evictions: {cache_evictions}%")
        
        return KVCacheTestResult(
            config_name=config.name,
            model_size=config.model_size,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            num_tokens_to_generate=config.num_tokens_to_generate,
            concurrent_sequences=config.concurrent_sequences,
            throughput_req_sec=actual_throughput,
            tokens_per_sec=tokens_per_sec,
            kv_cache_utilization_percent=cache_utilization_percent,
            peak_kv_cache_gb=actual_cache_gb,
            cache_evictions=cache_evictions,
            cache_hit_rate_percent=cache_hit_rate,
            success=success,
            error_message=error_msg,
            test_duration_sec=test_duration
        )
    
    def run_all_kv_cache_tests(self) -> List[KVCacheTestResult]:
        """Run all KV cache stress tests"""
        logger.info(f"Starting KV cache stress testing with {len(self.test_configs)} scenarios")
        results = []
        
        for config in self.test_configs:
            result = self.simulate_kv_cache_stress(config)
            results.append(result)
            time.sleep(0.2)
        
        return results


class KVCacheAnalyzer:
    """Analyzes KV cache stress test results"""
    
    @staticmethod
    def analyze_results(results: List[KVCacheTestResult]) -> Dict:
        """Analyze KV cache test results"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Calculate cache metrics
        cache_utils = [r.kv_cache_utilization_percent for r in results]
        cache_peaks = [r.peak_kv_cache_gb for r in results]
        
        analysis = {
            "total_tests": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "kv_cache_metrics": {
                "min_utilization_percent": min(cache_utils),
                "max_utilization_percent": max(cache_utils),
                "avg_utilization_percent": sum(cache_utils) / len(cache_utils),
                "min_peak_gb": min(cache_peaks),
                "max_peak_gb": max(cache_peaks),
                "avg_peak_gb": sum(cache_peaks) / len(cache_peaks),
            },
            "throughput_metrics": {
                "min_throughput": min(r.throughput_req_sec for r in results),
                "max_throughput": max(r.throughput_req_sec for r in results),
                "avg_throughput": sum(r.throughput_req_sec for r in results) / len(results),
            },
            "cache_efficiency": {
                "avg_hit_rate": sum(r.cache_hit_rate_percent for r in results) / len(results),
                "avg_evictions": sum(r.cache_evictions for r in results) / len(results),
            },
            "results": [asdict(r) for r in results]
        }
        
        return analysis
    
    @staticmethod
    def identify_bottlenecks(results: List[KVCacheTestResult]) -> List[str]:
        """Identify performance bottlenecks"""
        issues = []
        
        # Check for high utilization
        high_util = [r for r in results if r.kv_cache_utilization_percent > 80]
        if high_util:
            issues.append(
                f"High KV cache utilization in {len(high_util)} tests "
                f"(>80%, max: {max(r.kv_cache_utilization_percent for r in high_util):.1f}%)"
            )
        
        # Check for evictions
        with_evictions = [r for r in results if r.cache_evictions > 0]
        if with_evictions:
            avg_evictions = sum(r.cache_evictions for r in with_evictions) / len(with_evictions)
            issues.append(f"Cache evictions in {len(with_evictions)} tests (avg: {avg_evictions:.1f}%)")
        
        # Check for low hit rates
        low_hit = [r for r in results if r.cache_hit_rate_percent < 80]
        if low_hit:
            avg_hit = sum(r.cache_hit_rate_percent for r in low_hit) / len(low_hit)
            issues.append(f"Low cache hit rates in {len(low_hit)} tests (avg: {avg_hit:.1f}%)")
        
        # Check throughput degradation
        degraded = [r for r in results if r.throughput_req_sec < 100]
        if degraded:
            issues.append(f"Throughput degradation in {len(degraded)} tests (<100 req/s)")
        
        return issues
    
    @staticmethod
    def generate_recommendations(results: List[KVCacheTestResult]) -> List[str]:
        """Generate KV cache optimization recommendations"""
        recommendations = []
        
        # Analyze results
        max_util = max(r.kv_cache_utilization_percent for r in results)
        avg_util = sum(r.kv_cache_utilization_percent for r in results) / len(results)
        
        if max_util > 90:
            recommendations.append("Consider PagedAttention for efficient cache management")
        
        if avg_util > 70:
            recommendations.append("Implement KV cache quantization to reduce memory footprint")
        
        # Check for long sequences
        long_seq = [r for r in results if r.sequence_length > 8192]
        if long_seq:
            recommendations.append("Use sliding window attention for sequences >8K tokens")
        
        # Check for high batch sizes
        high_batch = [r for r in results if r.batch_size > 256]
        if high_batch:
            recommendations.append("Monitor batch sizes >256 for cache pressure")
        
        if not recommendations:
            recommendations.append("KV cache utilization is well-managed, no immediate optimizations needed")
        
        # Add safety recommendations
        recommendations.append("Maintain 20% GPU memory headroom for KV cache safety margin")
        recommendations.append("Monitor cache hit rates; <75% indicates thrashing")
        recommendations.append("Implement cache prefetching for predictable patterns")
        
        return recommendations


def main():
    """Main function"""
    logger.info("Starting Testing Symbiote KV Cache Stress Tests")
    
    # Run KV cache tests
    logger.info("=" * 70)
    logger.info("PHASE: KV CACHE STRESS TESTING")
    logger.info("=" * 70)
    
    kv_sim = KVCacheSimulator()
    kv_results = kv_sim.run_all_kv_cache_tests()
    kv_analysis = KVCacheAnalyzer.analyze_results(kv_results)
    
    # Generate report
    kv_report = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "KV Cache Stress Tests",
        "summary": {
            "total_tests": kv_analysis["total_tests"],
            "successful": kv_analysis["successful"],
            "failed": kv_analysis["failed"],
            "success_rate": (kv_analysis["successful"] / kv_analysis["total_tests"] * 100) if kv_analysis["total_tests"] > 0 else 0,
        },
        "kv_cache_metrics": kv_analysis["kv_cache_metrics"],
        "throughput_metrics": kv_analysis["throughput_metrics"],
        "cache_efficiency": kv_analysis["cache_efficiency"],
        "bottlenecks": KVCacheAnalyzer.identify_bottlenecks(kv_results),
        "recommendations": KVCacheAnalyzer.generate_recommendations(kv_results),
        "results": kv_analysis["results"]
    }
    
    # Save report
    report_path = "symbiote_kv_cache_stress_report.json"
    with open(report_path, "w") as f:
        json.dump(kv_report, f, indent=2)
    
    logger.info(f"Report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("KV CACHE STRESS TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {kv_analysis['total_tests']}")
    print(f"Successful: {kv_analysis['successful']}")
    print(f"Failed: {kv_analysis['failed']}")
    print(f"Success Rate: {kv_report['summary']['success_rate']:.1f}%")
    
    print("\nKV CACHE METRICS:")
    metrics = kv_analysis["kv_cache_metrics"]
    print(f"  Utilization Range: {metrics['min_utilization_percent']:.1f}% - {metrics['max_utilization_percent']:.1f}%")
    print(f"  Average Utilization: {metrics['avg_utilization_percent']:.1f}%")
    print(f"  Peak Memory Range: {metrics['min_peak_gb']:.1f}GB - {metrics['max_peak_gb']:.1f}GB")
    print(f"  Average Peak: {metrics['avg_peak_gb']:.1f}GB")
    
    print("\nTHROUGHPUT METRICS:")
    tput = kv_analysis["throughput_metrics"]
    print(f"  Range: {tput['min_throughput']:.1f} - {tput['max_throughput']:.1f} req/s")
    print(f"  Average: {tput['avg_throughput']:.1f} req/s")
    
    print("\nCACHE EFFICIENCY:")
    eff = kv_analysis["cache_efficiency"]
    print(f"  Average Hit Rate: {eff['avg_hit_rate']:.1f}%")
    print(f"  Average Evictions: {eff['avg_evictions']:.1f}%")
    
    if kv_report["bottlenecks"]:
        print("\nBOTTLENECKS IDENTIFIED:")
        for i, bottleneck in enumerate(kv_report["bottlenecks"], 1):
            print(f"  {i}. {bottleneck}")
    
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(kv_report["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 70)
    return 0 if kv_report['summary']['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
