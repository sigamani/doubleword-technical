import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CostMetrics:
    """Cost tracking for derived metrics"""
    gpu_cost_per_hour: float = 2.0  # $2/hour per GPU
    cpu_cost_per_hour: float = 0.1  # $0.1/hour per CPU
    memory_cost_per_gb: float = 0.05  # $0.05/GB/hour
    storage_cost_per_gb: float = 0.01  # $0.01/GB/month
    network_cost_per_gb: float = 0.01  # $0.01/GB transferred

@dataclass
class DerivedMetrics:
    """Derived metrics calculated from raw metrics"""
    # Efficiency metrics
    tokens_per_dollar: float = 0.0
    requests_per_dollar: float = 0.0
    effective_batch_utilization: float = 0.0
    
    # Memory and preemption metrics
    preemptions_per_1k_tokens: float = 0.0
    cache_hit_rate: float = 0.0
    memory_efficiency: float = 0.0
    
    # Latency and queue metrics
    queue_wait_ratio: float = 0.0
    sla_risk_score: float = 0.0
    eta_accuracy: float = 0.0
    
    # Cost metrics
    cost_per_1k_tokens: float = 0.0
    cost_per_request: float = 0.0
    hourly_cost: float = 0.0
    
    # Performance metrics
    throughput_variance: float = 0.0
    latency_p99: float = 0.0
    error_recovery_time: float = 0.0

class DerivedMetricsCalculator:
    """Calculator for derived metrics"""
    
    def __init__(self, cost_metrics: CostMetrics = None):
        self.cost_metrics = cost_metrics or CostMetrics()
        self.metrics_history = []
        self.max_history_size = 1000
        
    def calculate_derived_metrics(self, raw_metrics: Dict[str, Any], 
                                 context: Dict[str, Any] = None) -> DerivedMetrics:
        """Calculate all derived metrics from raw metrics"""
        context = context or {}
        derived = DerivedMetrics()
        
        # Efficiency metrics
        derived.tokens_per_dollar = self._calculate_tokens_per_dollar(raw_metrics)
        derived.requests_per_dollar = self._calculate_requests_per_dollar(raw_metrics)
        derived.effective_batch_utilization = self._calculate_effective_batch_utilization(raw_metrics)
        
        # Memory and preemption metrics
        derived.preemptions_per_1k_tokens = self._calculate_preemptions_per_1k_tokens(raw_metrics)
        derived.cache_hit_rate = self._calculate_cache_hit_rate(raw_metrics)
        derived.memory_efficiency = self._calculate_memory_efficiency(raw_metrics)
        
        # Latency and queue metrics
        derived.queue_wait_ratio = self._calculate_queue_wait_ratio(raw_metrics)
        derived.sla_risk_score = self._calculate_sla_risk_score(raw_metrics)
        derived.eta_accuracy = self._calculate_eta_accuracy(raw_metrics)
        
        # Cost metrics
        derived.cost_per_1k_tokens = self._calculate_cost_per_1k_tokens(raw_metrics, context)
        derived.cost_per_request = self._calculate_cost_per_request(raw_metrics, context)
        derived.hourly_cost = self._calculate_hourly_cost(context)
        
        # Performance metrics
        derived.throughput_variance = self._calculate_throughput_variance(raw_metrics)
        derived.latency_p99 = self._calculate_latency_p99(raw_metrics)
        derived.error_recovery_time = self._calculate_error_recovery_time(raw_metrics)
        
        # Update history
        self._update_history(raw_metrics, derived)
        
        return derived
    
    def _calculate_tokens_per_dollar(self, metrics: Dict[str, Any]) -> float:
        """Calculate tokens per dollar"""
        tokens_processed = metrics.get("tokens_processed", 0)
        hourly_cost = self._calculate_hourly_cost({})
        
        if hourly_cost == 0:
            return 0.0
        
        # Tokens per hour divided by cost per hour
        tokens_per_hour = metrics.get("tokens_per_sec", 0) * 3600
        return tokens_per_hour / hourly_cost
    
    def _calculate_requests_per_dollar(self, metrics: Dict[str, Any]) -> float:
        """Calculate requests per dollar"""
        requests_processed = metrics.get("completed_requests", 0)
        hourly_cost = self._calculate_hourly_cost({})
        
        if hourly_cost == 0:
            return 0.0
        
        requests_per_hour = metrics.get("throughput_req_per_sec", 0) * 3600
        return requests_per_hour / hourly_cost
    
    def _calculate_effective_batch_utilization(self, metrics: Dict[str, Any]) -> float:
        """Calculate effective batch utilization"""
        actual_batch_size = metrics.get("actual_batch_size", 0)
        target_batch_size = metrics.get("target_batch_size", 1)
        
        if target_batch_size == 0:
            return 0.0
        
        return actual_batch_size / target_batch_size
    
    def _calculate_preemptions_per_1k_tokens(self, metrics: Dict[str, Any]) -> float:
        """Calculate preemptions per 1k tokens"""
        preemption_count = metrics.get("preemption_count", 0)
        tokens_processed = metrics.get("tokens_processed", 0)
        
        if tokens_processed == 0:
            return 0.0
        
        return (preemption_count * 1000) / tokens_processed
    
    def _calculate_cache_hit_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate cache hit rate"""
        cache_hits = metrics.get("cache_hits", 0)
        cache_misses = metrics.get("cache_misses", 0)
        total_accesses = cache_hits + cache_misses
        
        if total_accesses == 0:
            return 0.0
        
        return cache_hits / total_accesses
    
    def _calculate_memory_efficiency(self, metrics: Dict[str, Any]) -> float:
        """Calculate memory efficiency"""
        utilized_memory = metrics.get("utilized_memory", 0)
        total_memory = metrics.get("total_memory", 1)
        
        if total_memory == 0:
            return 0.0
        
        return utilized_memory / total_memory
    
    def _calculate_queue_wait_ratio(self, metrics: Dict[str, Any]) -> float:
        """Calculate queue wait ratio"""
        queue_wait_time = metrics.get("queue_wait_time", 0)
        total_latency = metrics.get("total_latency", 1)
        
        if total_latency == 0:
            return 0.0
        
        return queue_wait_time / total_latency
    
    def _calculate_sla_risk_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate SLA risk score (0-1, higher = more risk)"""
        eta_hours = metrics.get("eta_hours", 0)
        remaining_sla_hours = metrics.get("remaining_sla_hours", 24)
        
        if remaining_sla_hours <= 0:
            return 1.0
        
        # Risk ratio: if ETA > remaining time, risk = 1
        if eta_hours > remaining_sla_hours:
            return 1.0
        
        # Linear risk based on how close ETA is to remaining time
        risk_ratio = eta_hours / remaining_sla_hours
        return min(1.0, risk_ratio * 2)  # Scale to make it more sensitive
    
    def _calculate_eta_accuracy(self, metrics: Dict[str, Any]) -> float:
        """Calculate ETA prediction accuracy"""
        predicted_eta = metrics.get("predicted_eta_hours", 0)
        actual_eta = metrics.get("actual_eta_hours", 0)
        
        if predicted_eta == 0:
            return 0.0
        
        # Accuracy as 1 - relative error
        relative_error = abs(predicted_eta - actual_eta) / predicted_eta
        return max(0.0, 1.0 - relative_error)
    
    def _calculate_cost_per_1k_tokens(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate cost per 1k tokens"""
        hourly_cost = self._calculate_hourly_cost(context)
        tokens_per_hour = metrics.get("tokens_per_sec", 0) * 3600
        
        if tokens_per_hour == 0:
            return 0.0
        
        cost_per_token = hourly_cost / tokens_per_hour
        return cost_per_token * 1000
    
    def _calculate_cost_per_request(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate cost per request"""
        hourly_cost = self._calculate_hourly_cost(context)
        requests_per_hour = metrics.get("throughput_req_per_sec", 0) * 3600
        
        if requests_per_hour == 0:
            return 0.0
        
        return hourly_cost / requests_per_hour
    
    def _calculate_hourly_cost(self, context: Dict[str, Any]) -> float:
        """Calculate hourly operational cost"""
        gpu_count = context.get("gpu_count", 1)
        cpu_count = context.get("cpu_count", 4)
        memory_gb = context.get("memory_gb", 16)
        storage_gb = context.get("storage_gb", 100)
        
        gpu_cost = gpu_count * self.cost_metrics.gpu_cost_per_hour
        cpu_cost = cpu_count * self.cost_metrics.cpu_cost_per_hour
        memory_cost = memory_gb * self.cost_metrics.memory_cost_per_hour
        storage_cost = (storage_gb * self.cost_metrics.storage_cost_per_gb) / (30 * 24)  # Convert monthly to hourly
        
        return gpu_cost + cpu_cost + memory_cost + storage_cost
    
    def _calculate_throughput_variance(self, metrics: Dict[str, Any]) -> float:
        """Calculate throughput variance from history"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_throughputs = [
            entry["metrics"].get("throughput_req_per_sec", 0)
            for entry in self.metrics_history[-10:]  # Last 10 entries
        ]
        
        if len(recent_throughputs) < 2:
            return 0.0
        
        mean_throughput = sum(recent_throughputs) / len(recent_throughputs)
        variance = sum((x - mean_throughput) ** 2 for x in recent_throughputs) / len(recent_throughputs)
        
        # Return coefficient of variation (normalized variance)
        if mean_throughput == 0:
            return 0.0
        
        return (variance ** 0.5) / mean_throughput
    
    def _calculate_latency_p99(self, metrics: Dict[str, Any]) -> float:
        """Calculate 99th percentile latency"""
        latency_samples = metrics.get("latency_samples", [])
        
        if not latency_samples:
            return metrics.get("avg_latency", 0)
        
        sorted_latencies = sorted(latency_samples)
        p99_index = int(len(sorted_latencies) * 0.99)
        
        return sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]
    
    def _calculate_error_recovery_time(self, metrics: Dict[str, Any]) -> float:
        """Calculate average error recovery time"""
        recovery_times = metrics.get("error_recovery_times", [])
        
        if not recovery_times:
            return 0.0
        
        return sum(recovery_times) / len(recovery_times)
    
    def _update_history(self, raw_metrics: Dict[str, Any], derived_metrics: DerivedMetrics):
        """Update metrics history for trend analysis"""
        timestamp = time.time()
        
        self.metrics_history.append({
            "timestamp": timestamp,
            "raw_metrics": raw_metrics.copy(),
            "derived_metrics": derived_metrics
        })
        
        # Trim history
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def get_trend_analysis(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get trend analysis for a specific metric"""
        cutoff_time = time.time() - (window_minutes * 60)
        
        recent_entries = [
            entry for entry in self.metrics_history
            if entry["timestamp"] > cutoff_time
        ]
        
        if len(recent_entries) < 2:
            return {"trend": "insufficient_data"}
        
        # Extract metric values
        if metric_name.startswith("derived."):
            # Derived metric
            metric_field = metric_name.replace("derived.", "")
            values = [getattr(entry["derived_metrics"], metric_field, 0) for entry in recent_entries]
        else:
            # Raw metric
            values = [entry["raw_metrics"].get(metric_name, 0) for entry in recent_entries]
        
        # Calculate trend
        if len(values) < 2:
            return {"trend": "insufficient_data"}
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        
        if second_avg > first_avg * 1.1:
            trend = "increasing"
        elif second_avg < first_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "current_value": values[-1] if values else 0,
            "average_value": sum(values) / len(values),
            "min_value": min(values),
            "max_value": max(values),
            "sample_count": len(values)
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive derived metrics report"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        derived = latest["derived_metrics"]
        
        return {
            "timestamp": latest["timestamp"],
            "efficiency_metrics": {
                "tokens_per_dollar": derived.tokens_per_dollar,
                "requests_per_dollar": derived.requests_per_dollar,
                "effective_batch_utilization": derived.effective_batch_utilization
            },
            "memory_metrics": {
                "preemptions_per_1k_tokens": derived.preemptions_per_1k_tokens,
                "cache_hit_rate": derived.cache_hit_rate,
                "memory_efficiency": derived.memory_efficiency
            },
            "latency_metrics": {
                "queue_wait_ratio": derived.queue_wait_ratio,
                "sla_risk_score": derived.sla_risk_score,
                "eta_accuracy": derived.eta_accuracy,
                "latency_p99": derived.latency_p99
            },
            "cost_metrics": {
                "cost_per_1k_tokens": derived.cost_per_1k_tokens,
                "cost_per_request": derived.cost_per_request,
                "hourly_cost": derived.hourly_cost
            },
            "performance_metrics": {
                "throughput_variance": derived.throughput_variance,
                "error_recovery_time": derived.error_recovery_time
            }
        }

def create_derived_metrics_calculator(cost_config: Dict[str, Any] = None) -> DerivedMetricsCalculator:
    """Create derived metrics calculator"""
    if cost_config:
        cost_metrics = CostMetrics(**cost_config)
    else:
        cost_metrics = CostMetrics()
    
    return DerivedMetricsCalculator(cost_metrics)