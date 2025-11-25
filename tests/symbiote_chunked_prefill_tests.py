"""
Chunked Prefill Tests for Symbiote
Tests prefill chunking strategies, latencies, and throughput impact
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from typing import List, Dict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("symbiote_chunked_prefill")


@dataclass
class PrefillConfig:
    """Configuration for prefill strategy"""
    name: str
    chunk_size: int
    description: str


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    num_params: int
    hidden_dim: int
    num_heads: int
    num_layers: int


@dataclass
class PrefillResult:
    """Result of prefill test"""
    model_name: str
    chunk_size: int
    prompt_length: int
    batch_size: int
    prefill_time_ms: float
    decode_time_ms: float
    total_latency_ms: float
    throughput_tokens_sec: float
    memory_peak_gb: float
    tokens_per_prefill_ms: float
    prefill_efficiency: float
    status: str


class ChunkedPrefillCalculator:
    """Calculate chunked prefill performance"""
    
    def __init__(self):
        self.results: List[PrefillResult] = []
        self.strategies = {
            "no_chunking": (8192, "Full prefill in one pass"),
            "medium_chunk": (512, "512 token chunks"),
            "small_chunk": (256, "256 token chunks"),
            "token_budget": (1024, "1K token chunks (vLLM default)"),
            "adaptive": (2048, "2K chunks, reduces under load"),
        }
        
        self.models = {
            "qwen2.5-0.5b": ModelConfig("qwen2.5-0.5b", 500_000_000, 896, 14, 24),
            "qwen2.5-7b": ModelConfig("qwen2.5-7b", 7_000_000_000, 4096, 32, 32),
        }
    
    def calc_prefill_latency(
        self, 
        prompt_length: int, 
        chunk_size: int, 
        hidden_dim: int, 
        num_heads: int, 
        num_layers: int,
        gpu_tflops: float = 80.0
    ) -> float:
        """Calculate prefill latency in milliseconds"""
        num_chunks = max(1, math.ceil(prompt_length / chunk_size))
        tokens_per_chunk = min(chunk_size, prompt_length)
        
        head_dim = hidden_dim // num_heads
        attention_flops = 2 * tokens_per_chunk * tokens_per_chunk * hidden_dim
        ffn_dim = int(hidden_dim * 4)
        mlp_flops = 2 * tokens_per_chunk * hidden_dim * ffn_dim
        total_flops_per_chunk = (attention_flops + mlp_flops) * num_layers
        
        time_per_chunk_ms = (total_flops_per_chunk / (gpu_tflops * 1e12)) * 1000
        return time_per_chunk_ms * num_chunks
    
    def calc_decode_latency(
        self, 
        output_tokens: int = 128, 
        hidden_dim: int = 4096, 
        num_heads: int = 32, 
        num_layers: int = 32,
        gpu_tflops: float = 80.0
    ) -> float:
        """Calculate decode latency (token generation) in milliseconds"""
        head_dim = hidden_dim // num_heads
        attention_flops = 2 * 1 * 1 * hidden_dim
        ffn_dim = int(hidden_dim * 4)
        mlp_flops = 2 * 1 * hidden_dim * ffn_dim
        total_flops_per_token = (attention_flops + mlp_flops) * num_layers
        
        time_per_token_ms = (total_flops_per_token / (gpu_tflops * 1e12)) * 1000
        return time_per_token_ms * output_tokens
    
    def calc_memory_usage(
        self, 
        chunk_size: int, 
        hidden_dim: int, 
        num_heads: int, 
        num_layers: int,
        batch_size: int = 1,
        seq_len: int = 2048
    ) -> float:
        """Calculate peak memory during prefill in GB"""
        head_dim = hidden_dim // num_heads
        
        activations_per_layer = batch_size * chunk_size * hidden_dim * 2
        attention_kv = batch_size * seq_len * hidden_dim * 2
        total_activations = (activations_per_layer * num_layers + attention_kv) * 2
        
        return (total_activations / (1024 ** 3))
    
    def test_prefill_strategy(
        self, 
        model_name: str,
        strategy_name: str,
        prompt_length: int,
        batch_size: int = 1,
        output_tokens: int = 128
    ) -> PrefillResult:
        """Test a prefill strategy"""
        model = self.models[model_name]
        chunk_size, _ = self.strategies[strategy_name]
        
        prefill_time = self.calc_prefill_latency(
            prompt_length, chunk_size, model.hidden_dim, 
            model.num_heads, model.num_layers
        )
        
        decode_time = self.calc_decode_latency(
            output_tokens, model.hidden_dim, 
            model.num_heads, model.num_layers
        ) * batch_size
        
        total_latency = prefill_time + decode_time
        total_tokens = prompt_length + output_tokens
        throughput = (total_tokens / total_latency * 1000) if total_latency > 0 else 0
        
        memory_peak = self.calc_memory_usage(
            chunk_size, model.hidden_dim, model.num_heads, 
            model.num_layers, batch_size
        )
        
        efficiency = prompt_length / prefill_time if prefill_time > 0 else 0
        
        status = "OK"
        if prefill_time > 100:
            status = "SLOW_PREFILL"
        if memory_peak > 20:
            status = "HIGH_MEMORY"
        if efficiency < 5:
            status = "LOW_EFFICIENCY"
        
        result = PrefillResult(
            model_name=model_name,
            chunk_size=chunk_size,
            prompt_length=prompt_length,
            batch_size=batch_size,
            prefill_time_ms=round(prefill_time, 2),
            decode_time_ms=round(decode_time, 2),
            total_latency_ms=round(total_latency, 2),
            throughput_tokens_sec=round(throughput, 2),
            memory_peak_gb=round(memory_peak, 3),
            tokens_per_prefill_ms=round(efficiency, 2),
            prefill_efficiency=round(efficiency * 100, 1),
            status=status
        )
        
        self.results.append(result)
        return result
    
    def run_full_matrix(self) -> List[PrefillResult]:
        """Run prefill tests for all combinations"""
        logger.info("Starting chunked prefill test matrix...")
        
        prompt_lengths = [512, 2048, 8192]
        models = ["qwen2.5-0.5b", "qwen2.5-7b"]
        
        test_count = 0
        for model_name in models:
            for prompt_length in prompt_lengths:
                for strategy_name in self.strategies.keys():
                    result = self.test_prefill_strategy(
                        model_name, strategy_name, prompt_length
                    )
                    test_count += 1
                    
                    status_symbol = "OK" if result.prefill_time_ms < 50 else "SLOW"
                    logger.info(
                        f"{status_symbol} {model_name} | Prompt: {prompt_length:5} | "
                        f"Strategy: {strategy_name:15} | "
                        f"Chunk: {result.chunk_size:5} | "
                        f"Prefill: {result.prefill_time_ms:7.2f}ms | "
                        f"Eff: {result.tokens_per_prefill_ms:6.2f} tok/ms"
                    )
        
        logger.info(f"Completed {test_count} prefill tests")
        return self.results


class ChunkedPrefillAnalyzer:
    """Analyze prefill test results"""
    
    def __init__(self, results: List[PrefillResult]):
        self.results = results
    
    def get_latency_by_strategy(self) -> Dict[str, float]:
        """Get average latency by strategy"""
        by_chunk = {}
        for r in self.results:
            key = r.chunk_size
            if key not in by_chunk:
                by_chunk[key] = []
            by_chunk[key].append(r.prefill_time_ms)
        
        result = {}
        for chunk_size, latencies in by_chunk.items():
            result[chunk_size] = sum(latencies) / len(latencies)
        return result
    
    def generate_report(self) -> str:
        """Generate analysis report"""
        report = []
        report.append("=" * 100)
        report.append("CHUNKED PREFILL ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 100)
        
        report.append("\nPREFILL LATENCY BY STRATEGY:")
        report.append("-" * 100)
        latency_by_strategy = self.get_latency_by_strategy()
        for chunk_size, avg_latency in sorted(latency_by_strategy.items()):
            report.append(f"Chunk Size {chunk_size:5} | Average Latency: {avg_latency:8.2f}ms")
        
        report.append("\nDETAILED RESULTS BY MODEL AND PROMPT LENGTH:")
        report.append("-" * 100)
        
        models = sorted(set(r.model_name for r in self.results))
        prompt_lengths = sorted(set(r.prompt_length for r in self.results))
        
        for model_name in models:
            for prompt_length in prompt_lengths:
                matching = [r for r in self.results 
                           if r.model_name == model_name and r.prompt_length == prompt_length]
                if matching:
                    report.append(f"\n{model_name} | Prompt: {prompt_length}")
                    matching = sorted(matching, key=lambda r: r.prefill_time_ms)
                    for r in matching:
                        report.append(
                            f"  Chunk {r.chunk_size:5} | "
                            f"Prefill: {r.prefill_time_ms:8.2f}ms | "
                            f"Decode: {r.decode_time_ms:8.2f}ms | "
                            f"Total: {r.total_latency_ms:8.2f}ms | "
                            f"Throughput: {r.throughput_tokens_sec:7.2f} tok/s | "
                            f"Mem: {r.memory_peak_gb:6.3f}GB"
                        )
        
        report.append("\nKEY METRICS:")
        report.append("-" * 100)
        prefill_times = [r.prefill_time_ms for r in self.results]
        efficiencies = [r.tokens_per_prefill_ms for r in self.results]
        report.append(f"Avg Prefill Latency: {sum(prefill_times)/len(prefill_times):.2f}ms")
        report.append(f"Min Prefill Latency: {min(prefill_times):.2f}ms")
        report.append(f"Max Prefill Latency: {max(prefill_times):.2f}ms")
        report.append(f"Avg Efficiency: {sum(efficiencies)/len(efficiencies):.2f} tok/ms")
        
        report.append("\nRECOMMENDATIONS:")
        report.append("-" * 100)
        report.append("1. Use token_budget (1K chunks) for balanced latency/throughput")
        report.append("2. Use small_chunk (256) for latency-sensitive workloads")
        report.append("3. Use adaptive chunking to adjust based on system load")
        report.append("4. Monitor prefill latency to stay below 50ms for best UX")
        
        return "\n".join(report)


def main():
    """Run chunked prefill tests"""
    calculator = ChunkedPrefillCalculator()
    results = calculator.run_full_matrix()
    
    analyzer = ChunkedPrefillAnalyzer(results)
    report = analyzer.generate_report()
    
    print(report)
    
    with open("symbiote_chunked_prefill_report.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("Results saved to symbiote_chunked_prefill_report.json")
    
    with open("symbiote_chunked_prefill_analysis.txt", "w") as f:
        f.write(report)
    logger.info("Analysis saved to symbiote_chunked_prefill_analysis.txt")


if __name__ == "__main__":
    main()
