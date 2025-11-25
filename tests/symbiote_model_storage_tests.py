"""
Model Weight Storage Size Tests for Symbiote
Tests quantization impact on weight storage and VRAM fit across GPU types
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("symbiote_model_storage")


@dataclass
class QuantizationScheme:
    """Represents a quantization format"""
    name: str
    bits_per_param: float
    description: str


@dataclass
class ModelSpec:
    """Model specification"""
    name: str
    num_params: int
    description: str


@dataclass
class GPUSpec:
    """GPU memory specification"""
    name: str
    total_memory_gb: int
    usable_memory_gb: int


@dataclass
class StorageResult:
    """Storage calculation result"""
    model_name: str
    gpu_name: str
    quantization: str
    weight_storage_gb: float
    activation_overhead_gb: float
    kv_cache_overhead_gb: float
    total_required_gb: float
    available_gb: float
    fits: bool
    utilization_percent: float
    status: str


class ModelStorageCalculator:
    """Calculate model weight storage requirements"""
    
    QUANTIZATION_SCHEMES = {
        "fp32": QuantizationScheme("FP32", 32, "Full precision (4 bytes/param)"),
        "fp16": QuantizationScheme("FP16", 16, "Half precision (2 bytes/param)"),
        "int8": QuantizationScheme("INT8", 8, "8-bit quantization (1 byte/param)"),
        "int4": QuantizationScheme("INT4", 4, "4-bit quantization (0.5 bytes/param)"),
    }
    
    MODELS = {
        "qwen2.5-0.5b": ModelSpec("qwen2.5-0.5b", 500_000_000, "0.5B parameters"),
        "qwen2.5-7b": ModelSpec("qwen2.5-7b", 7_000_000_000, "7B parameters"),
        "qwen2.5-13b": ModelSpec("qwen2.5-13b", 13_000_000_000, "13B parameters"),
    }
    
    GPUS = {
        "rtx3090": GPUSpec("RTX 3090", 24, 22),
        "a100-40gb": GPUSpec("A100 40GB", 40, 37),
        "h100": GPUSpec("H100 80GB", 80, 75),
        "rtx4090": GPUSpec("RTX 4090", 24, 22),
        "l40": GPUSpec("L40", 48, 45),
    }
    
    def __init__(self):
        self.results: List[StorageResult] = []
    
    def calc_weight_storage(self, num_params: int, quantization: str) -> float:
        """Calculate weight storage in GB"""
        scheme = self.QUANTIZATION_SCHEMES[quantization]
        bytes_per_param = scheme.bits_per_param / 8
        total_bytes = num_params * bytes_per_param
        return total_bytes / (1024 ** 3)
    
    def calc_activation_overhead(self, num_params: int, batch_size: int = 1, seq_len: int = 2048) -> float:
        """Estimate activation memory overhead during inference"""
        hidden_size = int((num_params / 4) ** 0.5)
        num_layers = 32
        bytes_per_activation = batch_size * seq_len * hidden_size * 4
        total_bytes = bytes_per_activation * num_layers * 2
        return total_bytes / (1024 ** 3)
    
    def calc_kv_cache_overhead(self, num_params: int, batch_size: int = 1, seq_len: int = 2048) -> float:
        """Estimate KV cache overhead for batch inference"""
        hidden_size = int((num_params / 4) ** 0.5)
        num_layers = 32
        head_dim = hidden_size // 32
        kv_per_seq = 2 * batch_size * seq_len * hidden_size * 2
        total_bytes = kv_per_seq * num_layers * 2
        return total_bytes / (1024 ** 3)
    
    def test_fit(self, model_name: str, gpu_name: str, quantization: str) -> StorageResult:
        """Test if model fits on GPU with given quantization"""
        model = self.MODELS[model_name]
        gpu = self.GPUS[gpu_name]
        
        weight_storage = self.calc_weight_storage(model.num_params, quantization)
        activation_overhead = self.calc_activation_overhead(model.num_params)
        kv_cache_overhead = self.calc_kv_cache_overhead(model.num_params)
        
        total_required = weight_storage + activation_overhead + kv_cache_overhead
        fits = total_required <= gpu.usable_memory_gb
        
        utilization = (total_required / gpu.usable_memory_gb * 100) if gpu.usable_memory_gb > 0 else 0
        
        status = "OK" if fits else "DOES NOT FIT"
        if utilization > 90:
            status = "TIGHT" if fits else "DOES NOT FIT"
        elif utilization > 75:
            status = "CAUTION" if fits else "DOES NOT FIT"
        
        result = StorageResult(
            model_name=model_name,
            gpu_name=gpu_name,
            quantization=quantization,
            weight_storage_gb=round(weight_storage, 3),
            activation_overhead_gb=round(activation_overhead, 3),
            kv_cache_overhead_gb=round(kv_cache_overhead, 3),
            total_required_gb=round(total_required, 3),
            available_gb=gpu.usable_memory_gb,
            fits=fits,
            utilization_percent=round(utilization, 1),
            status=status
        )
        
        self.results.append(result)
        return result
    
    def run_full_matrix(self) -> List[StorageResult]:
        """Run storage tests for all model/GPU/quantization combinations"""
        logger.info("Starting model storage matrix tests...")
        
        test_count = 0
        for model_name in self.MODELS:
            for gpu_name in self.GPUS:
                for quantization in self.QUANTIZATION_SCHEMES:
                    result = self.test_fit(model_name, gpu_name, quantization)
                    test_count += 1
                    
                    emoji_status = "✓" if result.fits else "✗"
                    logger.info(
                        f"{emoji_status} {result.model_name} on {result.gpu_name} "
                        f"with {result.quantization}: {result.total_required_gb}GB "
                        f"({result.utilization_percent}% utilization) - {result.status}"
                    )
        
        logger.info(f"Completed {test_count} storage tests")
        return self.results


class ModelStorageAnalyzer:
    """Analyze storage test results"""
    
    def __init__(self, results: List[StorageResult]):
        self.results = results
    
    def get_best_fit_per_model_gpu(self) -> Dict[Tuple[str, str], StorageResult]:
        """Get most efficient quantization per model-GPU pair"""
        best = {}
        for result in self.results:
            if result.fits:
                key = (result.model_name, result.gpu_name)
                if key not in best or result.utilization_percent < best[key].utilization_percent:
                    best[key] = result
        return best
    
    def get_fits_summary(self) -> Dict[str, Dict[str, int]]:
        """Summary of what fits on each GPU"""
        summary = {}
        for gpu_name in ModelStorageCalculator.GPUS:
            summary[gpu_name] = {"total_tests": 0, "fits": 0, "does_not_fit": 0}
        
        for result in self.results:
            summary[result.gpu_name]["total_tests"] += 1
            if result.fits:
                summary[result.gpu_name]["fits"] += 1
            else:
                summary[result.gpu_name]["does_not_fit"] += 1
        
        return summary
    
    def get_quantization_comparison(self, model_name: str, gpu_name: str) -> List[StorageResult]:
        """Compare all quantizations for a model on a GPU"""
        filtered = [
            r for r in self.results
            if r.model_name == model_name and r.gpu_name == gpu_name
        ]
        return sorted(filtered, key=lambda r: r.weight_storage_gb)
    
    def generate_report(self) -> str:
        """Generate analysis report"""
        report = []
        report.append("=" * 80)
        report.append("MODEL WEIGHT STORAGE ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        
        summary = self.get_fits_summary()
        report.append("\nGPU COMPATIBILITY SUMMARY:")
        report.append("-" * 80)
        for gpu_name, counts in summary.items():
            fits_pct = (counts["fits"] / counts["total_tests"] * 100) if counts["total_tests"] > 0 else 0
            report.append(
                f"{gpu_name:15} | Total: {counts['total_tests']:2} | "
                f"Fits: {counts['fits']:2} | Does Not Fit: {counts['does_not_fit']:2} | "
                f"Success Rate: {fits_pct:.1f}%"
            )
        
        report.append("\nBEST QUANTIZATION EFFICIENCY (Lowest Storage, Fits):")
        report.append("-" * 80)
        best = self.get_best_fit_per_model_gpu()
        for (model_name, gpu_name), result in sorted(best.items()):
            report.append(
                f"{model_name:20} on {gpu_name:15} | "
                f"Quantization: {result.quantization:6} | "
                f"Storage: {result.weight_storage_gb:7.3f}GB | "
                f"Total: {result.total_required_gb:7.3f}GB | "
                f"Utilization: {result.utilization_percent:6.1f}%"
            )
        
        report.append("\nDETAILED QUANTIZATION COMPARISON:")
        report.append("-" * 80)
        for model_name in ModelStorageCalculator.MODELS:
            for gpu_name in ModelStorageCalculator.GPUS:
                comparisons = self.get_quantization_comparison(model_name, gpu_name)
                if comparisons:
                    report.append(f"\n{model_name} on {gpu_name}:")
                    for result in comparisons:
                        fits_str = "FITS" if result.fits else "DOES NOT FIT"
                        report.append(
                            f"  {result.quantization:6} | Weight: {result.weight_storage_gb:6.3f}GB | "
                            f"Activation: {result.activation_overhead_gb:6.3f}GB | "
                            f"KV Cache: {result.kv_cache_overhead_gb:6.3f}GB | "
                            f"Total: {result.total_required_gb:6.3f}GB | "
                            f"Util: {result.utilization_percent:6.1f}% | {fits_str}"
                        )
        
        return "\n".join(report)


def main():
    """Run model storage tests"""
    calculator = ModelStorageCalculator()
    results = calculator.run_full_matrix()
    
    analyzer = ModelStorageAnalyzer(results)
    report = analyzer.generate_report()
    
    print(report)
    
    with open("symbiote_model_storage_report.json", "w") as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            indent=2
        )
    logger.info("Results saved to symbiote_model_storage_report.json")
    
    with open("symbiote_model_storage_analysis.txt", "w") as f:
        f.write(report)
    logger.info("Analysis saved to symbiote_model_storage_analysis.txt")


if __name__ == "__main__":
    main()
