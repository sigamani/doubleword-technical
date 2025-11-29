import logging
from typing import Any, Dict, List
import time

logger = logging.getLogger(__name__)

try:
    import ray
    from ray.data import Dataset
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    
    class MockDataset:
        def __init__(self, items):
            self.items = items
        
        def count(self):
            return len(self.items)
        
        def take_all(self):
            return self.items
        
        def map_batches(self, func):
            return self
    
    class MockRay:
        class data:
            @staticmethod
            def from_items(items):
                return MockDataset(items)
    
    ray = MockRay()
    Dataset = MockDataset


def create_ray_dataset(prompts: List[str]):
    """Create Ray dataset from list of prompts."""
    if not RAY_AVAILABLE:
        logger.warning("Ray not available, using mock dataset")
    
    return ray.data.from_items([{"prompt": prompt} for prompt in prompts])


def preprocess_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess batch data - identity function for now."""
    return batch


def postprocess_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Postprocess batch results - extract generated text."""
    return {"results": batch.get("generated_texts", [])}


class RayBatchProcessor:
    """Ray Data batch processor for LLM inference"""
    
    def __init__(self, vllm_runner=None):
        self.vllm_runner = vllm_runner
        self.ray_available = RAY_AVAILABLE
    
    def process_batch(self, prompts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Process a batch of prompts using Ray Data"""
        start_time = time.time()
        
        try:
            # Create Ray dataset
            ds = create_ray_dataset(prompts)
            logger.info(f"Created dataset with {ds.count()} samples")
            
            if self.vllm_runner and self.vllm_runner.is_available():
                # Use vLLM for processing
                results = self._process_with_vllm(prompts)
            else:
                # Use Ray Data map_batches (mock implementation)
                results = self._process_with_ray(ds, batch_size)
            
            processing_time = time.time() - start_time
            logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return self._fallback_process(prompts)
    
    def _process_with_vllm(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Process prompts using vLLM runner"""
        if not self.vllm_runner:
            raise RuntimeError("vLLM runner not available")
        
        return self.vllm_runner.generate_batch(prompts)
    
    def _process_with_ray(self, ds, batch_size: int) -> List[Dict[str, Any]]:
        """Process dataset using Ray Data map_batches"""
        def process_ray_batch(batch):
            """Process a single Ray batch"""
            prompts = [item["prompt"] for item in batch]
            
            if self.vllm_runner and self.vllm_runner.is_available():
                return self._process_with_vllm(prompts)
            else:
                # Fallback processing
                results = []
                for prompt in prompts:
                    response = f"Processed: {str(prompt)[:50]}..."
                    results.append({
                        "prompt": prompt,
                        "response": response,
                        "tokens": len(response.split()),
                        "processing_time": 0.001
                    })
                return results
        
        # Apply map_batches
        processed_ds = ds.map_batches(
            process_ray_batch,
            batch_size=batch_size
        )
        
        # Collect results
        all_results = []
        batches = processed_ds.take_all()
        
        for batch in batches:
            if isinstance(batch, list):
                all_results.extend(batch)
            else:
                all_results.append(batch)
        
        return all_results
    
    def _fallback_process(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Fallback processing when both Ray and vLLM are unavailable"""
        results = []
        for prompt in prompts:
            response = f"Fallback processed: {str(prompt)[:50]}..."
            results.append({
                "prompt": prompt,
                "response": response,
                "tokens": len(response.split()),
                "processing_time": 0.001
            })
        
        return results


if __name__ == "__main__":
    # Test Ray batch processor
    processor = RayBatchProcessor()
    
    test_prompts = [
        "Hello world",
        "What is artificial intelligence?",
        "Explain machine learning",
        "Test prompt 4",
        "Test prompt 5"
    ]
    
    results = processor.process_batch(test_prompts)
    
    print(f"Processed {len(results)} prompts:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['prompt'][:30]}... -> {result['response'][:30]}...")