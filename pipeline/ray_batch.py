""" Ray Batch Processor for handling batch inference with vLLM or mock processing. """

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
import logging
import time
from typing import List, Dict, Any
from config import EnvironmentConfig, ModelConfig
logger = logging.getLogger(__name__)

from config import EnvironmentConfig, ModelConfig
from pipeline.inference import create_dataset, create_mock_result

class RayBatchProcessor:
    def __init__(self, model_config: ModelConfig, env_config: EnvironmentConfig):
        self.model_config = model_config
        self.env_config = env_config
        self.vllm_engine = None
        
        if env_config.is_gpu_available and not env_config.is_dev:
            self._init_vllm_engine()
            logger.info("STAGE: Using real vLLM engine")
        else:
            self.processor = None
            logger.info("DEV: Using mock Ray Data processor")
    
    def _init_vllm_engine(self):
        try:
            from vllm import LLM, SamplingParams
            
            self.vllm_engine = LLM(
                model=self.model_config.model_name,
                max_model_len=self.model_config.max_model_len,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
                dtype="float16",
                enable_chunked_prefill=True,
                max_num_batched_tokens=2048,
            )
            
            self.sampling_params = SamplingParams(
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
            )
            
            logger.info(f"vLLM engine initialized with model: {self.model_config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            if self.vllm_engine:
                results = self._execute_vllm_batch(prompts)
            else:
                results = self._execute_batch_processing(prompts)
            self._log_completion(start_time)
            return results
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return self._fallback_process(prompts)
    
    def _execute_vllm_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Execute batch using real vLLM engine"""
        logger.info(f"Processing {len(prompts)} prompts with vLLM")
        
        outputs = self.vllm_engine.generate(prompts, self.sampling_params)
        
        results = []
        for i, output in enumerate(outputs):
            results.append({
                "prompt": prompts[i],
                "response": output.outputs[0].text,
                "tokens": len(output.outputs[0].token_ids),
                "processing_time": output.finish_time - output.start_time if hasattr(output, 'finish_time') else 0.001
            })
        
        return results
    
    def _execute_batch_processing(self, prompts: List[str]) -> List[Dict[str, Any]]:
        ds = create_dataset(prompts)
        logger.info(f"Created dataset with {ds.count()} samples")
        return self._process_with_mock(ds)
    
    def _process_with_mock(self, ds) -> List[Dict[str, Any]]:
        logger.info("Using Ray Data map_batches with mock inference")
        def process_batch_with_mock(batch, is_dev):
            results = []
            if isinstance(batch, dict) and 'prompt' in batch:
                prompts = batch['prompt']
                if hasattr(prompts, '__iter__') and not isinstance(prompts, str):
                    for prompt in prompts:
                        result = create_mock_result(str(prompt), is_dev)
                        results.append(result.to_dict())
                else:
                    result = create_mock_result(str(prompts), is_dev)
                    results.append(result.to_dict())
            else:
                for item in batch:
                    if hasattr(item, 'get'):
                        prompt = item.get('prompt', str(item))
                    else:
                        prompt = str(item)
                    result = create_mock_result(prompt, is_dev)
                    results.append(result.to_dict())
            return {"results": results}
        
        batch_fn = lambda batch: process_batch_with_mock(batch, self.env_config.is_dev)
        processed_ds = ds.map_batches(batch_fn, batch_size=self.model_config.batch_size)
        batches = processed_ds.take_all()
        
        all_results = []
        for batch in batches:
            if isinstance(batch, dict) and 'results' in batch:
                all_results.extend(batch['results'])
            elif isinstance(batch, list):
                all_results.extend(batch)
            else:
                all_results.append(batch)
        return all_results
    
    def _fallback_process(self, prompts: List[str]) -> List[Dict[str, Any]]:
        results = [create_mock_result(p, self.env_config.is_dev) for p in prompts]
        return [r.to_dict() for r in results]
    
    def _log_completion(self, start_time: float):
        duration = time.time() - start_time
        logger.info(f"Batch processing completed in {duration:.2f} seconds")