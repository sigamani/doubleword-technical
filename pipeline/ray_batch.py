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
from pipeline.inference import create_mock_result

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
            import requests
            
            self.vllm_api_url = "http://vllm:8001/v1/completions"
            self.vllm_engine = True  
            
            logger.info(f"vLLM HTTP client initialized for API: {self.vllm_api_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM HTTP client: {e}")
            raise
    
    def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            if hasattr(self, 'vllm_api_url'):
                results = self._execute_vllm_batch(prompts)
            else:
                results = self._execute_batch_processing(prompts)
            self._log_completion(start_time)
            return results
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return self._fallback_process(prompts)
    
    def _execute_vllm_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Execute batch using real vLLM HTTP API"""
        logger.info(f"Processing {len(prompts)} prompts with vLLM HTTP API")
        
        results = []
        import requests
        import time
        
        for i, prompt in enumerate(prompts):
            start_time = time.time()
            try:
                response = requests.post(
                    self.vllm_api_url,
                    json={
                        "model": self.model_config.model_name,
                        "prompt": prompt,
                        "max_tokens": self.model_config.max_tokens,
                        "temperature": self.model_config.temperature
                    },
                    timeout=10,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data["choices"][0]["text"]
                    tokens = len(data["choices"][0].get("logprobs", {}).get("token_ids", []))
                else:
                    response_text = f"Error: HTTP {response.status_code}"
                    tokens = 0
                
                processing_time = time.time() - start_time
                
                results.append({
                    "prompt": prompt,
                    "response": response_text,
                    "tokens": tokens,
                    "processing_time": processing_time
                })
                
            except Exception as e:
                logger.error(f"Failed to process prompt {i}: {e}")
                results.append({
                    "prompt": prompt,
                    "response": f"Error: {str(e)}",
                    "tokens": 0,
                    "processing_time": 0.001
                })
        
        return results
    
    def _execute_batch_processing(self, prompts: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Processing {len(prompts)} prompts with mock inference")
        return self._fallback_process(prompts)
    
    def _fallback_process(self, prompts: List[str]) -> List[Dict[str, Any]]:
        import time
        """Simulate processing delay and return mock results"""
        time.sleep(0.5)  
        results = [create_mock_result(p, self.env_config.is_dev) for p in prompts]
        return [r.to_dict() for r in results]
    
    def _log_completion(self, start_time: float):
        duration = time.time() - start_time
        logger.info(f"Batch processing completed in {duration:.2f} seconds")