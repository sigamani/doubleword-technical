""" Batch worker that processes jobs from the queue. """

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import BatchConfig, EnvironmentConfig, ModelConfig

from api.job_queue import SimpleQueue
from config import ModelConfig, EnvironmentConfig
from pipeline.ray_batch import RayBatchProcessor
from api.gpu_scheduler import MockGPUScheduler
from api.models import priorityLevels

import logging
import time

import threading
import os
import json
from typing import Dict, Any, List


logger = logging.getLogger(__name__)

class BatchWorker:
    def __init__(self, queue: SimpleQueue, batch_dir: str | None = None, gpu_scheduler: MockGPUScheduler | None = None):
        self.queue = queue
        config = BatchConfig()
        self.batch_dir = config.batch_dir if batch_dir is None else batch_dir
        self.pipeline = RayBatchProcessor(
            ModelConfig.default(), 
            EnvironmentConfig.from_env()
        )
        self.running = False
        self.worker_thread = None
        self.gpu_scheduler = gpu_scheduler if gpu_scheduler is not None else MockGPUScheduler()

    def start(self):
        if self.running:
            logger.warning("Worker is already running")
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Batch worker started")
        
    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Batch worker stopped")
        
    def _worker_loop(self) -> None:
        logger.info("Worker loop started")
        while self.running:
            try:
                messages = self.queue.dequeue(count=1)
                
                if not messages:
                    time.sleep(0.1)
                    continue
                    
                job_msg = messages[0]
                logger.info(f"Got job: {job_msg.payload.get('job_id')}")
                self._process_job(job_msg.payload, job_msg.priority)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1.0)  

    def _process_job(self, job_data: Dict[str, Any], priority: priorityLevels = priorityLevels.LOW) -> None:
        job_id = job_data.get("job_id")
        input_file = job_data.get("input_file")
        output_file = job_data.get("output_file")
        
        if not job_id or not input_file or not output_file:
            raise ValueError(f"Missing required job data: job_id={job_id}, input_file={input_file}, output_file={output_file}")
        
        try:
            allocation_result = self.gpu_scheduler.allocate_gpu(job_id, priority_level=priority)
            if not allocation_result.allocated:
                logger.error(f"Failed to allocate GPU for job {job_id}: {allocation_result.reason}")
                self._update_job_status(job_id, "failed", completed_at=time.time())
                return
            
            logger.info(f"Processing job {job_id}")
            self._update_job_status(job_id, "running")            
            logger.info(f"Loading prompts from {input_file}")
            prompts = self._load_prompts(input_file)
            logger.info(f"Loaded {len(prompts)} prompts")
            
            logger.info("Starting batch inference...")
            
            if hasattr(self.pipeline.env_config, 'is_dev') and self.pipeline.env_config.is_dev:
                logger.info("DEV mode detected, using fallback processing")
                results = self.pipeline._fallback_process(prompts)
            else:
                try:
                    import threading
                    
                    timeout_occurred = threading.Event()
                    
                    def timeout_handler():
                        timeout_occurred.set()
                    
                    timer = threading.Timer(5.0, timeout_handler)  
                    timer.start()
                    
                    results = self.pipeline.process_batch(prompts)
                    timer.cancel()
                    
                    if timeout_occurred.is_set():
                        logger.warning("Batch inference timed out, using fallback")
                        results = self.pipeline._fallback_process(prompts)
                    else:
                        logger.info(f"Inference completed, got {len(results)} results")
                        
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}")
                    logger.info("Using fallback processing")
                    results = self.pipeline._fallback_process(prompts)
                    logger.info(f"Fallback inference completed, got {len(results)} results")
            
            logger.info(f"Saving results to {output_file}")
            self._save_results(output_file, results)
            
            self._update_job_status(job_id, "completed", completed_at=time.time())
            self.gpu_scheduler.release_gpu(job_id)
            next_job_id = self.gpu_scheduler.process_waiting_queue()
            if next_job_id:
                logger.info(f"GPU freed from completed job {job_id}, allocated to waiting job {next_job_id}")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            error_file = job_data.get("error_file", "").replace("/tmp/", "/tmp/")
            with open(error_file, "w") as f:
                json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
                f.write("\n")
            
            self._update_job_status(job_id, "failed", completed_at=time.time())
            self.gpu_scheduler.release_gpu(job_id)
            next_job_id = self.gpu_scheduler.process_waiting_queue()
            if next_job_id:
                logger.info(f"GPU freed from failed job {job_id}, allocated to waiting job {next_job_id}")
                
    def _load_prompts(self, input_file: str) -> List[str]:
        prompts = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    prompts.append(data.get("prompt", ""))
        return prompts
        
    def _save_results(self, output_file: str, results: List[Dict[str, Any]]) -> None:
        with open(output_file, 'w') as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")
                
    def _update_job_status(self, job_id: str, status: str, completed_at: float | None = None):
        job_path = os.path.join(self.batch_dir, f"job_{job_id}.json")
        try:
            os.makedirs(self.batch_dir, exist_ok=True)
            
            job_data = {}
            if os.path.exists(job_path):
                with open(job_path, 'r') as f:
                    job_data = json.load(f)
            
            job_data["status"] = status
            if completed_at is not None:
                job_data["completed_at"] = completed_at
                    
            with open(job_path, 'w') as f:
                json.dump(job_data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to update job status for {job_id}: {e}")
