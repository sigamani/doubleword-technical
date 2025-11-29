import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from api.job_queue import SimpleQueue
from api.models import priorityLevels
from pipeline.config import ModelConfig, EnvironmentConfig
from pipeline.ray_batch import RayBatchProcessor

import logging
import time
import json
import threading
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class BatchWorker:
    
    def __init__(self, queue: SimpleQueue, batch_dir: str = "/tmp"):
        self.queue = queue
        self.batch_dir = batch_dir
        self.pipeline = RayBatchProcessor(
            ModelConfig.default(), 
            EnvironmentConfig.from_env()
        )
        self.running = False
        self.worker_thread = None
        
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
        
    def _worker_loop(self):
        logger.info("Worker loop started")
        while self.running:
            try:
                messages = self.queue.dequeue(count=1)
                
                if not messages:
                    time.sleep(0.1)
                    continue
                    
                job_msg = messages[0]
                logger.info(f"Got job: {job_msg.payload.get('job_id')}")
                self._process_job(job_msg.payload)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1.0)  

    def _process_job(self, job_data: Dict[str, Any]):
        job_id = job_data.get("job_id")
        input_file = job_data.get("input_file")
        output_file = job_data.get("output_file")
        
        if not job_id or not input_file or not output_file:
            raise ValueError(f"Missing required job data: job_id={job_id}, input_file={input_file}, output_file={output_file}")
        
        try:
            logger.info(f"Processing job {job_id}")
            self._update_job_status(job_id, "running")            
            logger.info(f"Loading prompts from {input_file}")
            prompts = self._load_prompts(input_file)
            logger.info(f"Loaded {len(prompts)} prompts")
            
            logger.info("Starting batch inference...")

            try:
                import threading
                
                timeout_occurred = threading.Event()
                
                def timeout_handler():
                    timeout_occurred.set()
                
                timer = threading.Timer(30.0, timeout_handler)  
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

            except TimeoutError:
                logger.warning("Batch inference timed out, using fallback")
                results = self.pipeline._fallback_process(prompts)
                logger.info(f"Fallback inference completed, got {len(results)} results")
     
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                logger.info("Using fallback processing")
                results = self.pipeline._fallback_process(prompts)
                logger.info(f"Fallback inference completed, got {len(results)} results")
            
            logger.info(f"Saving results to {output_file}")
            self._save_results(output_file, results)
            
            self._update_job_status(job_id, "completed", completed_at=time.time())
            
            
        except Exception as e:
            logger.error(f"❌ Job {job_id} failed: {e}")
            import traceback
            traceback.print_exc()
            
            error_file = job_data.get("error_file", "").replace("/tmp/", "/tmp/")
            with open(error_file, "w") as f:
                json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
                f.write("\n")
            
            self._update_job_status(job_id, "failed", completed_at=time.time())
        
    def _load_prompts(self, input_file: str) -> List[str]:
        prompts = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    prompts.append(data.get("prompt", ""))
        return prompts
        
    def _save_results(self, output_file: str, results: List[Dict[str, Any]]):
        with open(output_file, 'w') as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")
                
    def _update_job_status(self, job_id: str, status: str, completed_at: float | None = None):
        job_path = os.path.join(self.batch_dir, f"job_{job_id}.json")
        try:
            with open(job_path, 'r') as f:
                job_data = json.load(f)
            
            job_data["status"] = status
            if completed_at is not None:
                job_data["completed_at"] = completed_at
                
            with open(job_path, 'w') as f:
                json.dump(job_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update job status for {job_id}: {e}")


if __name__ == "__main__":
    
    from api.job_queue import SimpleQueue
    import json
    import tempfile
    import os
    
    queue = SimpleQueue()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        job_id = "test-job-123"
        input_file = os.path.join(temp_dir, f"{job_id}_input.jsonl")
        output_file = os.path.join(temp_dir, f"{job_id}_output.jsonl")
        job_file = os.path.join(temp_dir, f"job_{job_id}.json")
        
        test_prompts = ["Hello world", "What is AI?", "Test prompt 3"]
        with open(input_file, 'w') as f:
            for prompt in test_prompts:
                json.dump({"prompt": prompt}, f)
                f.write("\n")
        
        job_data = {
            "id": job_id,
            "model": "test-model",
            "status": "queued",
            "created_at": 1234567890,
            "num_prompts": len(test_prompts),
            "input_file": input_file,
            "output_file": output_file,
            "error_file": os.path.join(temp_dir, f"{job_id}_errors.jsonl")
        }
        
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        worker = BatchWorker(queue, batch_dir=temp_dir)
        
        job_payload = {
            "job_id": job_id,
            "input_file": input_file,
            "output_file": output_file,
            "model": "test-model",
            "max_tokens": 64,
            "temperature": 0.7,
            "error_file": job_data["error_file"]
        }
        
        msg_id = queue.enqueue(job_payload, priority=priorityLevels.LOW)
        worker._process_job(job_payload)
        
        if os.path.exists(output_file):
            print("Output file created")
            with open(output_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
                print(f"   Generated {len(results)} results")
                for i, result in enumerate(results[:2]):  # Show first 2
                    print(f"   {i+1}. {result}...")
        
        with open(job_file, 'r') as f:
            updated_job = json.load(f)
            print(f"Job status: {updated_job['status']}")
