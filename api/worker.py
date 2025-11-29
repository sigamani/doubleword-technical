import logging
import time
import json
import threading
import os
from typing import Dict, Any, List
from api.queue import SimpleQueue
from engine.vllm_runner import InferencePipeline
logger = logging.getLogger(__name__)
class BatchWorker:
    """Background worker that processes batch jobs from queue"""
    
    def __init__(self, queue: SimpleQueue, batch_dir: str = "/tmp"):
        self.queue = queue
        self.batch_dir = batch_dir
        self.pipeline = InferencePipeline()
        self.running = False
        self.worker_thread = None
        
    def start(self):
        """Start background worker thread"""
        if self.running:
            logger.warning("Worker is already running")
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Batch worker started")
        
    def stop(self):
        """Stop background worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Batch worker stopped")
        
    def _worker_loop(self):
        """Main worker loop - runs in background thread"""
        print("🔄 Worker loop started")
        while self.running:
            try:
                # Get next job from queue (blocking with timeout)
                messages = self.queue.dequeue(count=1)
                
                if not messages:
                    # No jobs in queue, wait a bit
                    time.sleep(0.1)
                    continue
                    
                # Process job
                job_msg = messages[0]
                print(f"📋 Got job: {job_msg.payload.get('job_id')}")
                self._process_job(job_msg.payload)
                
            except Exception as e:
                print(f"❌ Worker loop error: {e}")
                time.sleep(1.0)  # Prevent tight error loop
                
    def _process_job(self, job_data: Dict[str, Any]):
        """Process a single batch job"""
        job_id = job_data.get("job_id")
        input_file = job_data.get("input_file")
        output_file = job_data.get("output_file")
        
        if not job_id or not input_file or not output_file:
            raise ValueError(f"Missing required job data: job_id={job_id}, input_file={input_file}, output_file={output_file}")
        
        try:
            print(f"🔄 Processing job {job_id}")
            
            # Update job status to running
            self._update_job_status(job_id, "running")
            
            # Load prompts from input file
            print(f"📖 Loading prompts from {input_file}")
            prompts = self._load_prompts(input_file)
            print(f"📝 Loaded {len(prompts)} prompts: {prompts}")
            
            # Execute batch inference with timeout
            print("🤖 Starting batch inference...")
            try:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Batch inference timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
                
                results = self.pipeline.execute_batch(prompts)
                signal.alarm(0)  # Cancel timeout
                print(f"✅ Inference completed, got {len(results)} results")
                
            except TimeoutError:
                print("⏰ Batch inference timed out, using fallback")
                results = self.pipeline._execute_fallback(prompts)
                print(f"✅ Fallback inference completed, got {len(results)} results")
            except Exception as e:
                print(f"❌ Batch inference failed: {e}")
                print("🔄 Using fallback processing")
                results = self.pipeline._execute_fallback(prompts)
                print(f"✅ Fallback inference completed, got {len(results)} results")
            
            # Save results to output file
            print(f"💾 Saving results to {output_file}")
            self._save_results(output_file, results)
            
            # Update job status to completed
            self._update_job_status(job_id, "completed", completed_at=time.time())
            
            print(f"🎉 Job {job_id} completed successfully with {len(results)} results")
            
        except Exception as e:
            print(f"❌ Job {job_id} failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error to error file
            error_file = job_data.get("error_file", "").replace("/tmp/", "/tmp/")
            with open(error_file, "w") as f:
                json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
                f.write("\n")
            
            self._update_job_status(job_id, "failed", completed_at=time.time())
        
    def _load_prompts(self, input_file: str) -> List[str]:
        """Load prompts from JSONL input file"""
        prompts = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    prompts.append(data.get("prompt", ""))
        return prompts
        
    def _save_results(self, output_file: str, results: List[Dict[str, Any]]):
        """Save results to JSONL output file"""
        with open(output_file, 'w') as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")
                
    def _update_job_status(self, job_id: str, status: str, completed_at: float | None = None):
        """Update job status in metadata file"""
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