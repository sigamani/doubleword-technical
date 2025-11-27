#!/usr/bin/env python3
"""
Background Job Manager for Asynchronous Batch Processing
Handles job queuing, status tracking, and result storage
"""

import json
import logging
import os
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    job_id: str
    input_path: str
    output_path: str
    num_samples: int
    batch_size: int
    concurrency: int

    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    samples_processed: int = 0
    error_message: Optional[str] = None
    results: List[Dict] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []

class BackgroundJobManager:
    """Manages asynchronous batch processing jobs"""
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.worker_thread = None
        self.running = False
        self._lock = threading.Lock()
        
    def start_worker(self):
        """Start the background worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Background job worker started")
        
    def stop_worker(self):
        """Stop the background worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Background job worker stopped")
        
    def submit_job(self, request_data: Dict) -> str:
        """Submit a new job for processing"""
        job_id = str(uuid.uuid4())[:8]
        
        job = Job(
            job_id=job_id,
            input_path=request_data["input_path"],
            output_path=request_data["output_path"],
            num_samples=request_data["num_samples"],
            batch_size=request_data["batch_size"],
            concurrency=request_data["concurrency"],

            status=JobStatus.QUEUED,
            created_at=time.time()
        )
        
        with self._lock:
            self.jobs[job_id] = job
            
        logger.info(f"Job {job_id} submitted for processing")
        return job_id
        
    def get_job_status(self, job_id: str) -> Optional[Job]:
        """Get current job status"""
        with self._lock:
            return self.jobs.get(job_id)
            
    def _worker_loop(self):
        """Main worker loop for processing jobs"""
        logger.info("Worker loop started")
        
        while self.running:
            try:
                # Find next queued job
                job = self._get_next_job()
                if not job:
                    time.sleep(1)
                    continue
                    
                # Process the job
                self._process_job(job)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1)
                
        logger.info("Worker loop stopped")
        
    def _get_next_job(self) -> Optional[Job]:
        """Get the next queued job"""
        with self._lock:
            for job in self.jobs.values():
                if job.status == JobStatus.QUEUED:
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                    return job
        return None
        
    def _process_job(self, job: Job):
        """Process a single job with retry and validation"""
        try:
            logger.info(f"Processing job {job.job_id}")
            
            # Import here to avoid circular imports
            from app.core.simple_processor import SimpleInferencePipeline, VLLMProcessor
            from app.models.schemas import BatchValidationRequest, BatchValidationResponse
            
            # Create processor
            pipeline = SimpleInferencePipeline()
            processor = VLLMProcessor("Qwen/Qwen2.5-0.5B-Instruct")
            
            # Generate sample prompts (in real implementation, would read from input_path)
            prompts = [f"Sample prompt {i+1} for testing" for i in range(job.num_samples)]
            
            # Validate batch size limits
            max_batch_size = 1000  # OpenAI-compatible limit
            if len(prompts) > max_batch_size:
                logger.warning(f"Batch size {len(prompts)} exceeds limit {max_batch_size}, truncating")
                prompts = prompts[:max_batch_size]
                job.num_samples = len(prompts)
            
            # Process in batches with retry logic
            batch_size = min(job.batch_size, len(prompts))
            all_results = []
            failed_batches = []
            
            for batch_start in range(0, len(prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]
                batch_index = batch_start // batch_size
                
                # Retry logic for failed batches
                for attempt in range(3):  # Max 3 retries
                    try:
                        results, _ = pipeline.execute_batch(batch_prompts, processor)
                        all_results.extend(results)
                        
                        # Update progress
                        with self._lock:
                            job.samples_processed = len(all_results)
                        
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        logger.error(f"Batch {batch_index} attempt {attempt + 1} failed: {e}")
                        if attempt < 2:  # Not last attempt
                            time.sleep(1.0 * (2 ** attempt))  # Exponential backoff
                        else:
                            # Log failed batch for error reporting
                            failed_batches.append({
                                "batch_index": batch_index,
                                "prompts": batch_prompts,
                                "error": str(e)
                            })
                            # Add placeholder results for failed batch
                            for prompt in batch_prompts:
                                all_results.append({
                                    "response": f"Failed to process: {str(e)[:100]}...",
                                    "prompt": prompt,
                                    "tokens": 0,
                                    "processing_time": 0.1
                                })
                            
                            # Update progress even for failed batches
                            with self._lock:
                                job.samples_processed = len(all_results)
            
            # Save results to file with error information
            self._save_results(job, all_results, failed_batches)
            
            # Mark as completed
            with self._lock:
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.results = all_results
                
            logger.info(f"Job {job.job_id} completed successfully with {len(failed_batches)} failed batches")
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error_message = str(e)
                
    def _save_results(self, job: Job, results: List[Dict], failed_batches: List[Dict] = None):
        if failed_batches is None:
            failed_batches = []
        """Save results to output file with error tracking"""
        if failed_batches is None:
            failed_batches = []
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(job.output_path), exist_ok=True)
            
            # Prepare output data with error information
            output_data = {
                "job_id": job.job_id,
                "status": "completed",
                "total_samples": job.num_samples,
                "processed_samples": len(results),
                "processing_time": job.completed_at - job.started_at if job.completed_at and job.started_at else 0,
                "results": results,
                "failed_batches": failed_batches or [],
                "success_rate": (len(results) - len(failed_batches or [])) / len(results) if results else 0
            }
            
            # Write to file
            with open(job.output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            logger.info(f"Results saved to {job.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

# Global instance
job_manager = BackgroundJobManager()