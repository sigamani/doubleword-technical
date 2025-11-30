""" Testing GPU allocation and scheduling integration """

import pytest
import json
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from api.worker import BatchWorker
from api.job_queue import SimpleQueue
from api.gpu_scheduler import MockGPUScheduler, PoolType
from api.models import priorityLevels

class TestGPUIntegration:
    
    @pytest.fixture
    def scheduler(self):
        return MockGPUScheduler(spot_capacity=2, dedicated_capacity=1)
    
    @pytest.fixture
    def queue(self):
        return SimpleQueue()
    
    @pytest.fixture
    def worker(self, queue, scheduler):
        temp_dir = "/tmp/batch_test_dir"
        os.makedirs(temp_dir, exist_ok=True)
        
        worker = BatchWorker(queue, batch_dir=temp_dir, gpu_scheduler=scheduler)
        yield worker
        worker.stop()
        
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_gpu_allocation_success_flow(self, worker, queue, scheduler):
        job_data = {
            "job_id": "gpu_test_job",
            "input_file": f"{worker.batch_dir}/gpu_input.jsonl",
            "output_file": f"{worker.batch_dir}/gpu_output.jsonl", 
            "error_file": f"{worker.batch_dir}/gpu_error.jsonl"
        }
        
        with open(job_data["input_file"], 'w') as f:
            f.write(json.dumps({"prompt": "GPU integration test"}) + "\n")
        
        msg_id = queue.enqueue(job_data, priorityLevels.HIGH)
        assert msg_id is not None
        
        worker.start()
        time.sleep(1.0)  # Check while job should still be processing
        
        print(f"DEBUG: Scheduler allocations: {scheduler.allocations}")
        print(f"DEBUG: Pool status: {scheduler.get_pool_status()}")
        
        # Now check that GPU is allocated while job is processing
        allocation = scheduler.get_job_allocation("gpu_test_job")
        assert allocation is not None, f"GPU should be allocated while job is processing. Got: {allocation}"
        assert allocation == PoolType.DEDICATED, "High priority job should get dedicated GPU"
        
        # Wait for job to complete
        time.sleep(2.0)
        
        # Check that job completed successfully
        job_file = f"{worker.batch_dir}/job_gpu_test_job.json"
        assert os.path.exists(job_file), "Job file should exist"
        
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        assert job_data["status"] == "completed", f"Job should be completed, got status: {job_data['status']}"
        
        # Check that output file was created
        assert os.path.exists(f"{worker.batch_dir}/gpu_output.jsonl"), "Output file should exist"
    
    def test_gpu_exhaustion_handling(self, worker, queue, scheduler):
        for i in range(3):  
            job_data = {
                "job_id": f"filler_job_{i}",
                "input_file": f"{worker.batch_dir}/filler_input_{i}.jsonl",
                "output_file": f"{worker.batch_dir}/filler_output_{i}.jsonl",
                "error_file": f"{worker.batch_dir}/filler_error_{i}.jsonl"
            }
            
            with open(job_data["input_file"], 'w') as f:
                f.write(json.dumps({"prompt": f"Filler job {i}"}) + "\n")
            
            queue.enqueue(job_data, priorityLevels.LOW)
        
        test_job_data = {
            "job_id": "exhausted_job",
            "input_file": f"{worker.batch_dir}/exhausted_input.jsonl",
            "output_file": f"{worker.batch_dir}/exhausted_output.jsonl",
            "error_file": f"{worker.batch_dir}/exhausted_error.jsonl"
        }
        
        with open(test_job_data["input_file"], 'w') as f:
            f.write(json.dumps({"prompt": "GPU exhaustion test"}) + "\n")
        
        queue.enqueue(test_job_data, priorityLevels.LOW)
        
        worker.start()
        time.sleep(1.0)  # Check while jobs are processing
        
        # At this point, 1 spot GPU should be allocated (worker processes sequentially)
        allocations = scheduler.allocations
        spot_allocated = sum(1 for pool in allocations.values() if pool == PoolType.SPOT)
        assert spot_allocated == 1, f"Expected 1 spot GPU allocated, got {spot_allocated}"
        # The other jobs should still be in queue or waiting to be processed
        
        # Wait for jobs to complete
        time.sleep(8.0)  # Give enough time for all jobs to process
        
        # Check that some jobs completed successfully by looking for output files
        output_files = [f for f in os.listdir(worker.batch_dir) if f.endswith("_output.jsonl")]
        assert len(output_files) > 0, "At least some jobs should have completed and created output files"
        
        worker.stop()
    
    def test_gpu_priority_allocation(self, worker, queue, scheduler):
        for i in range(2):  
            job_data = {
                "job_id": f"spot_filler_{i}",
                "input_file": f"{worker.batch_dir}/spot_input_{i}.jsonl",
                "output_file": f"{worker.batch_dir}/spot_output_{i}.jsonl",
                "error_file": f"{worker.batch_dir}/spot_error_{i}.jsonl"
            }
            
            with open(job_data["input_file"], 'w') as f:
                f.write(json.dumps({"prompt": f"Spot filler {i}"}) + "\n")
            
            queue.enqueue(job_data, priorityLevels.LOW)
        
        high_job_data = {
            "job_id": "priority_job",
            "input_file": f"{worker.batch_dir}/priority_input.jsonl",
            "output_file": f"{worker.batch_dir}/priority_output.jsonl",
            "error_file": f"{worker.batch_dir}/priority_error.jsonl"
        }
        
        with open(high_job_data["input_file"], 'w') as f:
            f.write(json.dumps({"prompt": "High priority test"}) + "\n")
        
        queue.enqueue(high_job_data, priorityLevels.HIGH)
        
        worker.start()
        time.sleep(0.3)
        
        allocation = scheduler.get_job_allocation("priority_job")
        assert allocation == PoolType.DEDICATED, "High priority job should get dedicated GPU"
        
        worker.stop()
    
    def test_gpu_resource_metrics(self, scheduler):
        scheduler.allocate_gpu("metrics_job_1", priorityLevels.LOW)
        scheduler.allocate_gpu("metrics_job_2", priorityLevels.HIGH)
        scheduler.allocate_gpu("metrics_job_3", priorityLevels.LOW)
        
        metrics = scheduler.get_metrics()
        
        assert metrics["total_capacity"] == 3, "Total capacity should be 3"
        assert metrics["total_allocations"] == 3, "Should have 3 allocations"
        assert metrics["utilization_rate"] == 1.0, "Should be 100% utilized"
        assert metrics["queue_length"] == 0, "Queue should be empty"
        assert "pool_status" in metrics, "Should include pool status"
        
        pool_status = metrics["pool_status"]
        assert pool_status["spot"]["utilized"] == 2, "Spot pool should have 2 utilized"
        assert pool_status["dedicated"]["utilized"] == 1, "Dedicated pool should have 1 utilized"