import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.scheduler import MockScheduler, GPUPoolState


class TestMockScheduler:
    """Test cases for MockScheduler"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.scheduler = MockScheduler()
    
    def test_initial_pool_state(self):
        """Test initial pool state"""
        status = self.scheduler.get_pool_status()
        
        assert status["spot"]["capacity"] == 2
        assert status["spot"]["available"] == 1
        assert status["dedicated"]["capacity"] == 1
        assert status["dedicated"]["available"] == 1
        assert status["active_jobs"] == 0
    
    def test_schedule_spot_job(self):
        """Test scheduling job to spot pool"""
        job_id = "test_job_1"
        job_data = {"prompts": ["test"]}
        
        pool_type = self.scheduler.schedule_job(job_id, job_data)
        
        assert pool_type == "spot"
        assert job_id in self.scheduler.job_assignments
        assert self.scheduler.job_assignments[job_id] == "spot"
        
        status = self.scheduler.get_pool_status()
        assert status["spot"]["available"] == 0
        assert status["active_jobs"] == 1
    
    def test_schedule_dedicated_job(self):
        """Test scheduling job to dedicated pool"""
        job_id = "test_job_1"
        job_data = {"prompts": ["test"]}
        
        # First fill up spot pool
        self.scheduler.schedule_job("fill_spot", {"prompts": ["test"]})
        
        # Then schedule another job
        pool_type = self.scheduler.schedule_job(job_id, job_data)
        
        assert pool_type == "dedicated"
        assert job_id in self.scheduler.job_assignments
        assert self.scheduler.job_assignments[job_id] == "dedicated"
        
        status = self.scheduler.get_pool_status()
        assert status["dedicated"]["available"] == 0
        assert status["active_jobs"] == 2
    
    def test_schedule_no_resources(self):
        """Test scheduling when no resources available"""
        # Fill up all pools
        self.scheduler.schedule_job("spot1", {"prompts": ["test"]})
        self.scheduler.schedule_job("spot2", {"prompts": ["test"]})
        self.scheduler.schedule_job("dedicated1", {"prompts": ["test"]})
        
        # Try to schedule another job
        pool_type = self.scheduler.schedule_job("no_resources", {"prompts": ["test"]})
        
        assert pool_type == "failed"
        assert "no_resources" not in self.scheduler.job_assignments
    
    def test_release_spot_job(self):
        """Test releasing spot job"""
        job_id = "test_job"
        self.scheduler.schedule_job(job_id, {"prompts": ["test"]})
        
        # Verify job is scheduled
        assert job_id in self.scheduler.job_assignments
        
        # Release job
        self.scheduler.release_job(job_id)
        
        # Verify job is released
        assert job_id not in self.scheduler.job_assignments
        
        status = self.scheduler.get_pool_status()
        assert status["spot"]["available"] == 1
        assert status["active_jobs"] == 0
    
    def test_release_dedicated_job(self):
        """Test releasing dedicated job"""
        job_id = "test_job"
        
        # Fill spot and schedule to dedicated
        self.scheduler.schedule_job("fill_spot", {"prompts": ["test"]})
        self.scheduler.schedule_job(job_id, {"prompts": ["test"]})
        
        # Verify job is scheduled to dedicated
        assert self.scheduler.job_assignments[job_id] == "dedicated"
        
        # Release job
        self.scheduler.release_job(job_id)
        
        # Verify job is released
        assert job_id not in self.scheduler.job_assignments
        
        status = self.scheduler.get_pool_status()
        assert status["dedicated"]["available"] == 1
        assert status["active_jobs"] == 1
    
    def test_release_nonexistent_job(self):
        """Test releasing a job that doesn't exist"""
        # Should not raise an error
        self.scheduler.release_job("nonexistent_job")
        
        status = self.scheduler.get_pool_status()
        assert status["active_jobs"] == 0


class TestGPUPoolState:
    """Test cases for GPUPoolState"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.pool = GPUPoolState()
    
    def test_initial_state(self):
        """Test initial pool state"""
        assert self.pool.spot_capacity == 2
        assert self.pool.spot_available == 1
        assert self.pool.dedicated_capacity == 1
        assert self.pool.dedicated_available == 1
    
    def test_allocate_spot(self):
        """Test allocating spot instance"""
        result = self.pool.allocate_spot()
        
        assert result is True
        assert self.pool.spot_available == 0
    
    def test_allocate_spot_unavailable(self):
        """Test allocating spot when unavailable"""
        self.pool.spot_available = 0
        result = self.pool.allocate_spot()
        
        assert result is False
        assert self.pool.spot_available == 0
    
    def test_allocate_dedicated(self):
        """Test allocating dedicated instance"""
        result = self.pool.allocate_dedicated()
        
        assert result is True
        assert self.pool.dedicated_available == 0
    
    def test_allocate_dedicated_unavailable(self):
        """Test allocating dedicated when unavailable"""
        self.pool.dedicated_available = 0
        result = self.pool.allocate_dedicated()
        
        assert result is False
        assert self.pool.dedicated_available == 0
    
    def test_release_spot(self):
        """Test releasing spot instance"""
        self.pool.spot_available = 0
        self.pool.release_spot()
        
        assert self.pool.spot_available == 1
    
    def test_release_spot_at_capacity(self):
        """Test releasing spot when already at capacity"""
        self.pool.release_spot()
        
        # Should not exceed capacity
        assert self.pool.spot_available == 1
    
    def test_release_dedicated(self):
        """Test releasing dedicated instance"""
        self.pool.dedicated_available = 0
        self.pool.release_dedicated()
        
        assert self.pool.dedicated_available == 1
    
    def test_release_dedicated_at_capacity(self):
        """Test releasing dedicated when already at capacity"""
        self.pool.release_dedicated()
        
        # Should not exceed capacity
        assert self.pool.dedicated_available == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])