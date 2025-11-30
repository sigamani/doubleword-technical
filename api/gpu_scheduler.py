"""
Mock GPU Scheduler for testing GPU allocation logic without real GPU resources.
Includes resource constraints, cost tracking, and stress testing capabilities.
"""

import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List
logger = logging.getLogger(__name__)

class PoolType(Enum):
    SPOT = "spot"
    DEDICATED = "dedicated"

@dataclass
class GPUResource:
    pool_type: PoolType
    capacity: int
    available: int
    cost_per_hour: float

@dataclass
class AllocationResult:
    pool_type: PoolType
    allocated: bool
    reason: str
    cost_estimate: float = 0.0
    wait_time: float = 0.0
    queue_position: int = 0

class MockGPUScheduler:
    def __init__(self, spot_capacity: int = 2, dedicated_capacity: int = 1):
        self.pools: Dict[PoolType, GPUResource] = {
            PoolType.SPOT: GPUResource(
                pool_type=PoolType.SPOT,
                capacity=spot_capacity,
                available=spot_capacity,  
                cost_per_hour=0.10
            ),
            PoolType.DEDICATED: GPUResource(
                pool_type=PoolType.DEDICATED,
                capacity=dedicated_capacity,
                available=dedicated_capacity,  
                cost_per_hour=0.50
            )
        }
        self.allocations: Dict[str, PoolType] = {}
        self.allocation_times: Dict[str, float] = {}
        self.waiting_queue: List[str] = []
        self.total_cost = 0.0
        self.rejected_jobs = 0
        self.total_allocations = 0 
        
    def allocate_gpu(self, job_id: str, priority_level: int = 1) -> AllocationResult:
        start_time = time.time()
        
        is_high_priority = priority_level >= 8
        
        if self._no_resources_available():
            if is_high_priority:
                self.waiting_queue.insert(0, job_id)
            else:
                self.waiting_queue.append(job_id)
            
            queue_pos = self.waiting_queue.index(job_id)
            return AllocationResult(
                pool_type=PoolType.SPOT,
                allocated=False,
                reason=f"No resources available, queued at position {queue_pos} (priority: {priority_level})",
                wait_time=time.time() - start_time,
                queue_position=queue_pos
            )
        
        if is_high_priority:
            allocation = self._try_allocate_dedicated(job_id)
            if not allocation.allocated:
                allocation = self._try_allocate_spot(job_id)
        else:
            allocation = self._try_allocate_spot(job_id)
            if not allocation.allocated:
                allocation = self._try_allocate_dedicated(job_id)
        
        if allocation.allocated:
            self.allocation_times[job_id] = time.time()
            self.total_allocations += 1
            logger.info(f"Allocated {allocation.pool_type.value} GPU for job {job_id} (priority: {priority_level})")
            
        return allocation
    
    def release_gpu(self, job_id: str) -> None:
        if job_id not in self.allocations:
            logger.warning(f"Job {job_id} not found in allocations")
            return
            
        pool_type = self.allocations.pop(job_id)
        pool = self.pools[pool_type]
        pool.available += 1
        logger.info(f"Released {pool_type.value} GPU for job {job_id}")
    
    def get_pool_status(self) -> Dict:
        return {
            pool_type.value: {
                "capacity": pool.capacity,
                "available": pool.available,
                "utilized": pool.capacity - pool.available,
                "cost_per_hour": pool.cost_per_hour
            }
            for pool_type, pool in self.pools.items()
        }
    
    def get_job_allocation(self, job_id: str) -> Optional[PoolType]:
        return self.allocations.get(job_id)
    
    def process_waiting_queue(self) -> Optional[str]:
        if self.waiting_queue and not self._no_resources_available():
            next_job_id = self.waiting_queue.pop(0)
            return next_job_id
        return None
    
    def _no_resources_available(self) -> bool:
        return all(pool.available <= 0 for pool in self.pools.values())

    def _try_allocate_spot(self, job_id: str) -> AllocationResult:
        spot_pool = self.pools[PoolType.SPOT]
        if spot_pool.available > 0:
            spot_pool.available -= 1
            self.allocations[job_id] = PoolType.SPOT
            return AllocationResult(
                pool_type=PoolType.SPOT,
                allocated=True,
                reason="Spot instance available (cost-effective)",
                cost_estimate=spot_pool.cost_per_hour
            )
        return AllocationResult(PoolType.SPOT, False, "No spot available")
    
    def _try_allocate_dedicated(self, job_id: str) -> AllocationResult:
        dedicated_pool = self.pools[PoolType.DEDICATED]
        if dedicated_pool.available > 0:
            dedicated_pool.available -= 1
            self.allocations[job_id] = PoolType.DEDICATED
            return AllocationResult(
                pool_type=PoolType.DEDICATED,
                allocated=True,
                reason="Spot unavailable, using dedicated instance",
                cost_estimate=dedicated_pool.cost_per_hour
            )
        return AllocationResult(PoolType.DEDICATED, False, "No dedicated available")
    
    def get_metrics(self) -> Dict:
        total_capacity = sum(pool.capacity for pool in self.pools.values())
        total_available = sum(pool.available for pool in self.pools.values())
        utilization_rate = (total_capacity - total_available) / total_capacity if total_capacity > 0 else 0
        
        return {
            "total_capacity": total_capacity,
            "total_available": total_available,
            "utilization_rate": utilization_rate,
            "total_allocations": self.total_allocations,
            "rejected_jobs": self.rejected_jobs,
            "total_cost": self.total_cost,
            "queue_length": len(self.waiting_queue),
            "pool_status": self.get_pool_status()
        }