import logging
import time
import random
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class GPUType(Enum):
    DEDICATED = "dedicated"
    SPOT = "spot"
    CPU_FALLBACK = "cpu_fallback"

class GPUStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    PREEMPTED = "preempted"
    UNAVAILABLE = "unavailable"

@dataclass
class GPUResource:
    """GPU resource simulation"""
    gpu_id: str
    gpu_type: GPUType
    status: GPUStatus
    memory_gb: float
    compute_capability: float
    cost_per_hour: float
    current_job_id: Optional[str] = None
    preemption_probability: float = 0.0  # For spot instances
    last_preemption_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "gpu_id": self.gpu_id,
            "gpu_type": self.gpu_type.value,
            "status": self.status.value,
            "memory_gb": self.memory_gb,
            "compute_capability": self.compute_capability,
            "cost_per_hour": self.cost_per_hour,
            "current_job_id": self.current_job_id,
            "preemption_probability": self.preemption_probability,
            "last_preemption_time": self.last_preemption_time
        }

class GPUSimulator:
    """Simulates GPU resources with spot and dedicated instances"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.gpu_resources: Dict[str, GPUResource] = {}
        self.job_gpu_assignments: Dict[str, str] = {}  # job_id -> gpu_id
        self.preemption_callbacks: List[Callable] = []
        
        # Load configuration
        self.config = self._load_config(config_path)
        self._initialize_gpu_resources()
        
        logger.info(f"GPU Simulator initialized with {len(self.gpu_resources)} GPUs")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load GPU simulation configuration"""
        default_config = {
            "dedicated_gpus": [
                {
                    "gpu_id": "dedicated_0",
                    "memory_gb": 16.0,
                    "compute_capability": 8.0,
                    "cost_per_hour": 2.5
                }
            ],
            "spot_gpus": [
                {
                    "gpu_id": "spot_0",
                    "memory_gb": 16.0,
                    "compute_capability": 8.0,
                    "cost_per_hour": 0.8,
                    "preemption_probability": 0.1  # 10% chance per hour
                },
                {
                    "gpu_id": "spot_1",
                    "memory_gb": 16.0,
                    "compute_capability": 8.0,
                    "cost_per_hour": 0.8,
                    "preemption_probability": 0.1
                }
            ],
            "cpu_fallback": {
                "enabled": True,
                "memory_gb": 8.0,
                "compute_capability": 1.0,
                "cost_per_hour": 0.1
            },
            "preemption_check_interval": 300  # 5 minutes
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
                logger.info(f"Loaded GPU config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load GPU config, using defaults: {e}")
        
        return default_config
    
    def _initialize_gpu_resources(self):
        """Initialize GPU resources based on configuration"""
        # Add dedicated GPUs
        for gpu_config in self.config.get("dedicated_gpus", []):
            gpu = GPUResource(
                gpu_id=gpu_config["gpu_id"],
                gpu_type=GPUType.DEDICATED,
                status=GPUStatus.AVAILABLE,
                memory_gb=gpu_config["memory_gb"],
                compute_capability=gpu_config["compute_capability"],
                cost_per_hour=gpu_config["cost_per_hour"],
                preemption_probability=0.0  # Dedicated GPUs don't get preempted
            )
            self.gpu_resources[gpu.gpu_id] = gpu
        
        # Add spot GPUs
        for gpu_config in self.config.get("spot_gpus", []):
            gpu = GPUResource(
                gpu_id=gpu_config["gpu_id"],
                gpu_type=GPUType.SPOT,
                status=GPUStatus.AVAILABLE,
                memory_gb=gpu_config["memory_gb"],
                compute_capability=gpu_config["compute_capability"],
                cost_per_hour=gpu_config["cost_per_hour"],
                preemption_probability=gpu_config.get("preemption_probability", 0.1)
            )
            self.gpu_resources[gpu.gpu_id] = gpu
        
        # Add CPU fallback
        if self.config.get("cpu_fallback", {}).get("enabled", True):
            cpu_config = self.config["cpu_fallback"]
            gpu = GPUResource(
                gpu_id="cpu_fallback",
                gpu_type=GPUType.CPU_FALLBACK,
                status=GPUStatus.AVAILABLE,
                memory_gb=cpu_config["memory_gb"],
                compute_capability=cpu_config["compute_capability"],
                cost_per_hour=cpu_config["cost_per_hour"],
                preemption_probability=0.0
            )
            self.gpu_resources[gpu.gpu_id] = gpu
    
    def request_gpu(self, job_id: str, preferred_type: Optional[GPUType] = None) -> Optional[str]:
        """Request a GPU for a job"""
        # Determine preferred GPU type
        if preferred_type is None:
            # Default preference: dedicated > spot > cpu
            preferred_order = [GPUType.DEDICATED, GPUType.SPOT, GPUType.CPU_FALLBACK]
        else:
            preferred_order = [preferred_type, GPUType.SPOT, GPUType.DEDICATED, GPUType.CPU_FALLBACK]
        
        # Try to find available GPU in preferred order
        for gpu_type in preferred_order:
            available_gpu = self._find_available_gpu(gpu_type)
            if available_gpu:
                # Assign GPU to job
                self._assign_gpu_to_job(available_gpu, job_id)
                logger.info(f"Assigned {available_gpu} ({gpu_type.value}) to job {job_id}")
                return available_gpu
        
        logger.warning(f"No GPU available for job {job_id}")
        return None
    
    def _find_available_gpu(self, gpu_type: GPUType) -> Optional[str]:
        """Find available GPU of specific type"""
        for gpu_id, gpu in self.gpu_resources.items():
            if gpu.gpu_type == gpu_type and gpu.status == GPUStatus.AVAILABLE:
                return gpu_id
        return None
    
    def _assign_gpu_to_job(self, gpu_id: str, job_id: str):
        """Assign GPU to job"""
        gpu = self.gpu_resources[gpu_id]
        gpu.status = GPUStatus.BUSY
        gpu.current_job_id = job_id
        self.job_gpu_assignments[job_id] = gpu_id
    
    def release_gpu(self, job_id: str):
        """Release GPU from job"""
        if job_id not in self.job_gpu_assignments:
            logger.warning(f"Job {job_id} has no GPU assigned")
            return
        
        gpu_id = self.job_gpu_assignments[job_id]
        gpu = self.gpu_resources[gpu_id]
        
        gpu.status = GPUStatus.AVAILABLE
        gpu.current_job_id = None
        
        del self.job_gpu_assignments[job_id]
        logger.info(f"Released {gpu_id} from job {job_id}")
    
    def check_preemptions(self):
        """Check for spot GPU preemptions"""
        current_time = time.time()
        preemptions = []
        
        for gpu_id, gpu in self.gpu_resources.items():
            if (gpu.gpu_type == GPUType.SPOT and 
                gpu.status == GPUStatus.BUSY and 
                gpu.current_job_id):
                
                # Simulate preemption probability
                # Check every preemption_check_interval seconds
                if (gpu.last_preemption_time is None or 
                    current_time - gpu.last_preemption_time > self.config["preemption_check_interval"]):
                    
                    if random.random() < gpu.preemption_probability:
                        # Preemption occurs
                        job_id = gpu.current_job_id
                        preemptions.append((gpu_id, job_id))
                        
                        # Update GPU status
                        gpu.status = GPUStatus.PREEMPTED
                        gpu.last_preemption_time = current_time
                        gpu.current_job_id = None
                        
                        # Remove from job assignments
                        if job_id in self.job_gpu_assignments:
                            del self.job_gpu_assignments[job_id]
                        
                        logger.warning(f"SPOT GPU {gpu_id} preempted from job {job_id}")
        
        # Trigger preemption callbacks
        for gpu_id, job_id in preemptions:
            self._trigger_preemption_callbacks(gpu_id, job_id)
        
        return preemptions
    
    def _trigger_preemption_callbacks(self, gpu_id: str, job_id: str):
        """Trigger preemption callbacks"""
        for callback in self.preemption_callbacks:
            try:
                callback(gpu_id, job_id)
            except Exception as e:
                logger.error(f"Preemption callback failed: {e}")
    
    def add_preemption_callback(self, callback: Callable):
        """Add callback for preemption events"""
        self.preemption_callbacks.append(callback)
    
    def get_gpu_status(self, gpu_id: str) -> Optional[Dict]:
        """Get status of specific GPU"""
        if gpu_id in self.gpu_resources:
            return self.gpu_resources[gpu_id].to_dict()
        return None
    
    def get_all_gpu_status(self) -> List[Dict]:
        """Get status of all GPUs"""
        return [gpu.to_dict() for gpu in self.gpu_resources.values()]
    
    def get_job_gpu_assignment(self, job_id: str) -> Optional[str]:
        """Get GPU assigned to job"""
        return self.job_gpu_assignments.get(job_id)
    
    def get_gpu_utilization(self) -> Dict:
        """Get GPU utilization statistics"""
        total_gpus = len(self.gpu_resources)
        busy_gpus = sum(1 for gpu in self.gpu_resources.values() if gpu.status == GPUStatus.BUSY)
        preempted_gpus = sum(1 for gpu in self.gpu_resources.values() if gpu.status == GPUStatus.PREEMPTED)
        
        utilization_by_type = {}
        for gpu_type in GPUType:
            type_gpus = [gpu for gpu in self.gpu_resources.values() if gpu.gpu_type == gpu_type]
            type_busy = sum(1 for gpu in type_gpus if gpu.status == GPUStatus.BUSY)
            utilization_by_type[gpu_type.value] = {
                "total": len(type_gpus),
                "busy": type_busy,
                "available": len(type_gpus) - type_busy,
                "utilization_percentage": (type_busy / len(type_gpus) * 100) if type_gpus else 0
            }
        
        return {
            "total_gpus": total_gpus,
            "busy_gpus": busy_gpus,
            "available_gpus": total_gpus - busy_gpus - preempted_gpus,
            "preempted_gpus": preempted_gpus,
            "overall_utilization_percentage": (busy_gpus / total_gpus * 100) if total_gpus > 0 else 0,
            "utilization_by_type": utilization_by_type
        }
    
    def get_cost_estimate(self, job_duration_hours: float, gpu_type: GPUType) -> float:
        """Get cost estimate for job duration"""
        # Find a GPU of the specified type
        for gpu in self.gpu_resources.values():
            if gpu.gpu_type == gpu_type:
                return gpu.cost_per_hour * job_duration_hours
        
        # Fallback to default costs
        default_costs = {
            GPUType.DEDICATED: 2.5,
            GPUType.SPOT: 0.8,
            GPUType.CPU_FALLBACK: 0.1
        }
        return default_costs.get(gpu_type, 1.0) * job_duration_hours
    
    def save_state(self, filepath: str):
        """Save GPU simulator state to file"""
        try:
            state = {
                "timestamp": time.time(),
                "config": self.config,
                "gpu_resources": {gpu_id: gpu.to_dict() for gpu_id, gpu in self.gpu_resources.items()},
                "job_gpu_assignments": self.job_gpu_assignments
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"GPU simulator state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save GPU state: {e}")
    
    def load_state(self, filepath: str):
        """Load GPU simulator state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore GPU resources
            for gpu_id, gpu_data in state.get("gpu_resources", {}).items():
                if gpu_id in self.gpu_resources:
                    gpu = self.gpu_resources[gpu_id]
                    gpu.status = GPUStatus(gpu_data["status"])
                    gpu.current_job_id = gpu_data.get("current_job_id")
                    gpu.last_preemption_time = gpu_data.get("last_preemption_time")
            
            # Restore job assignments
            self.job_gpu_assignments = state.get("job_gpu_assignments", {})
            
            logger.info(f"GPU simulator state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load GPU state: {e}")

# Global GPU simulator instance
gpu_simulator = GPUSimulator()