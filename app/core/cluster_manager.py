"""
Ray cluster management utilities
"""

import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

def initialize_ray_cluster() -> bool:
    """Initialize Ray cluster for Docker environment"""
    try:
        import ray
        
        # For CPU-only development in Docker
        if os.getenv("CUDA_VISIBLE_DEVICES") == "":
            logger.info("Initializing Ray cluster in CPU mode")
            ray.init(
                log_to_driver=True,
                _system_config={
                    "metrics_report_interval_ms": 1000,
                }
            )
        else:
            # GPU mode initialization would go here
            logger.info("Initializing Ray cluster in GPU mode")
            ray.init(
                log_to_driver=True,
                _system_config={
                    "metrics_report_interval_ms": 1000,
                }
            )
        return True
            
    except Exception as e:
        logger.error(f"Failed to initialize Ray cluster: {e}")
        return False

def get_ray_cluster_status() -> Dict[str, Any]:
    """Get current Ray cluster status"""
    try:
        import ray
        from ray import status
        
        if not ray.is_initialized():
            return {
                "status": "not_initialized",
                "nodes": 0,
                "error": "Ray cluster not initialized"
            }
        
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        
        return {
            "status": "ready",
            "nodes": len(ray.nodes()),
            "cpus": int(cluster_resources.get("CPU", 0)),
            "gpus": int(cluster_resources.get("GPU", 0)),
            "memory_gb": cluster_resources.get("memory", 0) / (1024**3),
            "dashboard_url": "http://localhost:8265",
            "cluster_resources": cluster_resources,
            "available_resources": available_resources
        }
    except Exception as e:
        logger.error(f"Failed to get Ray cluster status: {e}")
        return {
            "status": "error",
            "nodes": 0,
            "error": str(e)
        }