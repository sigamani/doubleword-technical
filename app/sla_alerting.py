"""
SLA Tracking and Automated Alerting for Batch Inference Jobs
Monitors job ETAs against 24-hour SLA windows and triggers alerts when at risk
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any
from enum import Enum
from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)

# Alert severity levels
class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

# Metrics
sla_violations = Counter(
    "sla_violations_total",
    "Total number of SLA violations detected",
    ["job_id", "severity"]
)

sla_margin = Gauge(
    "sla_remaining_hours",
    "Hours remaining before SLA violation",
    ["job_id"]
)

@dataclass
class SLAAlert:
    job_id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    eta_hours: float = 0.0
    sla_remaining_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "eta_hours": self.eta_hours,
            "sla_remaining_hours": self.sla_remaining_hours,
        }

class SLATracker:
    """Monitor batch inference jobs against 24-hour SLA"""
    
    def __init__(self, redis_client: Optional[Any] = None,
                 sla_hours: float = 24.0, alert_threshold_factor: float = 0.7):
        self.redis_client = redis_client
        self.sla_hours = sla_hours
        self.alert_threshold_hours = sla_hours * alert_threshold_factor
        self.alert_callbacks: Dict[AlertSeverity, Callable] = {}
        self.active_alerts: Dict[str, SLAAlert] = {}
        self.job_data: Dict[str, Dict[str, Any]] = {}
    
    def register_alert_callback(self, severity: AlertSeverity, callback: Callable) -> None:
        """Register callback function to handle alerts"""
        self.alert_callbacks[severity] = callback
    
    def track_job(self, job_id: str, total_requests: int) -> None:
        """Start tracking a new batch job"""
        self.job_data[job_id] = {
            "job_id": job_id,
            "created_at": datetime.now(),
            "total_requests": total_requests,
            "completed_requests": 0,
            "failed_requests": 0,
            "tokens_processed": 0,
            "status": "running",
        }
        logger.info(f"Started tracking job {job_id}")
    
    def update_progress(self, job_id: str, completed: int, tokens: int, failed: int = 0) -> None:
        """Update job progress metrics"""
        if job_id not in self.job_data:
            return
        
        self.job_data[job_id].update({
            "completed_requests": completed,
            "failed_requests": failed,
            "tokens_processed": tokens,
            "last_update": datetime.now(),
        })
        self._check_sla(job_id)
    
    def _check_sla(self, job_id: str) -> None:
        """Evaluate SLA status for job"""
        try:
            if job_id not in self.job_data:
                return
            
            job = self.job_data[job_id]
            created = job["created_at"]
            elapsed = (datetime.now() - created).total_seconds() / 3600
            
            total = job.get("total_requests", 0)
            completed = job.get("completed_requests", 0)
            
            if total == 0 or completed == 0:
                return
            
            # Calculate ETA
            throughput = completed / elapsed if elapsed > 0 else 0
            if throughput <= 0:
                return
            
            remaining_requests = total - completed
            eta_hours = remaining_requests / throughput / 3600
            sla_remaining = self.sla_hours - elapsed
            
            # Update metrics
            sla_margin.labels(job_id=job_id).set(max(0, sla_remaining))
            
            # Check alert thresholds
            if sla_remaining < 0:
                # SLA VIOLATED
                self._trigger_alert(
                    job_id, AlertSeverity.CRITICAL,
                    f"SLA VIOLATED: Job exceeded 24-hour window. Elapsed: {elapsed:.2f}h",
                    eta_hours, sla_remaining
                )
            elif eta_hours > sla_remaining:
                # ETA exceeds remaining time
                self._trigger_alert(
                    job_id, AlertSeverity.CRITICAL,
                    f"SLA AT CRITICAL RISK: ETA {eta_hours:.2f}h > Remaining {sla_remaining:.2f}h",
                    eta_hours, sla_remaining
                )
            elif elapsed > self.alert_threshold_hours:
                # Approaching threshold
                self._trigger_alert(
                    job_id, AlertSeverity.WARNING,
                    f"SLA APPROACHING: {elapsed:.2f}h/{self.sla_hours}h used. ETA: {eta_hours:.2f}h",
                    eta_hours, sla_remaining
                )
        except Exception as e:
            logger.error(f"Error checking SLA for {job_id}: {e}")
    
    def _trigger_alert(self, job_id: str, severity: AlertSeverity, message: str, 
                      eta_hours: float, sla_remaining: float) -> None:
        """Trigger an alert and execute registered callback"""
        alert = SLAAlert(
            job_id=job_id,
            severity=severity,
            message=message,
            eta_hours=eta_hours,
            sla_remaining_hours=sla_remaining
        )
        
        # Store alert
        self.active_alerts[job_id] = alert
        sla_violations.labels(job_id=job_id, severity=severity.value).inc()
        
        # Execute callback if registered
        if severity in self.alert_callbacks:
            try:
                self.alert_callbacks[severity](alert)
            except Exception as e:
                logger.error(f"Error executing alert callback: {e}")
        
        # Log alert
        log_level = logging.CRITICAL if severity == AlertSeverity.CRITICAL else logging.WARNING
        logger.log(log_level, f"[{severity.value.upper()}] {message}")
    
    def complete_job(self, job_id: str) -> None:
        """Mark job as complete"""
        if job_id in self.job_data:
            self.job_data[job_id]["status"] = "completed"
        logger.info(f"Job {job_id} completed successfully")
        if job_id in self.active_alerts:
            del self.active_alerts[job_id]
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a job"""
        return self.job_data.get(job_id, None)
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> list:
        """Get list of active alerts"""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def estimate_capacity(self, throughput_req_per_sec: float) -> int:
        """Estimate number of requests that can complete within SLA"""
        return int(throughput_req_per_sec * self.sla_hours * 3600)
