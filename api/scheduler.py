import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List

class SLAStatus(Enum):
    WITHIN_TARGET = "within_target"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    COMPLETED = "completed"

@dataclass
class SLAMetrics:
    job_id: str
    target_hours: float = 24.0
    samples_total: int = 0
    samples_processed: int = 0
    sla_status: SLAStatus = SLAStatus.WITHIN_TARGET
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def update_progress(self, processed: int):
        self.samples_processed = processed
        if self.samples_processed >= self.samples_total:
            self.sla_status = SLAStatus.COMPLETED
            self.completed_at = time.time()
        else:
            elapsed = (time.time() - (self.started_at or time.time()))
            if elapsed > self.target_hours * 3600 * 0.8:
                self.sla_status = SLAStatus.AT_RISK

    def mark_started(self):
        self.started_at = time.time()

@dataclass
class SLAManager:
    target_hours: float = 24.0
    sla_metrics: Dict[str, SLAMetrics] = field(default_factory=dict)

    def create_job(self, job_id: str, total_samples: int) -> SLAMetrics:
        sla = SLAMetrics(job_id=job_id, target_hours=self.target_hours, samples_total=total_samples)
        self.sla_metrics[job_id] = sla
        return sla

    def update_job(self, job_id: str, processed: int):
        if job_id in self.sla_metrics:
            self.sla_metrics[job_id].update_progress(processed)

    def mark_started(self, job_id: str):
        if job_id in self.sla_metrics:
            self.sla_metrics[job_id].mark_started()

    def get_status(self, job_id: str) -> Optional[SLAStatus]:
        return self.sla_metrics.get(job_id).sla_status if job_id in self.sla_metrics else None

# Usage example
if __name__ == "__main__":
    manager = SLAManager()
    job = manager.create_job("job_1", 100)
    job.mark_started()
    manager.update_job("job_1", 30)
    print(manager.get_status("job_1"))  # AT_RISK or WITHIN_TARGET depending on time
