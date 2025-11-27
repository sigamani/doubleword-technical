import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LoadSheddingAction(Enum):
    REJECT_NEW_REQUESTS = "reject_new_requests"
    DEFER_TO_NEXT_BATCH = "defer_to_next_batch"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    LOWER_CONCURRENCY = "lower_concurrency"
    ESCALATE_PRIORITY = "escalate_priority"

class LoadSheddingTrigger(Enum):
    QUEUE_DEPTH_EXCEEDED = "queue_depth_exceeded"
    SLA_AT_RISK = "sla_at_risk"
    MEMORY_PRESSURE = "memory_pressure"
    HIGH_ERROR_RATE = "high_error_rate"
    CONCURRENCY_LIMIT = "concurrency_limit"

@dataclass
class LoadSheddingPolicy:
    """Load shedding policy configuration"""
    queue_depth_threshold: int = 5000
    sla_risk_threshold_hours: float = 2.0
    memory_pressure_threshold: float = 0.95
    error_rate_threshold: float = 0.05
    concurrency_limit_threshold: int = 100
    
    primary_action: LoadSheddingAction = LoadSheddingAction.REJECT_NEW_REQUESTS
    fallback_actions: List[LoadSheddingAction] = None
    
    auto_recovery: bool = True
    recovery_check_interval: int = 300  # 5 minutes
    max_shedding_duration: int = 3600  # 1 hour
    
    def __post_init__(self):
        if self.fallback_actions is None:
            self.fallback_actions = [
                LoadSheddingAction.DEFER_TO_NEXT_BATCH,
                LoadSheddingAction.REDUCE_BATCH_SIZE
            ]

@dataclass
class LoadSheddingState:
    """Current load shedding state"""
    is_active: bool = False
    trigger: Optional[LoadSheddingTrigger] = None
    action: Optional[LoadSheddingAction] = None
    start_time: float = 0.0
    requests_shed: int = 0
    last_recovery_check: float = 0.0
    
    def should_check_recovery(self, interval: int) -> bool:
        return time.time() - self.last_recovery_check > interval
    
    def is_expired(self, max_duration: int) -> bool:
        return time.time() - self.start_time > max_duration

class LoadSheddingManager:
    """Load shedding manager for SLA guarantees"""
    
    def __init__(self, policy: LoadSheddingPolicy):
        self.policy = policy
        self.state = LoadSheddingState()
        self.metrics_history = []
        self.max_history_size = 1000
        
    def evaluate_load_shedding(self, metrics: Dict[str, Any]) -> Tuple[bool, Optional[LoadSheddingAction]]:
        """Evaluate if load shedding should be activated"""
        triggers = []
        
        # Check queue depth
        queue_depth = metrics.get("queue_depth", 0)
        if queue_depth > self.policy.queue_depth_threshold:
            triggers.append((LoadSheddingTrigger.QUEUE_DEPTH_EXCEEDED, queue_depth))
        
        # Check SLA risk
        eta_hours = metrics.get("eta_hours", float('inf'))
        remaining_hours = metrics.get("remaining_sla_hours", 24.0)
        if eta_hours < remaining_hours and (remaining_hours - eta_hours) < self.policy.sla_risk_threshold_hours:
            triggers.append((LoadSheddingTrigger.SLA_AT_RISK, eta_hours))
        
        # Check memory pressure
        memory_utilization = metrics.get("memory_utilization", 0.0)
        if memory_utilization > self.policy.memory_pressure_threshold:
            triggers.append((LoadSheddingTrigger.MEMORY_PRESSURE, memory_utilization))
        
        # Check error rate
        error_rate = metrics.get("error_rate", 0.0)
        if error_rate > self.policy.error_rate_threshold:
            triggers.append((LoadSheddingTrigger.HIGH_ERROR_RATE, error_rate))
        
        # Check concurrency limit
        active_concurrency = metrics.get("active_concurrency", 0)
        if active_concurrency > self.policy.concurrency_limit_threshold:
            triggers.append((LoadSheddingTrigger.CONCURRENCY_LIMIT, active_concurrency))
        
        # Determine action
        if triggers and not self.state.is_active:
            trigger, value = max(triggers, key=lambda x: self._get_trigger_priority(x[0]))
            action = self._select_action(trigger, metrics)
            
            self._activate_load_shedding(trigger, action)
            logger.warning(f"Load shedding activated: {trigger.value} (value: {value})")
            return True, action
        
        # Check recovery
        if self.state.is_active and self._should_deactivate(metrics):
            self._deactivate_load_shedding()
            logger.info("Load shedding deactivated - conditions recovered")
            return False, None
        
        return self.state.is_active, self.state.action
    
    def _get_trigger_priority(self, trigger: LoadSheddingTrigger) -> int:
        """Get priority for trigger (higher = more urgent)"""
        priorities = {
            LoadSheddingTrigger.MEMORY_PRESSURE: 5,
            LoadSheddingTrigger.SLA_AT_RISK: 4,
            LoadSheddingTrigger.QUEUE_DEPTH_EXCEEDED: 3,
            LoadSheddingTrigger.HIGH_ERROR_RATE: 2,
            LoadSheddingTrigger.CONCURRENCY_LIMIT: 1
        }
        return priorities.get(trigger, 0)
    
    def _select_action(self, trigger: LoadSheddingTrigger, metrics: Dict[str, Any]) -> LoadSheddingAction:
        """Select appropriate load shedding action"""
        # Primary action based on trigger
        if trigger == LoadSheddingTrigger.QUEUE_DEPTH_EXCEEDED:
            return LoadSheddingAction.REJECT_NEW_REQUESTS
        elif trigger == LoadSheddingTrigger.SLA_AT_RISK:
            return LoadSheddingAction.ESCALATE_PRIORITY
        elif trigger == LoadSheddingTrigger.MEMORY_PRESSURE:
            return LoadSheddingAction.REDUCE_BATCH_SIZE
        elif trigger == LoadSheddingTrigger.HIGH_ERROR_RATE:
            return LoadSheddingAction.DEFER_TO_NEXT_BATCH
        else:
            return self.policy.primary_action
    
    def _activate_load_shedding(self, trigger: LoadSheddingTrigger, action: LoadSheddingAction):
        """Activate load shedding"""
        self.state.is_active = True
        self.state.trigger = trigger
        self.state.action = action
        self.state.start_time = time.time()
        self.state.requests_shed = 0
        self.state.last_recovery_check = time.time()
    
    def _should_deactivate(self, metrics: Dict[str, Any]) -> bool:
        """Check if load shedding should be deactivated"""
        if not self.policy.auto_recovery:
            return False
        
        if self.state.is_expired(self.policy.max_shedding_duration):
            logger.warning("Load shedding expired - forcing deactivation")
            return True
        
        if not self.state.should_check_recovery(self.policy.recovery_check_interval):
            return False
        
        # Check if conditions improved
        queue_depth = metrics.get("queue_depth", 0)
        eta_hours = metrics.get("eta_hours", float('inf'))
        remaining_hours = metrics.get("remaining_sla_hours", 24.0)
        memory_utilization = metrics.get("memory_utilization", 0.0)
        error_rate = metrics.get("error_rate", 0.0)
        
        # All conditions must be healthy
        conditions_healthy = (
            queue_depth < self.policy.queue_depth_threshold * 0.8 and
            (eta_hours >= remaining_hours or (remaining_hours - eta_hours) > self.policy.sla_risk_threshold_hours * 2) and
            memory_utilization < self.policy.memory_pressure_threshold * 0.9 and
            error_rate < self.policy.error_rate_threshold * 0.5
        )
        
        self.state.last_recovery_check = time.time()
        return conditions_healthy
    
    def _deactivate_load_shedding(self):
        """Deactivate load shedding"""
        self.state.is_active = False
        self.state.trigger = None
        self.state.action = None
        self.state.start_time = 0.0
    
    def should_shed_request(self, request_metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if a request should be shed"""
        if not self.state.is_active:
            return False, "Load shedding inactive"
        
        self.state.requests_shed += 1
        
        # Apply action-specific logic
        if self.state.action == LoadSheddingAction.REJECT_NEW_REQUESTS:
            return True, f"Queue depth exceeded ({self.policy.queue_depth_threshold})"
        
        elif self.state.action == LoadSheddingAction.DEFER_TO_NEXT_BATCH:
            # Could implement deferral logic here
            return True, "Deferring to next batch window"
        
        elif self.state.action == LoadSheddingAction.REDUCE_BATCH_SIZE:
            # Would be handled by batch size adjustment
            return False, "Reducing batch size instead"
        
        elif self.state.action == LoadSheddingAction.LOWER_CONCURRENCY:
            # Would be handled by concurrency adjustment
            return False, "Lowering concurrency instead"
        
        elif self.state.action == LoadSheddingAction.ESCALATE_PRIORITY:
            # Would be handled by priority adjustment
            return False, "Escalating priority instead"
        
        return True, f"Load shedding active: {self.state.action.value}"
    
    def get_adjusted_parameters(self, original_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get adjusted parameters based on load shedding action"""
        if not self.state.is_active:
            return original_params
        
        adjusted = original_params.copy()
        
        if self.state.action == LoadSheddingAction.REDUCE_BATCH_SIZE:
            adjusted["batch_size"] = max(16, original_params.get("batch_size", 128) // 2)
            logger.info(f"Reduced batch size to {adjusted['batch_size']}")
        
        elif self.state.action == LoadSheddingAction.LOWER_CONCURRENCY:
            adjusted["concurrency"] = max(1, original_params.get("concurrency", 4) // 2)
            logger.info(f"Lowered concurrency to {adjusted['concurrency']}")
        
        elif self.state.action == LoadSheddingAction.ESCALATE_PRIORITY:
            adjusted["priority"] = original_params.get("priority", 1) + 1
            logger.info(f"Escalated priority to {adjusted['priority']}")
        
        return adjusted
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get load shedding metrics"""
        uptime = time.time() - self.state.start_time if self.state.is_active else 0
        
        return {
            "load_shedding_active": self.state.is_active,
            "load_shedding_trigger": self.state.trigger.value if self.state.trigger else None,
            "load_shedding_action": self.state.action.value if self.state.action else None,
            "load_shedding_uptime_seconds": uptime,
            "requests_shed": self.state.requests_shed,
            "shed_rate_per_second": self.state.requests_shed / uptime if uptime > 0 else 0,
            "auto_recovery_enabled": self.policy.auto_recovery,
            "recovery_check_interval": self.policy.recovery_check_interval,
            "max_shedding_duration": self.policy.max_shedding_duration
        }
    
    def update_metrics_history(self, metrics: Dict[str, Any]):
        """Update metrics history for trend analysis"""
        timestamp = time.time()
        self.metrics_history.append({
            "timestamp": timestamp,
            "metrics": metrics.copy(),
            "load_shedding_active": self.state.is_active
        })
        
        # Trim history
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]

def create_load_shedding_manager(config: Dict[str, Any]) -> LoadSheddingManager:
    """Create load shedding manager from configuration"""
    policy = LoadSheddingPolicy(
        queue_depth_threshold=config.get("queue_depth_threshold", 5000),
        sla_risk_threshold_hours=config.get("sla_risk_threshold_hours", 2.0),
        memory_pressure_threshold=config.get("memory_pressure_threshold", 0.95),
        error_rate_threshold=config.get("error_rate_threshold", 0.05),
        concurrency_limit_threshold=config.get("concurrency_limit_threshold", 100),
        primary_action=LoadSheddingAction(config.get("primary_action", "reject_new_requests")),
        auto_recovery=config.get("auto_recovery", True),
        recovery_check_interval=config.get("recovery_check_interval", 300),
        max_shedding_duration=config.get("max_shedding_duration", 3600)
    )
    
    return LoadSheddingManager(policy)