import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PreemptionType(Enum):
    FCFS = "fcfs"  # First Come First Served
    LRU = "lru"    # Least Recently Used
    PRIORITY = "priority"  # Priority-based

@dataclass
class KVCacheEntry:
    """KV cache entry with metadata"""
    sequence_id: str
    created_at: float
    last_accessed: float
    tokens: int
    priority: int
    is_finished: bool = False

@dataclass
class PreemptionPolicy:
    """KV cache preemption policy configuration"""
    type: PreemptionType = PreemptionType.FCFS
    eviction_target: str = "oldest-unfinished-sequence"
    backpressure_threshold: float = 0.85
    max_cache_fraction: float = 0.9
    min_cache_fraction: float = 0.7
    
    def compute_optimal_fraction(self, available_memory: float, model_size: float) -> float:
        """Compute optimal KV cache fraction based on available memory"""
        base_fraction = (available_memory - model_size) / available_memory
        return min(self.max_cache_fraction, max(self.min_cache_fraction, base_fraction))

class KVCacheManager:
    """KV cache manager with FCFS preemption policy"""
    
    def __init__(self, policy: PreemptionPolicy):
        self.policy = policy
        self.cache_entries: Dict[str, KVCacheEntry] = {}
        self.total_cache_size = 0
        self.max_cache_size = 0
        self.preemption_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def set_max_cache_size(self, max_size: int):
        """Set maximum cache size in tokens"""
        self.max_cache_size = max_size
    
    def add_sequence(self, sequence_id: str, tokens: int, priority: int = 1) -> bool:
        """Add sequence to KV cache"""
        current_time = time.time()
        
        # Check if we need to preempt
        if not self._check_capacity(tokens):
            if not self._preempt_sequences(tokens):
                logger.warning(f"Failed to add sequence {sequence_id}, insufficient cache space")
                return False
        
        entry = KVCacheEntry(
            sequence_id=sequence_id,
            created_at=current_time,
            last_accessed=current_time,
            tokens=tokens,
            priority=priority
        )
        
        self.cache_entries[sequence_id] = entry
        self.total_cache_size += tokens
        
        logger.debug(f"Added sequence {sequence_id} with {tokens} tokens")
        return True
    
    def access_sequence(self, sequence_id: str) -> bool:
        """Access sequence and update metadata"""
        if sequence_id in self.cache_entries:
            self.cache_entries[sequence_id].last_accessed = time.time()
            self.cache_hits += 1
            return True
        else:
            self.cache_misses += 1
            return False
    
    def finish_sequence(self, sequence_id: str):
        """Mark sequence as finished"""
        if sequence_id in self.cache_entries:
            self.cache_entries[sequence_id].is_finished = True
    
    def _check_capacity(self, required_tokens: int) -> bool:
        """Check if cache has enough capacity"""
        return (self.total_cache_size + required_tokens) <= self.max_cache_size
    
    def _preempt_sequences(self, required_tokens: int) -> bool:
        """Preempt sequences based on policy"""
        if self.policy.type == PreemptionType.FCFS:
            return self._preempt_fcfs(required_tokens)
        elif self.policy.type == PreemptionType.LRU:
            return self._preempt_lru(required_tokens)
        else:
            return self._preempt_priority(required_tokens)
    
    def _preempt_fcfs(self, required_tokens: int) -> bool:
        """FCFS preemption: evict oldest unfinished sequences"""
        candidates = [
            entry for entry in self.cache_entries.values()
            if not entry.is_finished
        ]
        
        # Sort by creation time (oldest first)
        candidates.sort(key=lambda x: x.created_at)
        
        freed_tokens = 0
        for entry in candidates:
            if freed_tokens >= required_tokens:
                break
            
            self.total_cache_size -= entry.tokens
            del self.cache_entries[entry.sequence_id]
            freed_tokens += entry.tokens
            self.preemption_count += 1
            
            logger.debug(f"Preempted sequence {entry.sequence_id} (FCFS)")
        
        return freed_tokens >= required_tokens
    
    def _preempt_lru(self, required_tokens: int) -> bool:
        """LRU preemption: evict least recently used sequences"""
        candidates = list(self.cache_entries.values())
        candidates.sort(key=lambda x: x.last_accessed)
        
        freed_tokens = 0
        for entry in candidates:
            if freed_tokens >= required_tokens:
                break
            
            self.total_cache_size -= entry.tokens
            del self.cache_entries[entry.sequence_id]
            freed_tokens += entry.tokens
            self.preemption_count += 1
            
            logger.debug(f"Preempted sequence {entry.sequence_id} (LRU)")
        
        return freed_tokens >= required_tokens
    
    def _preempt_priority(self, required_tokens: int) -> bool:
        """Priority-based preemption: evict lowest priority first"""
        candidates = list(self.cache_entries.values())
        candidates.sort(key=lambda x: x.priority)
        
        freed_tokens = 0
        for entry in candidates:
            if freed_tokens >= required_tokens:
                break
            
            self.total_cache_size -= entry.tokens
            del self.cache_entries[entry.sequence_id]
            freed_tokens += entry.tokens
            self.preemption_count += 1
            
            logger.debug(f"Preempted sequence {entry.sequence_id} (Priority)")
        
        return freed_tokens >= required_tokens
    
    def get_cache_utilization(self) -> float:
        """Get current cache utilization as fraction"""
        if self.max_cache_size == 0:
            return 0.0
        return self.total_cache_size / self.max_cache_size
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses == 0:
            return 0.0
        return self.cache_hits / total_accesses
    
    def get_preemptions_per_1k_tokens(self) -> float:
        """Get preemptions per 1k tokens processed"""
        if self.total_cache_size == 0:
            return 0.0
        return (self.preemption_count * 1000) / self.total_cache_size
    
    def should_apply_backpressure(self) -> bool:
        """Check if backpressure should be applied"""
        return self.get_cache_utilization() > self.policy.backpressure_threshold
    
    def get_metrics(self) -> Dict:
        """Get comprehensive cache metrics"""
        return {
            "cache_utilization": self.get_cache_utilization(),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "preemption_count": self.preemption_count,
            "preemptions_per_1k_tokens": self.get_preemptions_per_1k_tokens(),
            "total_cache_size": self.total_cache_size,
            "max_cache_size": self.max_cache_size,
            "active_sequences": len([e for e in self.cache_entries.values() if not e.is_finished]),
            "finished_sequences": len([e for e in self.cache_entries.values() if e.is_finished]),
            "should_apply_backpressure": self.should_apply_backpressure()
        }

def create_kv_cache_manager(config: Dict) -> KVCacheManager:
    """Create KV cache manager from configuration"""
    policy = PreemptionPolicy(
        type=PreemptionType(config.get("preemption_type", "fcfs")),
        eviction_target=config.get("eviction_target", "oldest-unfinished-sequence"),
        backpressure_threshold=config.get("backpressure_threshold", 0.85),
        max_cache_fraction=config.get("max_cache_fraction", 0.9),
        min_cache_fraction=config.get("min_cache_fraction", 0.7)
    )
    
    return KVCacheManager(policy)