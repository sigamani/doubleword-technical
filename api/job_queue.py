"""Simple in-memory queue implementation for batch jobs."""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import time
import uuid
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from api.models import priorityLevels

logger = logging.getLogger(__name__)


@dataclass
class QueueMessage:
    message_id: str
    payload: Dict[str, Any]
    timestamp: float
    priority: priorityLevels
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
            
class SimpleQueue:
    def __init__(self):
        self.queue = deque()
        self.priority_queue = deque()
        self.max_depth = 5000
        self._lock = threading.Lock()

    def enqueue(self, payload: Dict[str, Any], priority: priorityLevels) -> Optional[str]:
        msg = QueueMessage(
            message_id=str(uuid.uuid4())[:8],
            payload=payload,
            priority=priority,
            timestamp=time.time()
        )
        
        with self._lock:
            if len(self.queue) + len(self.priority_queue) >= self.max_depth:
                logger.warning(f"Queue at max depth {self.max_depth}, rejecting job {msg.message_id}")
                return None 
                
            if priority == priorityLevels.HIGH:
                self.priority_queue.append(msg)
            elif priority == priorityLevels.LOW:
                self.queue.append(msg)
        
        return msg.message_id
    
    def dequeue(self, count: int = 1) -> List[QueueMessage]:
        msgs = []
        with self._lock: 
            while len(msgs) < count and (self.priority_queue or self.queue):
                if self.priority_queue:
                    msgs.append(self.priority_queue.popleft())
                else:
                    msgs.append(self.queue.popleft())
        return msgs

    def get_depth(self) -> int:
        with self._lock: 
            return len(self.queue) + len(self.priority_queue)
