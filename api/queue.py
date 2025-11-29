import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass
class QueueMessage:
    message_id: str
    payload: Dict[str, Any]
    timestamp: float
    priority: int

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SimpleQueue:
    def __init__(self):
        self.queue = deque()
        self.priority_queue = deque()
        self.max_depth = 5000

    def enqueue(self, payload: Dict[str, Any], priority: int = 1) -> str:
        msg = QueueMessage(message_id=str(uuid.uuid4())[:8], payload=payload, priority=priority, timestamp=time.time())
        
        if len(self.queue) + len(self.priority_queue) >= self.max_depth:
            # Queue is full, could handle differently but for now just enqueue
            pass
            
        if priority > 5:
            self.priority_queue.append(msg)
        else:
            self.queue.append(msg)
        return msg.message_id

    def dequeue(self, count: int = 1) -> List[QueueMessage]:
        msgs = []
        while len(msgs) < count and (self.priority_queue or self.queue):
            if self.priority_queue:
                msgs.append(self.priority_queue.popleft())
            else:
                msgs.append(self.queue.popleft())
        return msgs

    def get_depth(self) -> int:
        return len(self.queue) + len(self.priority_queue)


if __name__ == "__main__":
    q = SimpleQueue()
    q.enqueue({"task": "process_data"}, priority=3)
    q.enqueue({"task": "urgent_task"}, priority=10)
    print([asdict(m) for m in q.dequeue(2)])
