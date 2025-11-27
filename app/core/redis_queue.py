import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueueStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    BACKLOG = "backlog"
    OVERLOADED = "overloaded"

@dataclass
class QueueMessage:
    """Queue message with metadata"""
    message_id: str
    payload: Dict[str, Any]
    timestamp: float
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "QueueMessage":
        return cls(
            message_id=data["message_id"],
            payload=data["payload"],
            timestamp=data["timestamp"],
            priority=data.get("priority", 1),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )

@dataclass
class ConsumerGroup:
    """Redis Streams consumer group"""
    name: str
    consumer_id: str
    last_message_id: str = "0"
    pending_count: int = 0
    processed_count: int = 0
    
    def get_info(self) -> Dict:
        return {
            "name": self.name,
            "consumer_id": self.consumer_id,
            "last_message_id": self.last_message_id,
            "pending_count": self.pending_count,
            "processed_count": self.processed_count
        }

class RedisStreamQueue:
    """Redis Streams based queue with consumer groups"""
    
    def __init__(self, stream_name: str, redis_client=None):
        self.stream_name = stream_name
        self.redis_client = redis_client
        self.consumer_groups: Dict[str, ConsumerGroup] = {}
        self.max_queue_depth = 5000
        self.message_timeout = 3600  # 1 hour
        self.block_timeout = 1000   # 1 second
        
        # Initialize stream
        self._initialize_stream()
    
    def _initialize_stream(self):
        """Initialize Redis stream and consumer groups"""
        if not self.redis_client:
            logger.warning("No Redis client provided, using in-memory queue")
            return
        
        try:
            # Create stream if not exists
            self.redis_client.xgroup_create(
                self.stream_name, "ray_workers", id="0", mkstream=True
            )
            logger.info(f"Created Redis stream {self.stream_name} with consumer group ray_workers")
        except Exception as e:
            logger.debug(f"Stream may already exist: {e}")
    
    def add_consumer_group(self, group_name: str, consumer_id: str) -> ConsumerGroup:
        """Add consumer group for Ray workers"""
        group = ConsumerGroup(name=group_name, consumer_id=consumer_id)
        self.consumer_groups[f"{group_name}_{consumer_id}"] = group
        
        if self.redis_client:
            try:
                self.redis_client.xgroup_create(
                    self.stream_name, group_name, id="0", mkstream=True
                )
                logger.info(f"Created consumer group {group_name}")
            except Exception as e:
                logger.debug(f"Consumer group may already exist: {e}")
        
        return group
    
    def enqueue(self, payload: Dict[str, Any], priority: int = 1) -> str:
        """Add message to queue"""
        message_id = f"msg_{int(time.time() * 1000000)}"
        message = QueueMessage(
            message_id=message_id,
            payload=payload,
            timestamp=time.time(),
            priority=priority
        )
        
        if self.redis_client:
            return self._enqueue_redis(message)
        else:
            return self._enqueue_memory(message)
    
    def _enqueue_redis(self, message: QueueMessage) -> str:
        """Enqueue message using Redis Streams"""
        try:
            # Check queue depth
            if self._get_queue_depth() >= self.max_queue_depth:
                logger.warning(f"Queue depth exceeded max {self.max_queue_depth}")
                return None
            
            # Add to stream
            result = self.redis_client.xadd(
                self.stream_name,
                {
                    "message_id": message.message_id,
                    "payload": json.dumps(message.payload),
                    "timestamp": str(message.timestamp),
                    "priority": str(message.priority),
                    "retry_count": str(message.retry_count),
                    "max_retries": str(message.max_retries)
                }
            )
            
            logger.debug(f"Enqueued message {message.message_id} to stream {self.stream_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to enqueue message: {e}")
            return None
    
    def _enqueue_memory(self, message: QueueMessage) -> str:
        """Fallback in-memory enqueue"""
        if not hasattr(self, '_memory_queue'):
            self._memory_queue = []
        
        self._memory_queue.append(message)
        return message.message_id
    
    def dequeue(self, group_name: str, consumer_id: str, count: int = 1) -> List[QueueMessage]:
        """Dequeue messages for consumer"""
        if self.redis_client:
            return self._dequeue_redis(group_name, consumer_id, count)
        else:
            return self._dequeue_memory(group_name, consumer_id, count)
    
    def _dequeue_redis(self, group_name: str, consumer_id: str, count: int) -> List[QueueMessage]:
        """Dequeue using Redis Streams with consumer groups"""
        try:
            # Read from consumer group
            results = self.redis_client.xreadgroup(
                group_name, consumer_id,
                {self.stream_name: ">"},
                count=count,
                block=self.block_timeout
            )
            
            messages = []
            for stream, msgs in results:
                for msg_id, fields in msgs:
                    message = QueueMessage(
                        message_id=fields.get("message_id", msg_id.decode()),
                        payload=json.loads(fields.get("payload", "{}")),
                        timestamp=float(fields.get("timestamp", "0")),
                        priority=int(fields.get("priority", "1")),
                        retry_count=int(fields.get("retry_count", "0")),
                        max_retries=int(fields.get("max_retries", "3"))
                    )
                    messages.append(message)
            
            # Update consumer group stats
            group_key = f"{group_name}_{consumer_id}"
            if group_key in self.consumer_groups:
                self.consumer_groups[group_key].pending_count += len(messages)
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to dequeue from Redis: {e}")
            return []
    
    def _dequeue_memory(self, group_name: str, consumer_id: str, count: int) -> List[QueueMessage]:
        """Fallback in-memory dequeue"""
        if not hasattr(self, '_memory_queue'):
            self._memory_queue = []
        
        messages = self._memory_queue[:count]
        self._memory_queue = self._memory_queue[count:]
        return messages
    
    def acknowledge(self, group_name: str, consumer_id: str, message_ids: List[str]) -> bool:
        """Acknowledge message processing"""
        if self.redis_client:
            return self._acknowledge_redis(group_name, message_ids)
        else:
            return self._acknowledge_memory(message_ids)
    
    def _acknowledge_redis(self, group_name: str, message_ids: List[str]) -> bool:
        """Acknowledge using Redis Streams"""
        try:
            self.redis_client.xack(self.stream_name, group_name, *message_ids)
            
            # Update consumer group stats
            for group_key, group in self.consumer_groups.items():
                if group.name == group_name:
                    group.pending_count -= len(message_ids)
                    group.processed_count += len(message_ids)
                    break
            
            logger.debug(f"Acknowledged {len(message_ids)} messages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge messages: {e}")
            return False
    
    def _acknowledge_memory(self, message_ids: List[str]) -> bool:
        """Fallback in-memory acknowledge"""
        return True
    
    def get_queue_status(self) -> QueueStatus:
        """Get current queue status"""
        depth = self._get_queue_depth()
        
        if depth == 0:
            return QueueStatus.IDLE
        elif depth > self.max_queue_depth * 0.8:
            return QueueStatus.OVERLOADED
        elif depth > self.max_queue_depth * 0.5:
            return QueueStatus.BACKLOG
        else:
            return QueueStatus.ACTIVE
    
    def _get_queue_depth(self) -> int:
        """Get current queue depth"""
        if self.redis_client:
            try:
                info = self.redis_client.xinfo_stream(self.stream_name)
                return info.get("length", 0)
            except Exception as e:
                logger.debug(f"Failed to get stream info: {e}")
                return 0
        else:
            return len(getattr(self, '_memory_queue', []))
    
    def get_consumer_group_info(self, group_name: str) -> Optional[Dict]:
        """Get consumer group information"""
        if self.redis_client:
            try:
                info = self.redis_client.xinfo_groups(self.stream_name)
                for group in info:
                    if group.get("name") == group_name:
                        return group
            except Exception as e:
                logger.debug(f"Failed to get consumer group info: {e}")
        
        # Fallback to local tracking
        for group in self.consumer_groups.values():
            if group.name == group_name:
                return group.get_info()
        
        return None
    
    def cleanup_expired_messages(self):
        """Clean up expired messages"""
        if not self.redis_client:
            return
        
        try:
            # Get pending messages older than timeout
            pending = self.redis_client.xpending(
                self.stream_name, "ray_workers", min="-", max="+", count=1000
            )
            
            current_time = time.time()
            expired_ids = []
            
            for msg_id, consumer, elapsed, deliveries in pending:
                if elapsed > self.message_timeout * 1000:  # Redis returns milliseconds
                    expired_ids.append(msg_id)
            
            if expired_ids:
                # Claim and delete expired messages
                self.redis_client.xclaim(
                    self.stream_name, "ray_workers", "cleanup",
                    *expired_ids, min_idle_time=self.message_timeout * 1000
                )
                self.redis_client.xdel(self.stream_name, *expired_ids)
                logger.info(f"Cleaned up {len(expired_ids)} expired messages")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired messages: {e}")
    
    def get_metrics(self) -> Dict:
        """Get comprehensive queue metrics"""
        return {
            "queue_depth": self._get_queue_depth(),
            "max_queue_depth": self.max_queue_depth,
            "queue_status": self.get_queue_status().value,
            "consumer_groups": len(self.consumer_groups),
            "message_timeout": self.message_timeout,
            "stream_name": self.stream_name
        }

def create_redis_queue(stream_name: str, redis_client=None) -> RedisStreamQueue:
    """Create Redis stream queue"""
    return RedisStreamQueue(stream_name, redis_client)