import pytest
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from api.job_queue import SimpleQueue, QueueMessage
from api.models import priorityLevels


class TestSimpleQueue:
    
    @pytest.fixture
    def queue(self):
        return SimpleQueue()
    
    @pytest.fixture
    def sample_payload(self):
        return {
            "job_id": "test_job_123",
            "input_file": "/tmp/test_input.jsonl",
            "output_file": "/tmp/test_output.jsonl",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_tokens": 256
        }
    
    def test_enqueue_single_message(self, queue, sample_payload):
        msg_id = queue.enqueue(sample_payload, priorityLevels.LOW)
        
        assert isinstance(msg_id, str)
        assert len(msg_id) == 8  
        
        assert queue.get_depth() == 1
        
        messages = queue.dequeue(count=1)
        assert len(messages) == 1
        
        msg = messages[0]
        assert isinstance(msg, QueueMessage)
        assert msg.message_id == msg_id
        assert msg.payload == sample_payload
        assert msg.priority == priorityLevels.LOW
        assert isinstance(msg.timestamp, float)
        assert msg.timestamp > 0

    def test_enqueue_multiple_messages(self, queue, sample_payload):
        msg_ids = []
        
        for i in range(5):
            payload = {**sample_payload, "job_id": f"test_job_{i}"}
            msg_id = queue.enqueue(payload, priorityLevels.LOW)
            msg_ids.append(msg_id)
        
        assert len(set(msg_ids)) == 5
        
        assert queue.get_depth() == 5
        
        messages = queue.dequeue(count=5)
        assert len(messages) == 5
        
        for i, msg in enumerate(messages):
            assert msg.message_id == msg_ids[i]
            assert msg.payload["job_id"] == f"test_job_{i}"

    def test_dequeue_empty_queue(self, queue):
        messages = queue.dequeue(count=1)
        assert len(messages) == 0
        assert queue.get_depth() == 0

    def test_dequeue_partial_count(self, queue, sample_payload):
        for i in range(3):
            payload = {**sample_payload, "job_id": f"test_job_{i}"}
            queue.enqueue(payload, priorityLevels.LOW)
        
        messages = queue.dequeue(count=2)
        assert len(messages) == 2
        assert queue.get_depth() == 1  
        
        remaining = queue.dequeue(count=1)
        assert len(remaining) == 1
        assert queue.get_depth() == 0

    def test_dequeue_more_than_available(self, queue, sample_payload):
        """Test dequeuing more messages than available."""
        for i in range(2):
            payload = {**sample_payload, "job_id": f"test_job_{i}"}
            queue.enqueue(payload, priorityLevels.LOW)
        
        messages = queue.dequeue(count=5)
        assert len(messages) == 2  
        assert queue.get_depth() == 0