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
        for i in range(2):
            payload = {**sample_payload, "job_id": f"test_job_{i}"}
            queue.enqueue(payload, priorityLevels.LOW)
        
        messages = queue.dequeue(count=5)
        assert len(messages) == 2  
        assert queue.get_depth() == 0

    def test_priority_high_first(self, queue, sample_payload):
        msg_ids = []
        
        for i in range(3):
            payload = {**sample_payload, "job_id": f"low_job_{i}"}
            msg_id = queue.enqueue(payload, priorityLevels.LOW)
            msg_ids.append(("low", msg_id))
        
        for i in range(2):
            payload = {**sample_payload, "job_id": f"high_job_{i}"}
            msg_id = queue.enqueue(payload, priorityLevels.HIGH)
            msg_ids.append(("high", msg_id))
        
        assert queue.get_depth() == 5
        
        messages = queue.dequeue(count=5)
        assert len(messages) == 5
        
        for i in range(2):
            assert messages[i].priority == priorityLevels.HIGH
            assert messages[i].payload["job_id"] == f"high_job_{i}"
        
        for i in range(2, 5):
            assert messages[i].priority == priorityLevels.LOW
            assert messages[i].payload["job_id"] == f"low_job_{i-2}"

    def test_priority_mixed_ordering(self, queue, sample_payload):
        order = []
        
        msg_id1 = queue.enqueue({**sample_payload, "job_id": "job_1"}, priorityLevels.LOW)
        order.append(("low", msg_id1))
        
        msg_id2 = queue.enqueue({**sample_payload, "job_id": "job_2"}, priorityLevels.HIGH)
        order.append(("high", msg_id2))
        
        msg_id3 = queue.enqueue({**sample_payload, "job_id": "job_3"}, priorityLevels.LOW)
        order.append(("low", msg_id3))
        
        msg_id4 = queue.enqueue({**sample_payload, "job_id": "job_4"}, priorityLevels.HIGH)
        order.append(("high", msg_id4))
        
        msg_id5 = queue.enqueue({**sample_payload, "job_id": "job_5"}, priorityLevels.LOW)
        order.append(("low", msg_id5))
        
        messages = queue.dequeue(count=5)
        
        expected_order = ["job_2", "job_4", "job_1", "job_3", "job_5"]
        actual_order = [msg.payload["job_id"] for msg in messages]
        
        assert actual_order == expected_order

    def test_priority_fifo_within_same_priority(self, queue, sample_payload):
        high_msg_ids = []
        low_msg_ids = []
        
        for i in range(3):
            payload = {**sample_payload, "job_id": f"high_job_{i}"}
            msg_id = queue.enqueue(payload, priorityLevels.HIGH)
            high_msg_ids.append(msg_id)
        
        for i in range(3):
            payload = {**sample_payload, "job_id": f"low_job_{i}"}
            msg_id = queue.enqueue(payload, priorityLevels.LOW)
            low_msg_ids.append(msg_id)
        
        messages = queue.dequeue(count=3)
        for i, msg in enumerate(messages):
            assert msg.priority == priorityLevels.HIGH
            assert msg.message_id == high_msg_ids[i]
        
        messages = queue.dequeue(count=3)
        for i, msg in enumerate(messages):
            assert msg.priority == priorityLevels.LOW
            assert msg.message_id == low_msg_ids[i]

    def test_partial_dequeue_with_priority(self, queue, sample_payload):
        low_ids = []
        high_ids = []
        
        for i in range(2):
            low_ids.append(queue.enqueue({**sample_payload, "job_id": f"low_{i}"}, priorityLevels.LOW))
        
        for i in range(2):
            high_ids.append(queue.enqueue({**sample_payload, "job_id": f"high_{i}"}, priorityLevels.HIGH))
        
        for i in range(2, 4):
            low_ids.append(queue.enqueue({**sample_payload, "job_id": f"low_{i}"}, priorityLevels.LOW))
        
        messages = queue.dequeue(count=3)
        assert len(messages) == 3
        
        assert messages[0].priority == priorityLevels.HIGH
        assert messages[1].priority == priorityLevels.HIGH
        
        assert messages[2].priority == priorityLevels.LOW
        
        assert queue.get_depth() == 3