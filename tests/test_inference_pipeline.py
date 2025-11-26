"""
Unit tests for core components
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from app.core.inference import InferencePipeline, BatchMetrics, InferenceMonitor, SLATier
from app.core.config import ModelConfig, InferenceConfig
from app.core.context import AppContext, get_context, set_context

class TestInferencePipeline:
    """Test suite for inference pipeline"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_processor = Mock()
        self.mock_metrics = BatchMetrics(total_requests=100)
        self.mock_monitor = InferenceMonitor(
            self.mock_metrics,
            SLATier.BASIC
        )
        self.pipeline = InferencePipeline(self.mock_processor)
    
    def test_preprocess(self):
        """Test preprocessing function"""
        test_data = [{"prompt": "test prompt"}]
        result = self.pipeline.preprocess(test_data)
        
        assert len(result) == 1
        assert result[0]["messages"][0]["role"] == "system"
        assert result[0]["messages"][1]["role"] == "user"
        assert result[0]["sampling_params"]["temperature"] == 0.7
    
    def test_metrics_calculation(self):
        """Test metrics calculations"""
        self.mock_metrics.completed_requests = 50
        self.mock_metrics.tokens_processed = 5000
        
        assert self.mock_metrics.progress_pct() == 50.0
        assert self.mock_metrics.throughput_per_sec() > 0
        assert self.mock_metrics.tokens_per_sec() > 0
    
    def test_sla_monitoring(self):
        """Test SLA monitoring"""
        self.mock_metrics.completed_requests = 80
        self.mock_metrics.start_time = 1000  # 1000 seconds ago
        
        # Should trigger SLA warning
        assert not self.mock_monitor.check_sla()
    
    def test_artifact_creation(self):
        """Test artifact creation and storage"""
        test_data = [{"prompt": "test"}]
        artifact = self.pipeline.create_data_artifact(test_data)
        
        assert artifact.sha256_hash
        assert artifact.content_hash
        assert artifact.version
        assert artifact.created_at
        assert artifact.size_bytes > 0

if __name__ == "__main__":
    pytest.main([__name__])