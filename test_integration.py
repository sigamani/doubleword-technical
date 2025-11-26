"""
Test configuration and client for authenticated batch jobs
"""

import os
import requests
import json
from typing import Dict, Any

# Configuration
API_BASE = "http://localhost:8000"
TEST_CONFIG = {
    "auth": {
        "valid_token": "test-valid-token-12345",
        "expired_token": "test-expired-token",
        "invalid_token": "test-invalid-token"
    },
    "job": {
        "num_samples": 100,
        "max_tokens": 128,
        "batch_size": 32,
        "concurrency": 2,
        "sla_tier": "basic"
    }
}

def get_auth_headers(token: str) -> Dict[str, str]:
    """Get authentication headers"""
    return {"Authorization": f"Bearer {token}"}

def test_unauthorized_request():
    """Test unauthorized request"""
    response = requests.post(
        f"{API_BASE}/generate_batch",
        json={"prompts": ["test prompt"]},
        headers={}
    )
    assert response.status_code == 401
    print("✅ Unauthorized request correctly rejected")

def test_valid_request():
    """Test valid authenticated request"""
    headers = get_auth_headers(TEST_CONFIG["auth"]["valid_token"])
    
    response = requests.post(
        f"{API_BASE}/generate_batch",
        json={
            "prompts": ["test prompt"],
            "max_tokens": TEST_CONFIG["job"]["max_tokens"],
            "batch_size": TEST_CONFIG["job"]["batch_size"]
        },
        headers=headers
    )
    
    assert response.status_code == 200
    result = response.json()
    assert len(result["results"]) == 1
    print("✅ Valid request succeeded")

def test_job_creation():
    """Test authenticated job creation"""
    headers = get_auth_headers(TEST_CONFIG["auth"]["valid_token"])
    
    job_request = {
        "input_path": "/tmp/test_input.json",
        "output_path": "/tmp/test_output.json",
        "num_samples": TEST_CONFIG["job"]["num_samples"],
        "max_tokens": TEST_CONFIG["job"]["max_tokens"],
        "temperature": 0.7,
        "batch_size": TEST_CONFIG["job"]["batch_size"],
        "concurrency": TEST_CONFIG"]["job"]["concurrency"],
        "sla_tier": TEST_CONFIG["job"]["sla_tier"]
    }
    
    response = requests.post(
        f"{API_BASE}/start",
        json=job_request,
        headers=headers
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "job_id" in result
    print(f"✅ Job created with ID: {result['job_id']}")

def test_rate_limiting():
    """Test rate limiting with multiple requests"""
    headers = get_auth_headers(TEST_CONFIG["auth"]["valid_token"])
    
    # Make multiple requests quickly
    responses = []
    for i in range(15):  # Should exceed basic tier limit
        response = requests.post(
            f"{API_BASE}/generate_batch",
            json={"prompts": [f"test prompt {i}"]},
            headers=headers
        )
        responses.append(response)
    
    # At least some should be rate limited
    rate_limited = any(r.status_code == 429 for r in responses)
    print(f"✅ Rate limiting test: {'passed' if rate_limited else 'failed'}")

def run_integration_tests():
    """Run all integration tests"""
    print("Running integration tests...")
    
    try:
        test_unauthorized_request()
        test_valid_request()
        test_job_creation()
        test_rate_limiting()
        
        print("✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)