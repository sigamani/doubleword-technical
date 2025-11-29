#!/usr/bin/env python3
"""Test FastAPI batch endpoints"""

import requests
import json
import time

def test_fastapi_endpoints():
    """Test FastAPI batch endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing FastAPI endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✓ Health endpoint works")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to FastAPI server. Start it first: python app/main.py")
        return False
    
    # Test /generate_batch endpoint
    try:
        batch_request = {
            "prompts": ["Hello world", "How are you?", "Test prompt"],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(f"{base_url}/generate_batch", json=batch_request)
        if response.status_code == 200:
            result = response.json()
            assert "results" in result
            assert "total_time" in result
            assert "total_prompts" in result
            assert len(result["results"]) == len(batch_request["prompts"])
            print("✓ /generate_batch endpoint works")
        else:
            print(f"❌ /generate_batch failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ /generate_batch test failed: {e}")
        return False
    
    # Test /v1/batches endpoint
    try:
        openai_request = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "input": [
                {"prompt": "Hello world"},
                {"prompt": "How are you?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(f"{base_url}/v1/batches", json=openai_request)
        if response.status_code == 200:
            batch_result = response.json()
            batch_id = batch_result.get("id")
            assert batch_id, "No batch ID returned"
            print(f"✓ /v1/batches endpoint works, batch_id: {batch_id}")
            
            # Test batch retrieval
            time.sleep(0.1)  # Small delay
            get_response = requests.get(f"{base_url}/v1/batches/{batch_id}")
            if get_response.status_code == 200:
                print("✓ Batch retrieval works")
                
                # Test batch results
                results_response = requests.get(f"{base_url}/v1/batches/{batch_id}/results")
                if results_response.status_code == 200:
                    results = results_response.json()
                    assert "data" in results
                    print("✓ Batch results retrieval works")
                else:
                    print(f"❌ Batch results failed: {results_response.status_code}")
                    return False
            else:
                print(f"❌ Batch retrieval failed: {get_response.status_code}")
                return False
        else:
            print(f"❌ /v1/batches failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ /v1/batches test failed: {e}")
        return False
    
    print("\n🎉 All FastAPI tests passed!")
    return True

if __name__ == "__main__":
    print("To run these tests:")
    print("1. Start the server: python app/main.py")
    print("2. Run this test: python test_fastapi.py")
    print("\nRunning tests now...\n")
    
    success = test_fastapi_endpoints()
    exit(0 if success else 1)