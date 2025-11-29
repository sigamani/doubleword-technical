#!/usr/bin/env python3
"""Test script for batch inference API"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_batch_workflow():
    print("🧪 Testing batch workflow...")
    
    # Test 1: Create batch
    print("\n1. Creating batch...")
    batch_request = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "input": [{"prompt": "What is 3+5?"}, {"prompt": "Hello world test"}],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    response = requests.post(f"{BASE_URL}/v1/batches", json=batch_request)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code != 200:
        print("❌ Batch creation failed")
        return
    
    batch_data = response.json()
    batch_id = batch_data["id"]
    print(f"✅ Batch created with ID: {batch_id}")
    
    # Test 2: Monitor status
    print(f"\n2. Monitoring job status for {batch_id}...")
    for i in range(10):
        status_response = requests.get(f"{BASE_URL}/v1/batches/{batch_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data["status"]
            print(f"Check {i+1}: {status}")
            
            if status in ["completed", "failed"]:
                break
        else:
            print(f"Status check failed: {status_response.status_code}")
        
        time.sleep(2)
    
    # Test 3: Get results
    print(f"\n3. Getting results...")
    results_response = requests.get(f"{BASE_URL}/v1/batches/{batch_id}/results")
    print(f"Results status: {results_response.status_code}")
    if results_response.status_code == 200:
        results_data = results_response.json()
        results = results_data.get("data", [])
        print(f"✅ Got {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result}")
    else:
        print(f"❌ Results failed: {results_response.text}")

if __name__ == "__main__":
    test_batch_workflow()