#!/usr/bin/env python3
"""
Simple test script to submit 1000 batch inference job
"""

import requests
import json
import time

def submit_batch_job():
    """Submit batch job to running service"""
    
    # Job parameters
    job_data = {
        "input_path": "/tmp/test_input.json",
        "output_path": "/tmp/test_output.json",
        "num_samples": 1000,
        "max_tokens": 512,
        "temperature": 0.7,
        "batch_size": 128,
        "concurrency": 2
    }
    
    try:
        # Submit job
        response = requests.post("http://localhost:8000/start", json=job_data)
        if response.status_code == 200:
            job_result = response.json()
            print(f"Job submitted successfully!")
            print(f"Job ID: {job_result['job_id']}")
            print(f"Status: {job_result['status']}")
            print(f"Estimated completion: {job_result['estimated_completion_hours']:.2f} hours")
            return job_result['job_id']
        else:
            print(f"Failed to submit job: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error submitting job: {e}")
        return None

def check_job_status(job_id):
    """Check job status via Redis"""
    
    try:
        # Connect to Redis to check job status
        import redis
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        job_data = redis_client.hgetall(f"job:{job_id}")
        if job_data:
            print(f"Job {job_id} status:")
            for key, value in job_data.items():
                print(f"  {key}: {value}")
        else:
            print(f"No job data found for {job_id}")
            
    except Exception as e:
        print(f"Error checking job status: {e}")

if __name__ == "__main__":
    print("Submitting 1000 sample batch inference job...")
    job_id = submit_batch_job()
    
    if job_id:
        print(f"\nChecking job status for {job_id}...")
        time.sleep(2)  # Small delay
        check_job_status(job_id)