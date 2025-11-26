#!/usr/bin/env python3
"""
Direct batch inference test using 1000 examples
"""

import requests
import json
import time

def test_direct_batch_inference():
    """Test direct batch inference with 1000 examples"""
    
    # Create 1000 test prompts
    base_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "How do neural networks work?",
        "What is deep learning?",
        "Describe computer vision.",
        "What is natural language processing?",
        "How does reinforcement learning work?",
        "What are transformers in AI?",
        "Explain convolutional neural networks.",
        "What is supervised learning?"
    ]
    
    # Repeat to create 1000 prompts
    prompts = base_prompts * 100  # 1000 prompts
    prompts = prompts[:1000]
    
    print(f"Created {len(prompts)} test prompts")
    
    # Batch inference request
    batch_request = {
        "prompts": prompts,
        "max_tokens": 128,
        "temperature": 0.7,
        "batch_size": 128
    }
    
    try:
        print("Submitting batch inference request...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/generate_batch", 
            json=batch_request,
            timeout=600  # 10 minute timeout
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("Batch inference completed successfully!")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Results returned: {len(result.get('results', []))}")
            print(f"Throughput: {result.get('throughput', 0):.2f} req/s")
            print(f"Service reported time: {result.get('total_time', 0):.2f} seconds")
            
            # Show a few sample results
            results = result.get('results', [])
            if results:
                print("\nSample results:")
                for i, res in enumerate(results[:3]):
                    print(f"  {i+1}. {res['text'][:100]}...")
            
            return True
            
        else:
            print(f"Batch inference failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing direct batch inference with 1000 examples...")
    success = test_direct_batch_inference()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")