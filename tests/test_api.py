#!/usr/bin/env python3
"""Test Ray dataset creation and basic processor logic"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.vllm_runner import create_ray_dataset, preprocess_batch, postprocess_batch, InferencePipeline

def test_ray_dataset_creation():
    """Test basic Ray dataset creation"""
    print("Testing Ray dataset creation...")
    
    try:
        # Test data
        prompts = ["Hello world", "How are you?", "Test prompt 3"]
        
        # Create dataset
        ds = create_ray_dataset(prompts)
        print(f"✓ Dataset created with {ds.count()} items")
        
        # Test dataset content
        items = ds.take_all()
        for i, item in enumerate(items):
            expected_prompt = prompts[i]
            actual_prompt = item["prompt"]
            assert actual_prompt == expected_prompt, f"Expected {expected_prompt}, got {actual_prompt}"
        
        print("✓ Dataset content verification passed")
        
        # Test preprocessing
        batch = {"prompt": "Test prompt"}
        processed = preprocess_batch(batch)
        assert processed == batch, "Preprocessing should be identity for now"
        print("✓ Preprocessing works")
        
        # Test postprocessing
        batch_with_results = {"generated_texts": ["Response 1", "Response 2"]}
        postprocessed = postprocess_batch(batch_with_results)
        expected = {"results": ["Response 1", "Response 2"]}
        assert postprocessed == expected, f"Expected {expected}, got {postprocessed}"
        print("✓ Postprocessing works")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_pipeline():
    """Test InferencePipeline with vLLM integration"""
    print("\nTesting InferencePipeline...")
    
    try:
        pipeline = InferencePipeline()
        prompts = ["Test prompt 1", "Test prompt 2"]
        
        results = pipeline.execute_batch(prompts)
        
        assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"
        
        for i, result in enumerate(results):
            assert "response" in result, f"Result {i} missing 'response' key"
            assert "prompt" in result, f"Result {i} missing 'prompt' key"
            assert result["prompt"] == prompts[i], f"Result {i} prompt mismatch"
        
        print("✓ InferencePipeline execution works")
        
        # Check if results were written to file
        import os
        import json
        if os.path.exists("/tmp/batch_results.json"):
            with open("/tmp/batch_results.json", "r") as f:
                saved_results = json.load(f)
            assert len(saved_results) == len(prompts), "Saved results count mismatch"
            print("✓ Results saved to file correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ InferencePipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_ray_dataset_creation()
    success2 = test_inference_pipeline()
    success = success1 and success2
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(0 if success else 1)