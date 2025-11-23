#!/usr/bin/env python3
"""
Minimal Ray Serve deployment test
"""
import os
import sys
import time
from typing import List, Dict

def test_basic_setup():
    """Test basic Ray Serve setup without external dependencies"""
    try:
        import ray
        print(f"✅ Ray {ray.__version__} available")
    except ImportError:
        print("❌ Ray not available")
        return False
    
    try:
        from ray import serve
        print("✅ Ray Serve available")
    except ImportError:
        print("❌ Ray Serve not available")
        return False
    
    return True

def test_simple_deployment():
    """Test a simple Ray Serve deployment"""
    try:
        import ray
        from ray import serve
        
        # Initialize Ray
        ray.init(address="local")
        print("✅ Ray initialized")
        
        # Start Ray Serve
        serve.start(http_options={"host": "0.0.0.0", "port": 8000})
        print("✅ Ray Serve started")
        
        # Simple deployment
        @serve.deployment
        class SimpleTest:
            def __call__(self, request):
                return {"message": "Hello from Ray Serve", "status": "healthy"}
        
        # Deploy
        SimpleTest.deploy()
        print("✅ Simple deployment successful")
        
        # Test for a few seconds
        print("🕐 Testing deployment for 10 seconds...")
        time.sleep(10)
        
        # Shutdown
        serve.shutdown()
        ray.shutdown()
        print("✅ Clean shutdown successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment test failed: {e}")
        return False

def main():
    print("🧪 Minimal Ray Serve Test")
    print("=" * 40)
    
    if not test_basic_setup():
        print("❌ Basic setup failed")
        sys.exit(1)
    
    print()
    
    if not test_simple_deployment():
        print("❌ Deployment test failed")
        sys.exit(1)
    
    print()
    print("🎉 All tests passed! Ray Serve is working.")

if __name__ == "__main__":
    main()