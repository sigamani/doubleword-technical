#!/usr/bin/env python3
"""
Simple runner for the testing agent
"""

import subprocess
import sys
import os

def main():
    """Run the testing agent"""
    print("🚀 Starting Repository Testing Agent")
    print("=" * 50)
    
    # Check if testing agent exists
    if not os.path.exists("testing_agent.py"):
        print("❌ testing_agent.py not found!")
        return 1
    
    # Run the testing agent
    try:
        result = subprocess.run([
            sys.executable, "testing_agent.py", 
            "--verbose",
            "--report-file", "test_execution_report.json"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"\n{'='*50}")
        print(f"Testing agent completed with exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ All tests completed successfully!")
        else:
            print("❌ Some tests failed - check report for details")
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Failed to run testing agent: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())