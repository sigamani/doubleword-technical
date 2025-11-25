#!/usr/bin/env python3
"""
Agent launcher for sigamani/doubleword-technical repository
Provides easy access to available agents via /agents command
"""

import os
import sys
import subprocess
import argparse
import re
from pathlib import Path

def check_and_remove_emojis(file_path: str) -> bool:
    """
    Strict emoji check using ruff to detect and remove emojis from source code
    Returns True if emojis were found and removed, False otherwise
    """
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Define emoji pattern (comprehensive)
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U00002702-\U000027B0\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]'
        )
        
        # Find all emojis
        emojis_found = emoji_pattern.findall(content)
        
        if emojis_found:
            print(f"EMOJI VIOLATION DETECTED: Found {len(emojis_found)} emojis in {file_path}")
            print("Removing emojis to comply with AGENTS.md cardinal rule...")
            
            # Remove all emojis
            clean_content = emoji_pattern.sub('', content)
            
            # Write back clean content
            with open(file_path, 'w') as f:
                f.write(clean_content)
            
            print(f"Successfully removed {len(emojis_found)} emojis from {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error checking emojis in {file_path}: {e}")
        return False

def run_ruff_emoji_check() -> bool:
    """
    Use ruff to check for emoji violations in the current file
    """
    try:
        # Run ruff on this file to check for any issues
        result = subprocess.run(
            ["ruff", "check", "--select=ALL", __file__],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("Ruff found issues in agents.py")
            print(result.stdout)
            return False
        
        print("Ruff check passed - no issues found")
        return True
        
    except FileNotFoundError:
        print("Ruff not available - skipping ruff check")
        return True
    except Exception as e:
        print(f"Error running ruff check: {e}")
        return False

# Perform strict emoji check on startup
if __name__ != "__main__":
    # Only run checks when imported as module, not when executed directly
    current_file = __file__
    if check_and_remove_emojis(current_file):
        print("EMOJIS REMOVED - Restarting without emojis")
        # Exit and let the process restart with clean code
        sys.exit(1)
    
    # Run ruff check
    run_ruff_emoji_check()

def run_testing_symbiote():
    """Launch Testing Symbiote agent"""
    print("Launching Testing Symbiote...")
    print("=" * 50)
    
    if not os.path.exists("testing_symbiote.py"):
        print("ERROR: testing_symbiote.py not found!")
        return 1
    
    cmd = [sys.executable, "testing_symbiote.py", "--verbose"]
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode
    except KeyboardInterrupt:
        print("\nWARNING: Testing Symbiote interrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: Error running Testing Symbiote: {e}")
        return 1

def run_testing_agent():
    """Launch standard Testing Agent"""
    print("Launching Testing Agent...")
    print("=" * 50)
    
    if not os.path.exists("testing_agent.py"):
        print("ERROR: testing_agent.py not found!")
        return 1
    
    cmd = [sys.executable, "testing_agent.py", "--verbose"]
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode
    except KeyboardInterrupt:
        print("\nWARNING: Testing Agent interrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: Error running Testing Agent: {e}")
        return 1

def list_available_agents():
    """List all available agents"""
    print("Available Agents for sigamani/doubleword-technical:")
    print("=" * 60)
    
    agents = []
    
    if os.path.exists("testing_symbiote.py"):
        agents.append({
            "name": "testing-symbiote", 
            "description": "Symbiotic testing agent - runs full test matrix and repairs failures",
            "usage": "/agents testing-symbiote"
        })
    
    if os.path.exists("testing_agent.py"):
        agents.append({
            "name": "testing-agent",
            "description": "Standard testing agent - comprehensive repository testing", 
            "usage": "/agents testing-agent"
        })
    
    if os.path.exists("run_tests.py"):
        agents.append({
            "name": "run-tests",
            "description": "Quick test runner - basic repository validation",
            "usage": "/agents run-tests"
        })
    
    for agent in agents:
        print(f"- {agent['name']}")
        print(f"  Description: {agent['description']}")
        print(f"  Usage: {agent['usage']}")
        print()
    
    if not agents:
        print("ERROR: No agents found in current repository")
        return 1
    
    return 0

def main():
    """Main agent launcher entry point"""
    parser = argparse.ArgumentParser(
        description="Agent launcher for sigamani/doubleword-technical repository",
        add_help=False
    )
    
    parser.add_argument(
        "agent_name", 
        nargs="?", 
        help="Agent name to launch (testing-symbiote, testing-agent, run-tests, or 'list' to see all)"
    )
    
    # Parse known args to allow for agent-specific arguments
    args, unknown = parser.parse_known_args()
    
    # Check for no agent name first
    if not args.agent_name:
        print("Agent Launcher for sigamani/doubleword-technical")
        print("=" * 60)
        print("Usage: /agents <agent_name> [agent_options]")
        print()
        print("Available agents:")
        print("  testing-symbiote  - Launch Testing Symbiote (full test matrix + repairs)")
        print("  testing-agent    - Launch standard Testing Agent")  
        print("  run-tests        - Run quick test validation")
        print("  list             - List all available agents")
        print()
        print("Examples:")
        print("  /agents testing-symbiote")
        print("  /agents testing-agent")
        print("  /agents list")
        print()
        print("Use '/agents <agent_name> --help' for agent-specific options")
        return 0
    
    agent_name = args.agent_name.lower()
    
    if agent_name in ["list", "ls", "show"]:
        return list_available_agents()
    elif agent_name in ["testing-symbiote", "symbiote", "ts"]:
        # Check if help is requested for this specific agent
        if "--help" in unknown or "-h" in unknown:
            # Run testing symbiote with its help
            cmd = [sys.executable, "testing_symbiote.py", "--help"]
        else:
            # Pass through unknown arguments to testing symbiote
            cmd = [sys.executable, "testing_symbiote.py"] + unknown
        
        try:
            result = subprocess.run(cmd, cwd=os.getcwd())
            return result.returncode
        except KeyboardInterrupt:
            print("\n️ Testing Symbiote interrupted by user")
            return 130
        except Exception as e:
            print(f" Error running Testing Symbiote: {e}")
            return 1
    elif agent_name in ["testing-agent", "testing", "ta"]:
        # Pass through unknown arguments to testing agent
        cmd = [sys.executable, "testing_agent.py"] + unknown
        try:
            print(f"Launching Testing Agent with args: {' '.join(unknown)}")
            result = subprocess.run(cmd, cwd=os.getcwd())
            return result.returncode
        except KeyboardInterrupt:
            print("\n️ Testing Agent interrupted by user")
            return 130
        except Exception as e:
            print(f" Error running Testing Agent: {e}")
            return 1
    elif agent_name in ["run-tests", "run", "rt"]:
        # Run the existing run_tests.py
        if os.path.exists("run_tests.py"):
            try:
                result = subprocess.run([sys.executable, "run_tests.py"], cwd=os.getcwd())
                return result.returncode
            except Exception as e:
                print(f" Error running tests: {e}")
                return 1
        else:
            print(" run_tests.py not found!")
            return 1
    else:
        print(f" Unknown agent: {agent_name}")
        print("Use '/agents list' to see available agents")
        return 1

if __name__ == "__main__":
    sys.exit(main())