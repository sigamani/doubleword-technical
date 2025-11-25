# Agent Launcher

Quick access to testing agents for sigamani/doubleword-technical repository.

## Usage

```bash
# List all available agents
/agents list

# Launch Testing Symbiote (full test matrix + repairs)
/agents testing-symbiote

# Launch Testing Symbiote with options
/agents testing-symbiote --verbose --report-file custom_report.json

# Launch standard Testing Agent
/agents testing-agent

# Quick test run
/agents run-tests

# Get help for specific agent
/agents testing-symbiote --help
```

## Available Agents

### 🧫 Testing Symbiote
- **Purpose**: Runs complete test matrix (batch size × concurrency × model-size permutations) and repairs failures
- **Features**: 
  - 10 different test configurations
  - Automated failure diagnosis and repair
  - Comprehensive performance reporting
  - SLA validation
- **Best for**: Full repository validation after changes

### 🔧 Testing Agent  
- **Purpose**: Comprehensive repository testing and issue resolution
- **Features**:
  - Syntax and dependency checks
  - Unit and integration tests
  - Docker build validation
  - Automated fixes
- **Best for**: General repository health checks

### ⚡ Run Tests
- **Purpose**: Quick repository validation
- **Features**: Fast test execution for basic checks
- **Best for**: Rapid validation during development

## Integration with Opencode

The Testing Symbiote is also configured as a subagent in opencode.json and can be invoked via:

```python
task(
    description="Run symbiote test matrix",
    prompt="Execute Testing Symbiote for comprehensive test matrix execution and repair",
    subagent_type="testing-symbiote"
)
```

This enables both command-line access via `/agents` and programmatic access via the task system.