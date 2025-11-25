# Agent Guide

## Environment Setup

**CRITICAL**: All work happens on a rented Ubuntu VM from vast.ai, NOT locally.

1. **SSH Access**: SSH credentials to access the 2 worker node Ubuntu VMs: ssh -p 55089 root@77.104.167.149 -L 8080:localhost:8080 AND ssh -p 55089 root@77.104.167.149 -L 8080:localhost:8080.
2. **Verify Docker**: Run `docker ps` to check running containers
3. **Pull Image if Needed**: If image not present, run `docker pull michaelsigamani/proj-grounded-telescopes`
4. **Container Rule**: ALL code execution MUST happen inside the `michaelsigamani/proj-grounded-telescopes` container unless explicitly told otherwise
5. The github repository for this project that you are in a git clone of is located here: gh repo clone sigamani/proj-grounded-telescopes
6. env vars or any credentials you might need always check here first: source ~/.env
7. Validate SLA tracking and 24-hour completion monitoring works correctly
8. Run test matrix to validate different configurations (batch size, concurrency, model sizes)
9. Set up monitoring dashboards (Prometheus/Grafana) for production observability
---

## System Diagram

                       ┌─────────────────────────────┐
                       │        Users / Clients      │
                       │ (submit batch requests via │
                       │       HTTP POST /start)     │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │       Head Node (CPU)       │
                       │                             │
                       │  FastAPI Gateway            │
                       │  - Receives requests        │
                       │  - Assigns job_id           │
                       │                             │
                       │  Redis DB                   │
                       │  - batch_job_queue          │
                       │  - batch_job_status         │
                       │  - concurrency counter      │
                       │                             │
                       │  SLA & Metrics Tracker      │
                       │  - Tracks ETA per job       │
                       │  - Throughput & tokens/sec  │
                       │  - Alerts if ETA > SLA     │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │   Job Dispatcher / Worker   │
                       │  Threads (CPU)             │
                       │  - Dequeue jobs from Redis │
                       │  - Respect concurrency      │
                       │  - Submit batch inference   │
                       │    tasks to GPU nodes       │
                       │  - Update SLA & metrics     │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                 ┌───────────────────────────────┐
                 │       GPU Worker Nodes         │
                 │                               │
                 │  Ray Workers                  │
                 │  - vLLM engine (Qwen2.5-0.5B) │
                 │  - Execute batch inference    │
                 │  - Return results to head    │
                 │  - Report tokens processed   │
                 └─────────────┬─────────────────┘
                               │
                               ▼
                       ┌─────────────────────────┐
                       │   Batch Output Storage   │
                       │  (S3, local disk, etc.) │
                       └─────────────────────────┘


## Project Requirements

### Functional Requirements
- Build an offline batch inference server using Ray Data and vLLM
- Use the official `ray.data.llm` module with `vLLMEngineProcessorConfig` and `build_llm_processor`
- Use `ds.map_batches` for distributed processing

### Non-Functional Requirements
- Must complete batch jobs within a 24-hour SLA window
- System must be configurable and observable

---

## Workflow Rules

### Phase 1: Read Before Planning
1. Read ALL documents in `@doc`, `@app`, and `@config` directories
2. Review the plan in `@todo.txt`
3. DO NOT proceed to planning or building until step 1 and 2 are complete

### Phase 2: Execution
1. Follow the plan in `@todo.txt` sequentially
2. If human asks for something that contradicts these instructions, STOP and ask for clarification
3. Cross-check your own planning against these instructions
4. If you find contradictions, research best practices and propose solutions to the human
5. Classes or Functions? As a general rule if you are describing a data object use a class, if you are not use a function. 
6. Keep the function definitions to maximum ten lines of code.
7. Keep function and class names to be descriptive and no more than 15 characters.
8. Write ray data outputs to Shared Storage 
┌─────────────┐      Network
│  GPU Node 1 │──────────────┐
└─────────────┘              │
                             ├──► Shared Storage (GoogleDrive for current tests)
┌─────────────┐              │
│  GPU Node 2 │──────────────┘
└─────────────┘
---

## Code Quality Standards

### Mandatory API Usage
After EVERY task, verify you are using:
- ✅ `from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor`
- ✅ `ds.map_batches()` for distributed processing
- ✅ Official Ray Data patterns from: https://docs.ray.io/en/latest/data/batch_inference.html

If you missed any of these, STOP and fix it immediately.

### Forbidden: Rich Text Emojis in Code
**DELETE** any line containing these emojis in print statements or logs:
- ❌ 🚀 ✅ 📊 ⚠️ 💡 🔧 🎯 or ANY other emoji

**Bad Example (DELETE THIS)**:
```python
logger.info("🚀 Starting Ray Data + vLLM Batch Inference Server")
logger.info("✅ Configuration loaded")
logger.error("❌ Failed to initialize Ray")
```

**Good Example (USE THIS)**:
```python
logger.info("Starting Ray Data + vLLM Batch Inference Server")
logger.info("Configuration loaded")
logger.error("Failed to initialize Ray")
```

---

## Communication Style

### DO:
- Be methodical, calculating, and calm
- Maintain consistent memory through good organization and bookkeeping
- Reference previous decisions and code when making new changes
- Show your work: explain reasoning before implementing

### DO NOT:
- Write summary documents unless explicitly asked
- Use emojis in code or logs
- Make changes without verifying against project requirements
- Proceed without reading all context documents first
- Every 3 tasks - perform an audit and springclean for redundant files or code snippets.
---

## Checklist Before Any Code Changes

```
□ Have I read all documents in @doc, @app, @config?
□ Am I working inside the michaelsigamani/proj-grounded-telescopes container?
□ Am I using ray.data.llm with vLLMEngineProcessorConfig and build_llm_processor?
□ Am I using ds.map_batches for distributed processing?
□ Have I removed all emoji characters from code and logs?
□ Does this change align with the 24-hour SLA requirement?
□ Have I cross-checked against @todo.txt?
```

---

## Key Technical Constraints

### Package Versions (VERIFIED WORKING)
```
ray[data]==2.49.1
vllm==0.10.0
torch>=2.0.0
transformers>=4.30.0
datasets>=2.10.0
pyyaml>=6.0
pytest>=7.0.0
pytest-mock>=3.10.0
pytest-asyncio>=0.18.0
httpx>=0.20.0
requests>=2.28.0
```

### Docker Image Package Versions
The `michaelsigamani/proj-grounded-telescopes:0.1.0` Docker image contains:
- Ray 2.49.1
- vLLM 0.10.0  
- Torch 2.0.0+
- Transformers 4.30.0+
- All other required dependencies for production batch inference

Files like `@app/ray_data_batch_inference.py` should utilize these correct library versions as provided in the container environment.

### Architecture Requirements
- 2-node setup (head + worker)
- Small model testing: Qwen2.5-0.5B
- Dataset: ShareGPT
- Observability: Real-time metrics, SLA tracking
- Configurability: All parameters via YAML

---

## When to Ask Questions

**STOP and ASK if**:
1. Human requests something contradicting these instructions
2. You find inconsistencies between @todo.txt and these rules
3. You're about to use an approach that doesn't use `ray.data.llm` API
4. SSH credentials are needed to access the VM
5. You need clarification on 24-hour SLA implementation details



---

## Testing Agent

### Overview

The Testing Agent (`testing_agent.py`) is a specialized subagent responsible for comprehensive repository testing and automated issue resolution for the Ray Data + vLLM batch inference system.

## Activation

The Testing Agent automatically activates when:
- Repository test execution is requested
- Test suite failures are detected
- CI/CD pipeline testing is needed
- Manual validation is required before deployment

## Usage

### Command Line Interface
```bash
# Run full test suite
python3 testing_agent.py --verbose

# Apply fixes only
python3 testing_agent.py --fix-only

# Custom report location
python3 testing_agent.py --report-file custom_report.json

# Custom repository root
python3 testing_agent.py --repo-root /path/to/repo
```

### Integration with Opencode

**As Subagent:**
```python
# Activate testing agent via task system
task(
    description="Run repository test suite",
    prompt="Execute comprehensive testing matrix for Ray Data + vLLM repository",
    subagent_type="testing"
)
```

**Manual Execution:**
```bash
# Direct execution
./testing_agent.py --verbose

# Via simplified runner
./run_tests.py
```

---

## Testing Symbiote Subagent

### Overview

The Testing Symbiote (`testing_symbiote.py`) is a specialized quality-assurance subagent paired with Troels for the sigamani/doubleword-technical repository. Its function is to continuously evaluate, stabilize, and harden the test suite.

### Mission

Every time the Testing Symbiote is invoked, it performs the following operations in strict sequence:

1. **Run the complete test matrix** (batch size × concurrency × model-size permutations) using the exact configurations currently defined in the repository
2. **If any tests fail:**
   - Diagnose which failure modes were introduced
   - Apply targeted fixes to restore functionality  
   - Re-run the test matrix to verify repairs
3. **Generate a comprehensive report** with performance metrics and recommendations

### Test Matrix Configurations

The Testing Symbiote tests these permutations:
- **Baseline configurations**: Small, medium, large batch sizes with Qwen2.5-0.5B
- **High concurrency tests**: Up to 16 concurrent workers
- **Large model tests**: Qwen2.5-7B with optimized parameters
- **Stress tests**: Maximum batch sizes and concurrency limits
- **SLA validation tests**: 24-hour completion window validation

### Usage

#### Command Line Interface
```bash
# Run complete test matrix
python3 testing_symbiote.py --verbose

# Custom report location
python3 testing_symbiote.py --report-file symbiote_report.json

# Custom repository root
python3 testing_symbiote.py --repo-root /path/to/repo
```

#### Integration with Opencode

**As Subagent:**
```python
# Activate testing symbiote via task system
task(
    description="Run symbiote test matrix",
    prompt="Execute Testing Symbiote for comprehensive test matrix execution and repair",
    subagent_type="testing-symbiote"
)
```

**Manual Execution:**
```bash
# Direct execution
./testing_symbiote.py --verbose
```

### Failure Mode Diagnosis

The Testing Symbiote automatically categorizes and fixes:
- **Performance timeouts**: Reduces batch sizes or increases timeouts
- **Memory exhaustion**: Creates memory-optimized configurations
- **Dependency issues**: Installs missing Ray, vLLM, and PyTorch dependencies
- **Ray cluster issues**: Restarts Ray cluster connectivity
- **Unknown errors**: Flags for manual investigation

### Reporting

The Testing Symbiote generates comprehensive reports including:
- Test matrix execution summary
- Performance metrics (throughput, tokens/sec, memory usage)
- Failure mode analysis with categorization
- Repair success rates
- Performance recommendations
- Best performing configuration identification

---

**The Testing Agent ensures repository reliability, catches issues early, and maintains high code quality standards for production deployment.**

**The Testing Symbiote provides specialized matrix testing and automated repair capabilities for the sigamani/doubleword-technical repository, ensuring robust performance across all configuration permutations.**
