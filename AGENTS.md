# Agent Guide

## Environment Setup

**CRITICAL**: All work happens on a rented Ubuntu VM from vast.ai, NOT locally.

1
2. **Verify Docker**: Run `docker ps` to check running containers
3. **Build and run lightweight dev Image**: Dockerfile.dev this is made to be lightweight for faster iteration and testing in github codespace. Use this for development and testing before moving to the full image. No GPU acceleration. 
   ```bash
   docker build -f Dockerfile.dev -t proj-grounded-telescopes-dev .
   docker run -it --rm -v $(pwd):/app proj-grounded-telescopes-dev /bin/bash
   ```
4. **Container Rule**: Do not install python packgages outside of the docker container unless needed to keep the space light. We only have 30 GiG to test. All packages will be build inside the Dockerfile.dev container.
5. The github repository for this project that you are in a git clone of is located here: [gh repo clone sigamani/proj-grounded-telescopes](https://github.com/sigamani/doubleword-technical)

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
2. Review the plan in PLAN (below) and check what has been implemented from that in the code base (test if necessary) and then move to either fixing gaps or moving to the next step

### Phase 2: Execution
2. If human asks for something that contradicts any of the the previous ways of working or instructions that do not follow good AI development practice, or instructions that will probably derail the main plan then STOP and ask for clarification and confirmation before proceeding. You are doing the user a favour by calling out bad ideas.
3. Cross-check your own planning against these instructions
5. Classes or Functions? As a general rule if you are describing a data object use a class, if you are not use a function. 
6. Keep the function definitions to maximum ten lines of code.
7. Keep function names / class names no more than 15 characters. Do not comment in code, use clear names instead.


---

## Code Quality Standards

### Mandatory API Usage
After EVERY task, verify you are using:
- ✅ `from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor`
- ✅ `ds.map_batches()` for distributed processing
- ✅ Official Ray Data patterns from: https://docs.ray.io/en/latest/data/batch_inference.html

If you missed any of these, STOP and fix it immediately.

### Forbidden: DO NOT EVER USE Rich Text Emojis in Code
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
- Write .md summaries or documentation EVER.
- Use emojis in code or logs
- Make changes without verifying against project requirements
- Proceed without reading all context documents first
- Every 3 tasks - perform an audit and springclean for redundant files or code snippets - use the symbiote if needed.
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
r

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

### PLAN

1. Authentication: issue a bearer token and inject it into the user’s curl request payload.

3. Data serialization: use SHA-based identifiers so all data is immutable, and store artifacts in a versioned store such as S3.

4. Logging: for the proof-of-concept I will write logs and metrics to a local output directory, but I will note production options such as Loki and Promtail.

5. Dashboarding: most teams use Grafana; for the proof-of-concept I will expose metrics as a local JSON file for simplicity.

6. Main Dependencies for PoC: Ray 2.4.9, vLLM 0.10.0, PyTorch, Hugging Face Hub, Transformers, Redis, Prometheus, FastAPI, Uvicorn, and Pydantic.