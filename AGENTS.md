# Agent Guide

## Environment Setup

**CRITICAL**: All work happens on a rented Ubuntu VM from vast.ai, NOT locally.

1. **Request SSH Access**: Ask the human for SSH credentials to access the VM
2. **Verify Docker**: Run `docker ps` to check running containers
3. **Pull Image if Needed**: If image not present, run `docker pull michaelsigamani/proj-grounded-telescopes`
4. **Container Rule**: ALL code execution MUST happen inside the `michaelsigamani/proj-grounded-telescopes` container unless explicitly told otherwise

---

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
```

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

