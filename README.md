# Offline Batch Inference PoC (OpenAI-Style)

### Ray Data + vLLM + FastAPI + In-Memory Queue

## Overview

This Proof-of-Concept demonstrates an **OpenAI-style offline batch inference system** designed to validate core architectural patterns for request marshalling, job lifecycle management, and compute resource allocation. The system uses minimal dependencies while remaining aligned with industry best practices, allowing the API semantics, batching workflow, and scheduling logic to be validated without requiring actual GPU infrastructure or distributed systems.

The PoC uses:

* **Ray Data** for batch ingestion and parallel map-style execution
* **vLLM** as the LLM execution engine
* **FastAPI** for the control plane
* **`collections.deque`** as an in-memory job queue
* **Docker + Docker Compose** for isolated, reproducible local deployment
* **Mocked resource pools** simulating spot and dedicated GPU allocation

Production-grade substitutes are outlined for each component.

---
<details>
<summary><strong> 1. Research Summary </strong></summary>

<br>

# Current Trends in Offline/Batch LLM Inference

<img width="2400" height="1600" alt="Trends_in_engines,_servers,_frameworks,_and_patterns_for_offline_and_batch_LLM_inference_in_recent_public_GitHub_projects" src="https://github.com/user-attachments/assets/6649c5b4-eacb-405a-b577-66a9fc2fd03f" />

Derived from analysis of public GitHub repositories and industry examples [See Key References](#9-key-references)

## 1.1 Engines

Recent open-source repositories cluster around:

* **vLLM** – high-throughput Python-native serving; common for PoCs and lightweight deployments
* **TensorRT-LLM** – GPU-optimized, often used with Triton; preferred in production environments
* **HuggingFace TGI** – standardized server with continuous batching & token limits
* **llama.cpp** – CPU/edge-optimized; sometimes used for PoCs or offline evaluation

## 1.2 Model Servers / Serving Layers

Typical choices:

* **Triton Inference Server** (TensorRT-LLM backend) for GPU batching, scheduling, inflight batching
* **HuggingFace TGI** for text-generation-focused deployments
* **vLLM's Python server** for simpler control-plane integration
* **Custom FastAPI/gRPC control planes** when queueing, task lifecycle, or custom semantics are needed

## 1.3 Orchestration Patterns

Two patterns dominate:

* **Queue + Worker model** – FastAPI control plane + background workers performing batch inference
* **Scheduler + Cluster runtime** – Ray, Kubernetes, or Slurm scheduling batch jobs over GPU pools

## 1.4 Common Characteristics of Offline/Batch Inference Repos

* Use of **map-style batch transforms** (Ray, Spark, TGI batch API, TensorRT-LLM batch scheduler)
* OpenAI-style patterns around:
  * upload file  
  * create batch job  
  * poll job status  
  * retrieve results  
* Dynamic batching or continuous batching where possible
* Separation between **control plane** (HTTP API) and **execution layer** (Ray/vLLM/Triton)
* PoCs avoid Redis, Celery, Kafka, etc.; they use local queues or in-memory runners

This PoC aligns with these trends.

</details>

---

<details>
<summary><strong> 2. Architecture of This PoC </strong></summary>

<br>

```
             ┌─────────────────────────────┐
             │          FastAPI            │
             │  (HTTP control plane)       │
             │  • submit batch job         │
             │  • poll job status          │
             │  • retrieve results         │
             │  • SLA metadata tracking    │
             └──────────────┬──────────────┘
                            │
                     enqueue/dequeue
                            │
                  ┌─────────▼─────────┐
                  │   In-Memory Queue │
                  │   (collections.deque)
                  │  + job metadata    │
                  └─────────┬─────────┘
                            │
                background worker thread
                  (with mocked scheduler)
                            │
                ┌───────────▼───────────┐
                │   Mocked GPU Pool     │
                │  • spot instances     │
                │  • dedicated instances│
                │  • capacity tracking  │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │        Ray Data        │
                │   map_batches pipeline │
                └───────────┬───────────┘
                            │
                   calls into vLLM
                            │
                ┌───────────▼──────────┐
                │      vLLM Engine      │
                │   (Qwen2.5 0.5B/7B)   │
                └────────────────────────┘
```

## Getting Started

```bash
# Clone repository
git clone https://github.com/sigamani/PoC-offline-batch-inference.git
cd PoC-offline-batch-inference

# Start all services
docker-compose up

# Submit a batch job (example)
curl -X POST http://localhost:8000/v1/batches \
  -H "Content-Type: application/json" \
  -d @examples/sample_batch.jsonl

# Check job status
curl http://localhost:8000/v1/batches/{batch_id}
```
</details>

---

<details>
<summary><strong> 3. Component Choices and Rationale </strong></summary>

<br>

## 3.1 Ray Data

* Native map-style **batch transformations** match the OpenAI offline inference model
* Built-in backpressure, parallelism, and scaling semantics
* Lightweight setup for a PoC without needing a Ray cluster

## 3.2 vLLM

* High throughput per GPU/CPU
* Simple Python API; minimal server overhead
* Aligns with current industry trends in PoCs and production prototypes

## 3.3 FastAPI

* Thin control plane for:
  * job submission with SLA metadata
  * queue inspection
  * job results retrieval
* Easiest to test and iterate on
* Used in many open-source LLM servers for HTTP orchestration

## 3.4 `collections.deque` (Queue)

* Zero external dependencies
* Allows rapid prototyping and unit testing of queue semantics
* Mirrors queue/worker pattern used in production without requiring Redis/Celery

**Production substitute:** Redis Streams, Redis Queue, or Celery with a broker.

## 3.5 Mocked GPU Pool Scheduler

* Simulates resource allocation decisions without actual GPU hardware
* Tests request marshalling logic for spot vs dedicated instance assignment
* Includes capacity tracking and fallback behavior
* Example state representation:

```python
gpu_pool = {
    "spot": {"capacity": 2, "available": 1},
    "dedicated": {"capacity": 1, "available": 1},
}
```

**Production substitute:** Real GPU scheduling with Ray autoscaler, Kubernetes, or cloud provider APIs.

## 3.6 Docker + Docker Compose

* Provides reproducibility for independent LLM engine, Ray runtime, and API
* Keeps the PoC self-contained and portable

**Production substitute:** Kubernetes, KubeRay, Ray Jobs API, or ECS.

</details>

---

<details>
<summary><strong> 4. Scope Definition </strong></summary>

<br>
 
## 4.1 In Scope (for PoC)

**Core API & Lifecycle:**
* OpenAI-style batch job API (submit → status → results)
* FastAPI control plane
* Single-node Ray Data pipeline
* vLLM running Qwen2.5-0.5B or Qwen2.5-7B
* In-memory queue backed by `deque`
* Simple job lifecycle: queued → running → completed/failed

**SLA Management (Mocked):**
* Job metadata includes `submitted_at` and `deadline_at` (submitted_at + 24h)
* Worker logic includes SLA-aware scheduling stubs
* Logging/metrics expose SLA compliance markers
* No real-time SLA enforcement or violation remediation

**Compute Pool Scheduling (Mocked):**
* Mock scheduler assigns jobs to spot or dedicated instance pools
* Simulated resource pool state with capacity tracking
* Fallback logic when spot capacity is unavailable
* Worker logs which pool was assigned to each job

**Infrastructure:**
* Docker-based reproducible environment
* Full runnable setup with Docker Compose
* No external service dependencies
* Clear repository layout and documentation

**Observability:**
* Metrics hooks or stubs (latency, throughput)
* Basic logging of scheduling decisions

## 4.2 Out of Scope (for PoC)

* Distributed multi-node Ray cluster
* Real GPU scheduling, placement, or hardware management
* Autoscaling based on load or spot availability
* Redis/Celery-based production queues
* Kafka or other message brokers
* Triton Inference Server integration
* TensorRT-LLM backends
* GPU multi-tenant provisioning
* Advanced scheduling (priority tiers, token-level batching, inflight batching)
* Real SLA violation handling or remediation workflows
* Enterprise authentication, authorization, audit logging
* Cost optimization strategies
* Multi-region deployment
* Real object storage integration (S3/GCS)
* Priority queues or tiered service plans

</details>

---

<details>
<summary><strong> 1. 5. How This PoC Aligns With Current Trends </strong></summary>

<br> 

This PoC mirrors the architecture patterns observed in modern repositories:

| Trend (GitHub)                          | PoC Alignment                          |
| --------------------------------------- | -------------------------------------- |
| FastAPI control plane                   | Yes                                    |
| Queue + Worker batching model           | Yes (in-memory queue)                  |
| vLLM or TGI as engine                   | vLLM                                   |
| Ray/TensorRT-LLM/Triton for production  | Ray Data in PoC, Triton noted for prod |
| OpenAI-style job lifecycle              | Yes                                    |
| SLA-aware job metadata                  | Yes (mocked)                           |
| Resource pool scheduling                | Yes (mocked spot/dedicated)            |
| Minimal PoC infra (no Redis/Kafka)      | Yes                                    |
| Dockerized dev environment              | Yes                                    |
| Replaceable components for staging/prod | Yes                                    |

</details>

---

<details>
<summary><strong> 6. Proposed Production Path </strong></summary>

<br>

If taken beyond PoC:

| PoC Component              | Production Upgrade                      |
| -------------------------- | --------------------------------------- |
| `deque` queue              | Redis Streams or Celery                 |
| Ray Data (local)           | Ray cluster, Ray Jobs, autoscaling      |
| vLLM Python runtime        | vLLM server or Triton backend           |
| Mocked GPU pool            | Real GPU scheduler with cloud provider  |
| Mocked SLA tracking        | Real-time SLA monitoring & enforcement  |
| FastAPI                    | gRPC or API Gateway front door          |
| Docker Compose             | Kubernetes or KubeRay                   |
| Local storage              | S3/GCS object storage                   |
| In-memory job metadata     | Database (PostgreSQL/DynamoDB)          |

</details>

---

<details>
<summary><strong> 7. Repository Structure </strong></summary>

<br>

```
repo/
│
├── api/
│   ├── main.py              # FastAPI endpoints
│   ├── models.py            # Pydantic models, job metadata
│   ├── queue.py             # In-memory queue + worker
│   └── scheduler.py         # Mocked GPU pool scheduler
│
├── engine/
│   ├── vllm_runner.py       # vLLM interface
│   └── model_config.yaml    # Model configuration
│
├── pipeline/
│   └── ray_batch.py         # Ray Data batch pipeline
│
├── docker/
│   ├── Dockerfile.api       # API service container
│   ├── Dockerfile.vllm      # vLLM engine container
│   └── docker-compose.yml   # Complete environment setup
│
├── examples/
│   ├── sample_batch.jsonl   # Example batch input
│   └── client_submit.ipynb  # Example client usage
│
├── tests/
│   ├── test_api.py
│   ├── test_scheduler.py
│   └── test_pipeline.py
│
└── README.md
```

</details>
---

<details>
<summary><strong> 8. Key Deliverables </strong></summary>

<br>

This PoC validates:

1. **Request marshalling patterns** - How batch jobs flow from API to execution
2. **Resource allocation logic** - Mocked scheduling between spot and dedicated pools
3. **SLA-aware job metadata** - Tracking deadlines and compliance markers
4. **OpenAI API compatibility** - Standard batch inference interface
5. **Component integration** - FastAPI + Ray Data + vLLM working together
6. **Deployment simplicity** - Single Docker Compose command to run

The PoC does **not** validate:

* Real GPU performance characteristics
* Production-scale throughput
* Cost optimization strategies
* Enterprise security requirements
* Multi-tenant isolation

</details>

---

<details>
<summary><strong> 9. Key References </strong></summary>

<br>

* GitHub repositories surveyed for offline/batch LLM inference (vLLM, TGI, TensorRT-LLM, custom pipelines)
* Summary research output from Perplexity PDF
* OpenAI Batch API documentation patterns

## References

- NVIDIA. “The Triton TensorRT-LLM Backend.” https://github.com/triton-inference-server/tensorrtllm_backend
- NVIDIA. “Dynamic Batcher - NVIDIA Triton Inference Server.” https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
- vLLM Project. “performance and concurrency questions · Issue #2308.” https://github.com/vllm-project/vllm/issues/2308
- NVIDIA. “Is there any plan to open source Inflight Batching for LLM Serving?” https://github.com/triton-inference-server/server/issues/6358
- Hugging Face. “Large Language Model Text Generation Inference on Habana Gaudi.” https://github.com/huggingface/tgi-gaudi
- NVIDIA. “Optimizing Inference on Large Language Models with TensorRT-LLM.” https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/
- NVIDIA. “inflight_batcher_llm example batching · Issue #558.” https://github.com/triton-inference-server/tensorrtllm_backend/issues/558
- Hannibal046. “Awesome-LLM: a curated list of Large Language Model.” https://github.com/Hannibal046/Awesome-LLM
- NVIDIA. “Overview — TensorRT-LLM Performance.” https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html
- NVIDIA. “triton-inference-server/backend.” https://github.com/triton-inference-server/backend
- Hugging Face. “Rules of thumb for setting max-batch-total-tokens · Issue #629.” https://github.com/huggingface/text-generation-inference/issues/629
- NVIDIA. “NVIDIA/TensorRT-LLM.” https://github.com/NVIDIA/TensorRT-LLM
- NVIDIA. “Expected batch dimension to be 1 for each request for input_ids · Issue #319.” https://github.com/triton-inference-server/tensorrtllm_backend/issues/319
- Hugging Face. “Text Generation Inference Documentation.” https://huggingface.co/docs/text-generation-inference/en/index
- Ray Project. “[data.llm] Support TensorRT-LLM offline inference with Ray.” https://github.com/ray-project/ray/issues/56989
- Hugging Face. “Large Language Model Text Generation Inference.” https://github.com/huggingface/text-generation-inference
- simonorzel26. “openai-batch-awaiter.” https://github.com/simonorzel26/openai-batch-awaiter
- miko-ai-org. “llmbatching: An openAI / LLM API wrapper.” https://github.com/miko-ai-org/llmbatching
- EasyLLM. “LangBatch.” https://github.com/EasyLLM/langbatch
- Microsoft. “Azure OpenAI Batch API Accelerator.” https://github.com/Azure-Samples/aoai-batch-api-accelerator
- SpellcraftAI. “oaib: Use the OpenAI Batch tool to make async requests.” https://github.com/SpellcraftAI/oaib
- Amazon Web Services. “Submit a batch of prompts with the OpenAI Batch API.” https://docs.aws.amazon.com/bedrock/latest/userguide/inference-openai-batch.html
- Koray Gocmen. “scheduler-worker-grpc.” https://github.com/KorayGocmen/scheduler-worker-grpc
- OpenAI. “Batch API – OpenAI Platform.” https://platform.openai.com/docs/guides/batch
- GitHub. “dynamic-batching · GitHub Topics.” https://github.com/topics/dynamic-batching
- Guy Gregory. “Azure OpenAI - Structured Outputs & Batch API.” https://github.com/guygregory/StructuredBatch
- iPieter. “llmq: A Scheduler for Batched LLM Inference.” https://github.com/iPieter/llmq
- zheqiaochen. “OpenAI Batch Tools.” https://github.com/zheqiaochen/openaibatch
- OpenAI. “Batch processing with the Batch API.” https://cookbook.openai.com/examples/batch_processing
- EthicalML. “awesome-production-machine-learning.” https://ethicalml.github.io/awesome-production-machine-learning/
- danvk. “gpt-batch-manager: Tools for splitting jobs across batches.” https://github.com/danvk/gpt-batch-manager
- alexrudall. “ruby-openai: OpenAI API + Ruby.” https://github.com/alexrudall/ruby-openai

</details>

