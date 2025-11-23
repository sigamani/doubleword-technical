# Ray + vLLM Multi-Node Batch Inference System

## 🚀 IMPORTANT: Ray Data Batch Inference

This project uses **Ray Data** for distributed batch inference, following to official documentation:
https://docs.ray.io/en/latest/data/batch_inference.html

### Key Pattern:
```python
# Ray Data map_batches for distributed vLLM inference
ds.map_batches(
    vllm_inference,
    num_gpus=1,  # Each actor gets 1 GPU
    concurrency=2,  # 2 parallel actors (one per node)
    batch_size=4   # Optimize batch size for GPU utilization
)
```

This approach provides:
- **Intelligent GPU orchestration** across nodes
- **Automatic load balancing** via Ray scheduler
- **Scalable batch processing** with optimal resource utilization
- **Fault tolerance** with automatic actor recovery

PoC batch inference stack for running small language models across multiple GPU
workers with monitoring, load balancing, and a FastAPI front-end.

## Overview
- **Inference API:** `app/inference_api_server.py` serves single and batch text
  generation, exposes Prometheus metrics on port 8001, and powers the
  client-facing REST interface.
- **Client CLI:** `app/llm_client.py` exercises the API for health checks, single
  prompts, batch prompts, or reading prompts from a file.
- **Monitoring:** `config/docker-compose-monitoring.yml` brings up Prometheus,
  Grafana, and node exporter dashboards for latency, request volume, and GPU
  utilization.
- **Ray Cluster Utilities:** `app/ray_cluster_fixed.py` helpers support forming a
  multi-node Ray cluster for distributed workers.

## Prerequisites
- Python 3.10+
- GPU with recent NVIDIA drivers (tested on RTX 3090) for local inference
- Docker (optional) if you plan to run the monitoring stack
- Access to the Hugging Face model `Qwen/Qwen2.5-0.5B`

## Local Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn torch transformers prometheus-client requests ruff pytest
```

## Running the Inference API
```bash
python app/inference_api_server.py
```
- REST API available at `http://localhost:8000`
- Prometheus metrics exposed at `http://localhost:8001/metrics`
- Default worker ID `gpu-worker-1`; override via `WORKER_ID` env var

## Client Commands
```bash
python app/llm_client.py --server http://localhost:8000 --health
python app/llm_client.py --server http://localhost:8000 --prompt "What is AI?"
python app/llm_client.py --server http://localhost:8000 --batch
python app/llm_client.py --server http://localhost:8000 --file prompts.txt --batch
```

## Monitoring Stack
```bash
cd config
docker compose -f docker-compose-monitoring.yml up -d
```
- Grafana: `http://localhost:3000` (default creds `admin/admin123`)
- Prometheus: `http://localhost:9090`
- Node exporter: `http://localhost:9100`

## Multi-Node Ray Cluster
Head node (already running inside the main container):
```bash
docker exec inference-server ray start --head --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 --redis-password=ray123
```
Worker node (second machine):
```bash
docker pull michaelsigamani/proj-grounded-telescopes:0.1.0
docker run -d --name ray-worker --gpus all -p 8002:8000 \
  michaelsigamani/proj-grounded-telescopes:0.1.0
docker exec ray-worker ray start --address='77.104.167.148:6379' \
  --redis-password='ray123'
docker exec ray-worker bash -c "cd /workspace && python ray_cluster_fixed.py \
  worker 77.104.167.148:6379"
```
Dashboard: `http://77.104.167.148:8265`

## SSH Port Forwarding Shortcut
```bash
ssh -p 40195 root@77.104.167.148 \
  -L 8000:localhost:8000 \
  -L 8001:localhost:8001 \
  -L 8080:localhost:8080 \
  -L 3000:localhost:3000 \
  -L 9090:localhost:9090 \
  -L 8265:localhost:8265
```

## Testing & Linting
```bash
python -m pytest                # Full suite
python -m pytest app/tests/...  # Single test
ruff check app                  # Lint
ruff format app                 # Format
```

## Repository Layout
```
app/      FastAPI server, Ray helpers, clients, tests
config/   Monitoring stack, Nginx configuration, setup scripts
docs/     Deployment guides, debugging notes, monitoring references
```

## Useful References
- Detailed monitoring and setup notes live under `docs/`
- Use `app/quick_test.py` to smoke-test the model on a single GPU
- `app/test_api.py` offers a scripted smoke test against a running server
