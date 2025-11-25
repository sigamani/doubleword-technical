# Ray Data + vLLM Batch Inference - Operations Runbook

## Table of Contents
1. [Getting Started](#getting-started)
2. [Deployment](#deployment)
3. [Monitoring & Alerting](#monitoring--alerting)
4. [Troubleshooting](#troubleshooting)
5. [Performance Tuning](#performance-tuning)
6. [SLA Management](#sla-management)

---

## Getting Started

### Prerequisites
- Docker and docker-compose installed
- NVIDIA Docker runtime for GPU support
- 2+ GPU nodes with CUDA 12.1+
- Minimum 16GB GPU VRAM per node
- Network connectivity between nodes

### Architecture
```
Users (HTTP POST /start)
        │
        ▼
   FastAPI Gateway (Head Node, CPU)
   - Job Queue (Redis)
   - SLA Tracker
   - Metrics Exporter
        │
        ├─────────────────────┐
        │                     │
        ▼                     ▼
   GPU Worker 1          GPU Worker 2
   (Ray + vLLM)          (Ray + vLLM)
```

---

## Deployment

### Quick Start (Single Machine)
```bash
cd /path/to/repo
./deploy/setup_production.sh
```

### Multi-Node Deployment (2-Node Cluster)
```bash
# Node 1 (Head Node - has GPUs)
./deploy/setup_production.sh 192.168.1.100 192.168.1.101

# This will:
# 1. Pull Docker image
# 2. Start Ray head node on 192.168.1.100:6379
# 3. Start Ray worker on 192.168.1.101 (joins head)
# 4. Start Prometheus + Grafana + Alertmanager
# 5. Start FastAPI inference server
```

### Verify Deployment
```bash
# Check Ray cluster status
docker exec ray-head ray status

# Check running containers
docker ps | grep ray

# View logs
docker logs ray-head
docker logs ray-inference-server
```

### Access Points
| Component | URL | Purpose |
|-----------|-----|---------|
| Ray Dashboard | http://localhost:8265 | Cluster monitoring |
| Prometheus | http://localhost:9090 | Metrics scraping |
| Grafana | http://localhost:3000 | Visualization (admin/admin) |
| Alertmanager | http://localhost:9093 | Alert management |
| FastAPI | http://localhost:8000/docs | API documentation |

---

## Monitoring & Alerting

### Import Grafana Dashboard
1. Go to http://localhost:3000
2. Click "Import"
3. Upload `config/grafana_dashboard.json`
4. Select Prometheus as data source
5. Click "Import"

### Key Metrics
| Metric | Definition | Target |
|--------|-----------|--------|
| Throughput (req/s) | Requests per second | > 5 req/s |
| Token/sec | Total tokens processed | > 1000 tok/s |
| P95 Latency | 95th percentile latency | < 5s |
| GPU Memory | GPU utilization | < 90% |
| Error Rate | Failed requests | < 5% |
| ETA vs SLA | Estimated time vs remaining time | ETA < Remaining |

### Alert Rules
Alerts are auto-triggered for:
- **CRITICAL**: SLA at risk (ETA > remaining time)
- **CRITICAL**: High GPU memory (>90%)
- **WARNING**: Low throughput (<5 req/s)
- **WARNING**: High error rate (>5%)
- **WARNING**: Batch queue backlog (>100 jobs)

View active alerts:
```bash
curl http://localhost:9093/api/v1/alerts
```

---

## Troubleshooting

### Ray Cluster Won't Start
```bash
# Check if ports are in use
lsof -i :6379  # Redis port
lsof -i :8076  # Object manager
lsof -i :8265  # Dashboard

# Kill existing Ray processes
pkill -f ray

# Restart cluster
docker-compose restart
```

### Out of Memory (OOM) Errors
```bash
# Reduce batch size
# Edit config/config.yaml:
# inference:
#   batch_size: 64  # was 128

# Restart server
docker restart ray-inference-server
```

### Low Throughput
```bash
# Check worker status
docker exec ray-head ray status

# Check GPU utilization
nvidia-smi -l 1  # Update every 1s

# If workers idle, check:
# 1. Job queue size: curl http://localhost:8000/queue
# 2. Worker logs: docker logs ray-worker-*
```

### SLA Alerts But Not Critical
```bash
# Check current progress
curl http://localhost:8000/jobs/{job_id}

# Monitor ETA in real-time
watch -n 5 'curl http://localhost:8000/jobs/{job_id} | jq .eta_hours'

# Increase concurrency if possible
# (Edit config/config.yaml, inference.concurrency)
```

### Network Issues Between Nodes
```bash
# Test connectivity
docker exec ray-head ping [worker-ip]

# Check Ray addresses
docker exec ray-head ray health-check

# Force cluster restart
docker exec ray-head ray stop
docker exec ray-head ray start --head
```

---

## Performance Tuning

### Configuration Parameters

#### Batch Size
- **Default**: 128
- **Increase for**: Better throughput (if GPU memory allows)
- **Decrease for**: Reduce memory usage, lower latency
- **Max**: 1024 (for 0.5B model on RTX3090)

```yaml
inference:
  batch_size: 128  # Edit this
```

#### Concurrency
- **Default**: 2 (num workers)
- **Increase for**: More parallel processing
- **Decrease for**: Reduce contention
- **Max**: Number of GPUs available

```yaml
inference:
  concurrency: 2
```

#### GPU Memory Utilization
- **Default**: 0.90 (90%)
- **Range**: 0.5 - 0.95
- **Reduce if**: Getting OOM errors

```yaml
inference:
  gpu_memory_utilization: 0.90
```

### Benchmarking a Configuration
```bash
# Run test with 100 samples
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "batch_size": 256,
    "concurrency": 4,
    "num_samples": 100
  }'
```

### Optimal Configuration (Tested)
For **Qwen2.5-0.5B** on **RTX3090**:
- Batch size: 256 (30-50% throughput gain)
- Concurrency: 2 (per GPU)
- GPU memory: 90%
- Chunked prefill: 512 tokens
- **Result**: 202 req/s, 20.1k tok/s, $4,380/year

---

## SLA Management

### Monitor SLA Status
```bash
# Check active jobs and ETA
curl http://localhost:8000/sla/status

# Returns:
# {
#   "jobs": [
#     {
#       "job_id": "job-123",
#       "progress_pct": 45.2,
#       "eta_hours": 2.3,
#       "sla_remaining_hours": 18.5,
#       "status": "on_track"
#     }
#   ]
# }
```

### SLA Alert Thresholds
- **Yellow Alert (70% of SLA used)**: ~16.8 hours elapsed
- **Red Alert (ETA > remaining)**: Immediate action required
- **Critical (>24 hours)**: SLA violated

### Actions When SLA at Risk

1. **Immediate**:
   ```bash
   # Increase batch size
   # Increase concurrency
   # Reduce max_tokens (may affect quality)
   ```

2. **Short-term**:
   ```bash
   # Add temporary worker nodes
   # Use higher-capacity GPUs
   # Reduce queue by pausing new jobs
   ```

3. **Long-term**:
   - Review baseline throughput expectations
   - Migrate to larger models or specialized hardware
   - Implement adaptive batching

### Export SLA Report
```bash
# Get detailed SLA analytics
curl http://localhost:8000/sla/report > sla_report.json

# Contains: throughput trend, ETA accuracy, violation history
```

---

## Maintenance

### Daily Checks
- [ ] Verify cluster is healthy: `docker exec ray-head ray status`
- [ ] Check active alerts: `curl http://localhost:9093/api/v1/alerts`
- [ ] Monitor disk space: `df -h /outputs`
- [ ] Check error logs: `docker logs ray-inference-server 2>&1 | grep ERROR`

### Weekly Tasks
- [ ] Review performance metrics in Grafana
- [ ] Check SLA compliance rate
- [ ] Verify backups of job results
- [ ] Update configuration if needed
- [ ] Review and prune old logs

### Monthly Tasks
- [ ] Update Docker image
- [ ] Review and optimize configurations
- [ ] Analyze throughput trends
- [ ] Plan capacity upgrades if needed
- [ ] Test disaster recovery procedures

### Log Rotation
```bash
# Configure docker log rotation (daemon.json)
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

---

## Emergency Procedures

### Graceful Shutdown
```bash
# Stop accepting new jobs
curl -X POST http://localhost:8000/shutdown-graceful

# Wait for in-flight jobs to complete
docker logs -f ray-inference-server

# Stop containers
docker-compose down
```

### Force Restart
```bash
# Stop all
docker-compose down
docker system prune -f

# Start fresh
./deploy/setup_production.sh
```

### Data Recovery
```bash
# Batch results are written to /outputs
# Verify local or S3 backup:
aws s3 ls s3://bucket/outputs/ --recursive

# Restore from backup
aws s3 sync s3://bucket/outputs/ /outputs/
```

---

## Support & Contact
- GitHub Issues: https://github.com/sigamani/proj-grounded-telescopes
- Documentation: https://github.com/sigamani/proj-grounded-telescopes/wiki
- Ray Community: https://discuss.ray.io

---

**Last Updated**: 2025-01-15
**Version**: 1.0.0
