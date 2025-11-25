#!/bin/bash
set -euo pipefail

# Production Deployment Setup Script for Ray Data + vLLM Batch Inference
# Usage: ./setup_production.sh [head_node_ip] [worker_node_ips...]

echo "Ray Data + vLLM Production Deployment Setup"
echo "==========================================="

HEAD_NODE_IP="${1:-localhost}"
WORKER_IPS=("${@:2}")

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_IMAGE="michaelsigamani/proj-grounded-telescopes:latest"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose-monitoring.yml"

# Validate prerequisites
validate_env() {
    echo "Validating environment..."
    
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker not found. Please install Docker."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "ERROR: docker-compose not found. Please install docker-compose."
        exit 1
    fi
    
    echo "Prerequisites validated."
}

# Pull Docker image
pull_image() {
    echo "Pulling Docker image: $DOCKER_IMAGE"
    docker pull "$DOCKER_IMAGE"
}

# Start Ray cluster
start_ray_cluster() {
    echo "Starting Ray cluster..."
    echo "  Head node: $HEAD_NODE_IP"
    
    if [ ${#WORKER_IPS[@]} -gt 0 ]; then
        echo "  Worker nodes: ${WORKER_IPS[*]}"
    else
        echo "  Worker nodes: (standalone head node)"
    fi
    
    # Start head node
    docker run -d \
        --name ray-head \
        --net host \
        -v /tmp/ray:/tmp/ray \
        -e RAY_ADDRESS="${HEAD_NODE_IP}:6379" \
        "$DOCKER_IMAGE" \
        ray start --head --port=6379 --object-manager-port=8076 --dashboard-port=8265 --disable-usage-stats
    
    sleep 5
    
    # Start worker nodes if specified
    for worker_ip in "${WORKER_IPS[@]}"; do
        echo "Starting Ray worker on $worker_ip..."
        docker run -d \
            --name "ray-worker-$worker_ip" \
            --net host \
            -v /tmp/ray:/tmp/ray \
            "$DOCKER_IMAGE" \
            ray start --address="${HEAD_NODE_IP}:6379" --object-manager-port=8076 --disable-usage-stats
        sleep 2
    done
    
    echo "Ray cluster started. Dashboard: http://${HEAD_NODE_IP}:8265"
}

# Start monitoring stack (Prometheus + Grafana + Alertmanager)
start_monitoring() {
    echo "Starting monitoring stack..."
    
    cd "$PROJECT_ROOT"
    
    # Ensure docker-compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo "ERROR: docker-compose file not found at $COMPOSE_FILE"
        exit 1
    fi
    
    docker-compose -f "$COMPOSE_FILE" up -d
    
    echo "Monitoring stack started:"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana: http://localhost:3000 (default: admin/admin)"
    echo "  Alertmanager: http://localhost:9093"
}

# Initialize configuration
init_config() {
    echo "Initializing configuration..."
    
    # Update config with cluster details
    cat > "${PROJECT_ROOT}/config/cluster.env" << EOF
RAY_HEAD_ADDRESS=${HEAD_NODE_IP}:6379
RAY_DASHBOARD_URL=http://${HEAD_NODE_IP}:8265
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
ALERTMANAGER_URL=http://localhost:9093
EOF
    
    echo "Configuration saved to config/cluster.env"
}

# Start batch inference server
start_inference_server() {
    echo "Starting batch inference server..."
    
    docker run -d \
        --name ray-inference-server \
        --net host \
        -v "${PROJECT_ROOT}:/app" \
        -e RAY_ADDRESS="${HEAD_NODE_IP}:6379" \
        -e PYTHONUNBUFFERED=1 \
        "$DOCKER_IMAGE" \
        python /app/app/ray_data_batch_inference.py
    
    sleep 3
    echo "Inference server started. API endpoint: http://localhost:8000"
}

# Health check
health_check() {
    echo "Performing health checks..."
    
    # Check Ray cluster
    if docker exec ray-head ray status &> /dev/null; then
        echo "  Ray cluster: OK"
    else
        echo "  Ray cluster: FAILED"
        return 1
    fi
    
    # Check Prometheus
    if curl -s http://localhost:9090/api/v1/query?query=up &> /dev/null; then
        echo "  Prometheus: OK"
    else
        echo "  Prometheus: UNREACHABLE (may still be starting)"
    fi
    
    # Check inference server
    if curl -s http://localhost:8000/health &> /dev/null; then
        echo "  Inference server: OK"
    else
        echo "  Inference server: UNREACHABLE"
    fi
    
    echo "Health checks complete."
}

# Display summary
display_summary() {
    echo ""
    echo "=========================================="
    echo "Production Deployment Complete!"
    echo "=========================================="
    echo ""
    echo "Cluster Configuration:"
    echo "  Head Node: http://${HEAD_NODE_IP}:8265"
    echo "  Ray Dashboard: http://${HEAD_NODE_IP}:8265"
    echo ""
    echo "Monitoring & Alerting:"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana: http://localhost:3000"
    echo "  Alertmanager: http://localhost:9093"
    echo ""
    echo "API Endpoints:"
    echo "  Inference: http://localhost:8000/docs"
    echo "  Metrics: http://localhost:8000/metrics"
    echo ""
    echo "Next Steps:"
    echo "  1. Import Grafana dashboard: config/grafana_dashboard.json"
    echo "  2. Configure alerting rules in Prometheus"
    echo "  3. Submit batch jobs to http://localhost:8000/start"
    echo ""
    echo "View logs:"
    echo "  docker logs ray-head"
    echo "  docker logs ray-inference-server"
    echo "  docker logs -f ray-inference-server"
    echo ""
}

# Main execution
main() {
    validate_env
    pull_image
    start_ray_cluster
    start_monitoring
    init_config
    start_inference_server
    health_check
    display_summary
}

main "$@"
