#!/bin/bash

# Ray + vLLM Multi-Node Deployment Script
# Usage: ./deploy-ray.sh [head|worker|all|stop|logs|status] [options]

set -e

# Default values
MODE="all"
ENV_FILE=".env"
    COMPOSE_FILE="docker-compose-ray-data.yml"
PROJECT_NAME="ray-vllm-inference"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENV_FILE="$2"
                shift 2
                ;;
            -f|--file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -p|--project)
                PROJECT_NAME="$2"
                shift 2
                ;;
            -w|--workers)
                WORKER_COUNT="$2"
                shift 2
                ;;
            -m|--model)
                MODEL_NAME="$2"
                shift 2
                ;;
            --head-ip)
                HEAD_IP="$2"
                shift 2
                ;;
            --worker-ips)
                WORKER_IPS="$2"
                shift 2
                ;;
            --with-fallback)
                ENABLE_FALLBACK=1
                shift
                ;;
            --with-monitoring)
                ENABLE_MONITORING=1
                shift
                ;;
            --gpu)
                ENABLE_GPU=1
                shift
                ;;
            --dev)
                DEV_MODE=1
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            head|worker|all|stop|logs|status|test|clean)
                MODE="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

show_usage() {
    cat << EOF
Ray + vLLM Multi-Node Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

COMMANDS:
    head         Deploy only the Ray head node
    worker       Deploy only worker nodes
    all          Deploy complete cluster (default)
    stop         Stop all services
    logs         Show logs for services
    status       Show cluster status
    test         Run API tests
    clean        Clean up containers and volumes

OPTIONS:
    -e, --env FILE         Environment file (default: .env)
    -f, --file FILE        Docker compose file (default: docker-compose-ray.yml)
    -p, --project NAME     Project name (default: ray-vllm-inference)
    -w, --workers NUM      Number of workers (overrides env file)
    -m, --model MODEL      Model name (overrides env file)
    --head-ip IP           Head node IP address
    --worker-ips IPS       Comma-separated worker IPs
    --with-fallback        Enable nginx fallback
    --with-monitoring      Enable Prometheus/Grafana monitoring
    --gpu                  Enable GPU support
    --dev                  Development mode
    -h, --help             Show this help message

EXAMPLES:
    # Deploy complete cluster with monitoring
    $0 --with-monitoring --workers 2

    # Deploy with fallback
    $0 --with-fallback --model "microsoft/DialoGPT-medium"

    # Deploy only head node
    $0 head

    # Stop all services
    $0 stop

    # View logs
    $0 logs

    # Run API tests
    $0 test
EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        log_warning "Environment file $ENV_FILE not found, creating from template..."
        cat > "$ENV_FILE" << EOF
# Ray + vLLM Configuration
RAY_PASSWORD=ray123
WORKER_COUNT=2
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct

# Network Configuration
HEAD_IP=77.104.167.149
WORKER_IPS=77.104.167.149
SSH_PORT=55089

# Port Configuration
HEAD_DASHBOARD_PORT=8265
HEAD_API_PORT=8000
METRICS_PORT=8001
NGINX_FALLBACK_PORT=8080
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# GPU Configuration
GPU_MEMORY_FRACTION=0.8
MAX_CONCURRENT_REQUESTS=10

# Monitoring Configuration
GRAFANA_PASSWORD=admin123
EOF
        log_info "Created $ENV_FILE with default configuration"
    fi
    
    # Check if app directory exists
    if [[ ! -d "app" ]]; then
        log_error "app directory not found"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Prepare environment
prepare_env() {
    log_info "Preparing environment..."
    
    # Create temporary env file with overrides
    local temp_env=$(mktemp)
    cp "$ENV_FILE" "$temp_env"
    
    # Apply command line overrides
    if [[ -n "$WORKER_COUNT" ]]; then
        sed -i.bak "s/WORKER_COUNT=.*/WORKER_COUNT=$WORKER_COUNT/" "$temp_env"
    fi
    
    if [[ -n "$MODEL_NAME" ]]; then
        sed -i.bak "s/MODEL_NAME=.*/MODEL_NAME=$MODEL_NAME/" "$temp_env"
    fi
    
    if [[ -n "$HEAD_IP" ]]; then
        sed -i.bak "s/HEAD_IP=.*/HEAD_IP=$HEAD_IP/" "$temp_env"
    fi
    
    if [[ -n "$WORKER_IPS" ]]; then
        sed -i.bak "s/WORKER_IPS=.*/WORKER_IPS=$WORKER_IPS/" "$temp_env"
    fi
    
    if [[ -n "$ENABLE_GPU" ]]; then
        echo "ENABLE_GPU=true" >> "$temp_env"
    fi
    
    if [[ -n "$DEV_MODE" ]]; then
        echo "DEBUG=true" >> "$temp_env"
    fi
    
    export ENV_FILE="$temp_env"
}

# Build Docker Compose command
build_compose_cmd() {
    local cmd="docker-compose"
    if docker compose version &> /dev/null; then
        cmd="docker compose"
    fi
    
    cmd="$cmd -f $COMPOSE_FILE -p $PROJECT_NAME --env-file $ENV_FILE"
    
    # Add profiles
    if [[ -n "$ENABLE_FALLBACK" ]]; then
        cmd="$cmd --profile fallback"
    fi
    
    if [[ -n "$ENABLE_MONITORING" ]]; then
        cmd="$cmd --profile monitoring"
    fi
    
    echo "$cmd"
}

# Deploy head node
deploy_head() {
    log_info "Deploying Ray head node..."
    local compose_cmd=$(build_compose_cmd)
    $compose_cmd up -d ray-head
    log_success "Head node deployed"
    
    # Wait for head node to be ready
    log_info "Waiting for head node to be ready..."
    sleep 30
    
    # Check health
    if $compose_cmd exec ray-head curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Head node is healthy"
    else
        log_warning "Head node not ready yet, check logs"
    fi
}

# Deploy workers
deploy_workers() {
    log_info "Deploying Ray workers..."
    local compose_cmd=$(build_compose_cmd)
    $compose_cmd up -d ray-worker
    log_success "Workers deployed"
    
    # Wait for workers to connect
    log_info "Waiting for workers to connect..."
    sleep 30
    
    # Check cluster status
    local worker_count=$($compose_cmd ps ray-worker | grep -c "Up" || true)
    log_info "Workers running: $worker_count"
}

# Deploy complete cluster
deploy_all() {
    log_info "Deploying complete Ray + vLLM cluster..."
    local compose_cmd=$(build_compose_cmd)
    
    # Deploy core services first
    $compose_cmd up -d ray-head ray-worker
    
    # Wait for core services
    log_info "Waiting for core services..."
    sleep 45
    
    # Deploy optional services
    if [[ -n "$ENABLE_FALLBACK" ]]; then
        log_info "Deploying nginx fallback..."
        $compose_cmd up -d nginx-fallback
    fi
    
    if [[ -n "$ENABLE_MONITORING" ]]; then
        log_info "Deploying monitoring stack..."
        $compose_cmd up -d prometheus grafana health-monitor
    fi
    
    log_success "Cluster deployed"
}

# Stop services
stop_services() {
    log_info "Stopping Ray cluster..."
    local compose_cmd=$(build_compose_cmd)
    $compose_cmd down
    log_success "Services stopped"
}

# Show logs
show_logs() {
    local compose_cmd=$(build_compose_cmd)
    $compose_cmd logs -f
}

# Show cluster status
show_status() {
    log_info "Cluster Status:"
    local compose_cmd=$(build_compose_cmd)
    $compose_cmd ps
    
    echo ""
    log_info "Service URLs:"
    echo "  Ray Dashboard: http://localhost:${HEAD_DASHBOARD_PORT:-8265}"
    echo "  Ray Serve API: http://localhost:${HEAD_API_PORT:-8000}"
    
    if [[ -n "$ENABLE_FALLBACK" ]]; then
        echo "  Nginx Fallback: http://localhost:${NGINX_FALLBACK_PORT:-8080}"
    fi
    
    if [[ -n "$ENABLE_MONITORING" ]]; then
        echo "  Prometheus: http://localhost:${PROMETHEUS_PORT:-9090}"
        echo "  Grafana: http://localhost:${GRAFANA_PORT:-3000}"
    fi
}

# Run API tests
run_tests() {
    log_info "Running API tests..."
    
    # Wait for services to be ready
    sleep 10
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if curl -f http://localhost:${HEAD_API_PORT:-8000}/health; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Test batch inference
    log_info "Testing batch inference..."
    curl -X POST http://localhost:${HEAD_API_PORT:-8000}/generate_batch \
         -H "Content-Type: application/json" \
         -d '{"prompts": ["Hello, how are you?", "What is AI?"], "max_tokens": 50}' || {
        log_error "Batch inference test failed"
        return 1
    }
    
    log_success "API tests completed"
}

# Clean up
cleanup() {
    log_info "Cleaning up..."
    local compose_cmd=$(build_compose_cmd)
    $compose_cmd down -v --remove-orphans
    docker system prune -f
    log_success "Cleanup completed"
}

# Main execution
main() {
    parse_args "$@"
    check_prerequisites
    prepare_env
    
    case $MODE in
        head)
            deploy_head
            ;;
        worker)
            deploy_workers
            ;;
        all)
            deploy_all
            ;;
        stop)
            stop_services
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        test)
            run_tests
            ;;
        clean)
            cleanup
            ;;
        *)
            log_error "Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac
    
    # Cleanup temporary env file
    if [[ -f "$ENV_FILE" && "$ENV_FILE" == *.tmp ]]; then
        rm -f "$ENV_FILE"
    fi
}

# Run main function with all arguments
main "$@"