#!/bin/bash
# Ray Cluster Setup Script
# Sets up 2-node Ray cluster for batch inference

set -e

echo "Setting up Ray Cluster for Batch Inference"

# Configuration
HEAD_SSH="ssh -p 59554 root@77.104.167.149"
WORKER_SSH="ssh -p 55089 root@77.104.167.149"
IMAGE="michaelsigamani/proj-grounded-telescopes:0.1.0"

# Function to check if container exists
check_container() {
    local container=$1
    local ssh_cmd=$2
    
    $ssh_cmd "docker ps -q -f name=$container" && return 0 || return 1
}

# Function to start head node
start_head() {
    echo "Starting head node..."
    
    # Check if head container exists
    if ! check_container "ray-head-new" "$HEAD_SSH"; then
            echo "Creating head container..."
        $HEAD_SSH "docker run -d --name ray-head-new \
            -p 6379:6379 -p 8265:8265 -p 8001:8001 \
            -v /app/doubleword-technical:/app/doubleword-technical \
            $IMAGE"
    fi
    
    # Start Ray head
    echo "Starting Ray head..."
    $HEAD_SSH "docker exec ray-head-new bash -c 'cd /app/doubleword-technical && python batch_inference_final.py --mode head'"
    
    echo "Head node started"
    echo "Dashboard: http://localhost:8265 (via SSH tunnel)"
    echo "Metrics: http://localhost:8001 (via SSH tunnel)"
}

# Function to start worker node
start_worker() {
    echo "Starting worker node..."
    
    # Check if worker container exists
    if ! check_container "ray-worker" "$WORKER_SSH"; then
        echo "Creating worker container..."
        $WORKER_SSH "docker run -d --name ray-worker \
            -p 8002:8000 \
            --gpus all \
            $IMAGE"
    fi
    
    # Start Ray worker
    echo "Starting Ray worker..."
    $WORKER_SSH "docker exec ray-worker bash -c 'cd /app/doubleword-technical && python batch_inference_final.py --mode worker --head-address 77.104.167.149:6379'"
    
    echo "Worker node started"
}

# Function to check cluster status
check_status() {
    echo "Checking cluster status..."
    
    echo "Head node status:"
    $HEAD_SSH "docker exec ray-head-new ray status" | head -10
    
    echo "Worker node status:"
    $WORKER_SSH "docker exec ray-worker ray status" | head -10
}

# Function to run batch inference
run_inference() {
    echo "Running batch inference..."
    $HEAD_SSH "docker exec ray-head-new bash -c 'cd /app/doubleword-technical && python batch_inference_final.py --mode inference'"
}

# Function to setup SSH tunnels
setup_tunnels() {
    echo "Setting up SSH tunnels..."
    echo "Run these commands in separate terminals:"
    echo ""
    echo "# Head node tunnels:"
    echo "$HEAD_SSH -L 8265:localhost:8265 \\"
    echo "$HEAD_SSH -L 8001:localhost:8001 \\"
    echo ""
    echo "# Then access:"
    echo "Ray Dashboard: http://localhost:8265"
    echo "API/Metrics: http://localhost:8001"
}

# Function to stop cluster
stop_cluster() {
    echo "Stopping Ray cluster..."
    
    $HEAD_SSH "docker stop ray-head-new || true"
    $WORKER_SSH "docker stop ray-worker || true"
    
    echo "Cluster stopped"
}

# Function to clean up
cleanup() {
    echo "Cleaning up..."
    
    $HEAD_SSH "docker rm ray-head-new || true"
    $WORKER_SSH "docker rm ray-worker || true"
    
    echo "Cleanup complete"
}

# Main menu
case "${1:-help}" in
    "start-head")
        start_head
        ;;
    "start-worker")
        start_worker
        ;;
    "start-cluster")
        start_head
        sleep 5
        start_worker
        ;;
    "status")
        check_status
        ;;
    "inference")
        run_inference
        ;;
    "tunnels")
        setup_tunnels
        ;;
    "stop")
        stop_cluster
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        echo "Ray Cluster Setup Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start-head      Start head node only"
        echo "  start-worker    Start worker node only"
        echo "  start-cluster  Start both head and worker"
        echo "  status         Check cluster status"
        echo "  inference      Run batch inference"
        echo "  tunnels        Show SSH tunnel commands"
        echo "  stop           Stop cluster"
        echo "  cleanup        Remove containers"
        echo "  help           Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 start-cluster    # Setup full cluster"
        echo "  $0 inference        # Run batch job"
        echo "  $0 status          # Check cluster health"
        ;;
esac