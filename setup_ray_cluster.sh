#!/bin/bash
# Ray Cluster Setup Script
# Sets up 2-node Ray cluster for batch inference

set -e

echo "Setting up Ray Cluster for Batch Inference"

# Configuration
HEAD_SSH="ssh -p 59554 root@77.104.167.149"
WORKER_SSH="ssh -p 55089 root@77.104.167.149"
IMAGE="michaelsigamani/proj-grounded-telescopes:0.1.0"

# Function to copy files to remote host
copy_files_to_remote() {
    echo "Copying files to remote hosts..."
    
    # Copy batch inference file to both hosts
    scp -P 59554 app/ray_data_batch_inference.py root@77.104.167.149:/tmp/ || true
    scp -P 55089 app/ray_data_batch_inference.py root@77.104.167.149:/tmp/ || true
    
    # Copy config files if they exist
    if [ -d "config" ]; then
        scp -P 59554 -r config root@77.104.167.149:/tmp/ || true
        scp -P 55089 -r config root@77.104.167.149:/tmp/ || true
    fi
}

# Function to check if container exists
check_container() {
    local container=$1
    local ssh_cmd=$2
    
    local result=$($ssh_cmd "docker ps -q -f name=$container" 2>/dev/null)
    if [ -n "$result" ]; then
        return 0
    else
        return 1
    fi
}

# Function to start head node
start_head() {
    echo "Starting head node..."
    
    # Copy files to remote first
    copy_files_to_remote
    
    # Pull image first
    echo "Pulling Docker image on head..."
    $HEAD_SSH "docker pull $IMAGE" || {
        echo "Failed to pull image on head"
        return 1
    }
    
    # Start monitoring stack
    echo "Starting Loki/Promtail monitoring..."
    $HEAD_SSH "cd /app/doubleword-technical && docker-compose -f docker-compose-monitoring.yml up -d" || {
        echo "Failed to start monitoring stack (may already be running)"
    }
    
    # Check if head container exists
    if ! check_container "ray-head-new" "$HEAD_SSH"; then
            echo "Creating head container..."
        $HEAD_SSH "docker run -d --name ray-head-new \
            -p 6379:6379 -p 8265:8265 -p 8001:8001 \
            --shm-size=10.24gb \
            -v /var/log:/var/log:ro \
            --log-driver json-file --log-opt max-size=10m --log-opt max-file=3 \
            $IMAGE"
    fi
    
    # Copy required files to container
    echo "Copying files to head container..."
    $HEAD_SSH "docker cp /tmp/ray_data_batch_inference.py ray-head-new:/app/"
    
    # Start Ray head
    echo "Starting Ray head..."
    $HEAD_SSH "docker exec -d ray-head-new bash -c 'cd /app && python ray_data_batch_inference.py --mode head'"
    
    echo "Head node started"
    echo "Dashboard: http://localhost:8265 (via SSH tunnel)"
    echo "Metrics: http://localhost:8001 (via SSH tunnel)"
    echo "Grafana: http://localhost:3000 (via SSH tunnel)"
}

# Function to start worker node
start_worker() {
    echo "Starting worker node..."
    
    # Copy files to remote first
    copy_files_to_remote
    
    # Check if image exists first
    echo "Checking Docker image on worker..."
    if ! $WORKER_SSH "docker images -q $IMAGE"; then
        echo "Image not found on worker, attempting to pull..."
        if ! $WORKER_SSH "timeout 120 docker pull $IMAGE"; then
            echo "Failed to pull image on worker (timeout or error)"
            echo "Worker node will be skipped. Head node can still function."
            echo "You can manually pull image later with: $WORKER_SSH 'docker pull $IMAGE'"
            return 0  # Don't fail the entire cluster setup
        fi
    else
        echo "Image already exists on worker"
    fi
    
    # Check if worker container exists
    if ! check_container "ray-worker" "$WORKER_SSH"; then
        echo "Creating worker container..."
        $WORKER_SSH "docker run -d --name ray-worker \
            -p 8002:8000 \
            --gpus all \
            --shm-size=10.24gb \
            -v /var/log:/var/log:ro \
            --log-driver json-file --log-opt max-size=10m --log-opt max-file=3 \
            $IMAGE"
    fi
    
    # Copy required files to container
    echo "Copying files to worker container..."
    $WORKER_SSH "docker cp /tmp/ray_data_batch_inference.py ray-worker:/app/" || {
        echo "Failed to copy files to worker container"
        return 0  # Don't fail the entire cluster setup
    }
    
    # Start Ray worker
    echo "Starting Ray worker..."
    $WORKER_SSH "docker exec -d ray-worker bash -c 'cd /app && python ray_data_batch_inference.py --mode worker --head-address 77.104.167.149:6379'" || {
        echo "Failed to start Ray worker process"
        return 0  # Don't fail the entire cluster setup
    }
    
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
    $HEAD_SSH "docker exec ray-head-new bash -c 'cd /app && python ray_data_batch_inference.py --mode inference'"
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
    echo "Grafana: http://localhost:3000 (admin/admin)"
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
        echo "Waiting 10 seconds for head node to initialize..."
        sleep 10
        start_worker
        echo ""
        echo "Cluster setup complete. Checking status..."
        sleep 5
        ./setup_ray_cluster.sh status
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
    "start-monitoring")
        echo "Starting monitoring stack..."
        $HEAD_SSH "cd /app/doubleword-technical && docker-compose -f docker-compose-monitoring.yml up -d"
        ;;
    "stop-monitoring")
        echo "Stopping monitoring stack..."
        $HEAD_SSH "cd /app/doubleword-technical && docker-compose -f docker-compose-monitoring.yml down"
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
        echo "  $0 start-monitoring # Start Loki/Promtail/Grafana"
        echo "  $0 stop-monitoring  # Stop monitoring stack"
        ;;
esac