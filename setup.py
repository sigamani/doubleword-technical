#!/usr/bin/env python3
"""
Ray Cluster Setup Script
Sets up 2-node Ray cluster for batch inference using Python instead of bash
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Union


class RayClusterSetup:
    def __init__(self):
        self.head_ssh = "ssh -p 59554 root@77.104.167.149"
        self.worker_ssh = "ssh -p 55089 root@77.104.167.149"
        self.image = "michaelsigamani/proj-grounded-telescopes:0.1.0"
        self.project_root = Path(__file__).parent
        
    def run_command(self, cmd: str, check: bool = True, capture_output: bool = False) -> Union[subprocess.CompletedProcess, subprocess.CalledProcessError]:
        """Run a shell command with error handling."""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=check, 
                capture_output=capture_output,
                text=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {cmd}")
            print(f"Error: {e}")
            if check:
                sys.exit(1)
            return e

    def run_remote_command(self, ssh_cmd: str, remote_cmd: str, check: bool = True) -> Union[subprocess.CompletedProcess, subprocess.CalledProcessError]:
        """Run a command on remote host via SSH."""
        full_cmd = f"{ssh_cmd} '{remote_cmd}'"
        return self.run_command(full_cmd, check=check)

    def copy_files_to_remote(self):
        """Copy required files to remote hosts."""
        print("Copying files to remote hosts...")
        
        # Copy batch inference file to both hosts
        batch_file = self.project_root / "app" / "ray_data_batch_inference.py"
        if batch_file.exists():
            self.run_command(f"scp -P 59554 {batch_file} root@77.104.167.149:/tmp/", check=False)
            self.run_command(f"scp -P 55089 {batch_file} root@77.104.167.149:/tmp/", check=False)
        
        # Copy config directory if it exists
        config_dir = self.project_root / "config"
        if config_dir.exists():
            self.run_command(f"scp -P 59554 -r {config_dir} root@77.104.167.149:/tmp/", check=False)
            self.run_command(f"scp -P 55089 -r {config_dir} root@77.104.167.149:/tmp/", check=False)

    def check_container_exists(self, container_name: str, ssh_cmd: str) -> bool:
        """Check if a Docker container exists on remote host."""
        result = self.run_remote_command(
            ssh_cmd, 
            f"docker ps -q -f name={container_name}", 
            check=False
        )
        return bool(result.stdout.strip())

    def start_head(self):
        """Start Ray head node."""
        print("Starting head node...")
        
        # Copy files to remote first
        self.copy_files_to_remote()
        
        # Pull image first
        print("Pulling Docker image on head...")
        self.run_remote_command(self.head_ssh, f"docker pull {self.image}", check=False)
        
        # Start monitoring stack
        print("Starting Loki/Promtail monitoring...")
        monitoring_cmd = "cd /app/doubleword-technical && docker-compose -f docker-compose-monitoring.yml up -d"
        self.run_remote_command(self.head_ssh, monitoring_cmd, check=False)
        
        # Check if head container exists
        if not self.check_container_exists("ray-head-new", self.head_ssh):
            print("Creating head container...")
            docker_run_cmd = (
                f"docker run -d --name ray-head-new "
                f"-p 6379:6379 -p 8265:8265 -p 8001:8001 "
                f"--shm-size=10.24gb "
                f"-v /var/log:/var/log:ro "
                f"--log-driver json-file --log-opt max-size=10m --log-opt max-file=3 "
                f"{self.image}"
            )
            self.run_remote_command(self.head_ssh, docker_run_cmd)
        
        # Copy required files to container
        print("Copying files to head container...")
        self.run_remote_command(self.head_ssh, "docker cp /tmp/ray_data_batch_inference.py ray-head-new:/app/", check=False)
        
        # Start Ray head
        print("Starting Ray head...")
        head_start_cmd = "cd /app && python ray_data_batch_inference.py --mode head"
        self.run_remote_command(self.head_ssh, f"docker exec -d ray-head-new bash -c '{head_start_cmd}'")
        
        print("Head node started")
        print("Dashboard: http://localhost:8265 (via SSH tunnel)")
        print("Metrics: http://localhost:8001 (via SSH tunnel)")
        print("Grafana: http://localhost:3000 (via SSH tunnel)")

    def start_worker(self):
        """Start Ray worker node."""
        print("Starting worker node...")
        
        # Copy files to remote first
        self.copy_files_to_remote()
        
        # Check if image exists first
        print("Checking Docker image on worker...")
        image_check = self.run_remote_command(
            self.worker_ssh, 
            f"docker images -q {self.image}", 
            check=False
        )
        
        if not image_check.stdout.strip():
            print("Image not found on worker, attempting to pull...")
            pull_result = self.run_remote_command(
                self.worker_ssh, 
                f"timeout 120 docker pull {self.image}", 
                check=False
            )
            if pull_result.returncode != 0:
                print("Failed to pull image on worker (timeout or error)")
                print("Worker node will be skipped. Head node can still function.")
                print(f"You can manually pull image later with: {self.worker_ssh} 'docker pull {self.image}'")
                return
        
        # Check if worker container exists
        if not self.check_container_exists("ray-worker", self.worker_ssh):
            print("Creating worker container...")
            docker_run_cmd = (
                f"docker run -d --name ray-worker "
                f"-p 8002:8000 "
                f"--gpus all "
                f"--shm-size=10.24gb "
                f"-v /var/log:/var/log:ro "
                f"--log-driver json-file --log-opt max-size=10m --log-opt max-file=3 "
                f"{self.image}"
            )
            self.run_remote_command(self.worker_ssh, docker_run_cmd)
        
        # Copy required files to container
        print("Copying files to worker container...")
        self.run_remote_command(self.worker_ssh, "docker cp /tmp/ray_data_batch_inference.py ray-worker:/app/", check=False)
        
        # Start Ray worker
        print("Starting Ray worker...")
        worker_start_cmd = "cd /app && python ray_data_batch_inference.py --mode worker --head-address 77.104.167.149:6379"
        self.run_remote_command(self.worker_ssh, f"docker exec -d ray-worker bash -c '{worker_start_cmd}'", check=False)
        
        print("Worker node started")

    def check_status(self):
        """Check cluster status."""
        print("Checking cluster status...")
        
        print("Head node status:")
        head_status = self.run_remote_command(
            self.head_ssh, 
            "docker exec ray-head-new ray status", 
            check=False
        )
        print(head_status.stdout[:500] if head_status.stdout else "Head node not responding")
        
        print("\nWorker node status:")
        worker_status = self.run_remote_command(
            self.worker_ssh, 
            "docker exec ray-worker ray status", 
            check=False
        )
        print(worker_status.stdout[:500] if worker_status.stdout else "Worker node not responding")

    def run_inference(self):
        """Run batch inference."""
        print("Running batch inference...")
        inference_cmd = "cd /app && python ray_data_batch_inference.py --mode inference"
        self.run_remote_command(self.head_ssh, f"docker exec ray-head-new bash -c '{inference_cmd}'")

    def setup_tunnels(self):
        """Show SSH tunnel commands."""
        print("Setting up SSH tunnels...")
        print("Run these commands in separate terminals:")
        print("")
        print("# Head node tunnels:")
        print(f"{self.head_ssh} -L 8265:localhost:8265 \\")
        print(f"{self.head_ssh} -L 8001:localhost:8001 \\")
        print("")
        print("# Then access:")
        print("Ray Dashboard: http://localhost:8265")
        print("API/Metrics: http://localhost:8001")
        print("Grafana: http://localhost:3000 (admin/admin)")

    def start_monitoring(self):
        """Start monitoring stack."""
        print("Starting monitoring stack...")
        monitoring_cmd = "cd /app/doubleword-technical && docker-compose -f docker-compose-monitoring.yml up -d"
        self.run_remote_command(self.head_ssh, monitoring_cmd, check=False)

    def stop_monitoring(self):
        """Stop monitoring stack."""
        print("Stopping monitoring stack...")
        monitoring_cmd = "cd /app/doubleword-technical && docker-compose -f docker-compose-monitoring.yml down"
        self.run_remote_command(self.head_ssh, monitoring_cmd, check=False)

    def stop_cluster(self):
        """Stop Ray cluster."""
        print("Stopping Ray cluster...")
        self.run_remote_command(self.head_ssh, "docker stop ray-head-new || true", check=False)
        self.run_remote_command(self.worker_ssh, "docker stop ray-worker || true", check=False)
        print("Cluster stopped")

    def cleanup(self):
        """Clean up containers."""
        print("Cleaning up...")
        self.run_remote_command(self.head_ssh, "docker rm ray-head-new || true", check=False)
        self.run_remote_command(self.worker_ssh, "docker rm ray-worker || true", check=False)
        print("Cleanup complete")

    def start_cluster(self):
        """Start full cluster."""
        self.start_head()
        print("Waiting 10 seconds for head node to initialize...")
        time.sleep(10)
        self.start_worker()
        print("")
        print("Cluster setup complete. Checking status...")
        time.sleep(5)
        self.check_status()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ray Cluster Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py start-cluster    # Setup full cluster
  python setup.py inference        # Run batch job
  python setup.py status          # Check cluster health
  python setup.py start-monitoring # Start Loki/Promtail/Grafana
  python setup.py stop-monitoring  # Stop monitoring stack
        """
    )
    
    parser.add_argument(
        "command",
        choices=[
            "start-head", "start-worker", "start-cluster", "status",
            "inference", "tunnels", "start-monitoring", "stop-monitoring",
            "stop", "cleanup", "help"
        ],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    if args.command == "help":
        parser.print_help()
        return
    
    setup = RayClusterSetup()
    
    # Map commands to methods
    command_map = {
        "start-head": setup.start_head,
        "start-worker": setup.start_worker,
        "start-cluster": setup.start_cluster,
        "status": setup.check_status,
        "inference": setup.run_inference,
        "tunnels": setup.setup_tunnels,
        "start-monitoring": setup.start_monitoring,
        "stop-monitoring": setup.stop_monitoring,
        "stop": setup.stop_cluster,
        "cleanup": setup.cleanup,
    }
    
    if args.command in command_map:
        command_map[args.command]()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
