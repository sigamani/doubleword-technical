#!/bin/bash

# Wait for locks with timeout
echo "Waiting for package locks to be released (max 60 seconds)..."
timeout=60
elapsed=0
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    if [ $elapsed -ge $timeout ]; then
        echo "Timeout waiting for locks. Attempting to kill unattended-upgrades..."
        sudo killall unattended-upgr apt apt-get
        sleep 5
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo "Still waiting... ($elapsed seconds)"
done

echo "Locks released. Proceeding with installation..."

# 1. Install Docker CE
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 2. Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$(. /etc/os-release; echo $ID$VERSION_ID)/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 4. Configure NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 5. Verify installation
echo "========================================"
echo "Verifying Docker installation..."
sudo docker --version
sudo docker compose version

echo "========================================"
echo "Verifying NVIDIA runtime..."
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

echo "========================================"
echo "Installation complete!"
echo "Use 'docker compose' (not 'docker-compose') for v2"
