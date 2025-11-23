#!/bin/bash

# NVIDIA Container Toolkit Installation Script for Ubuntu
# This script installs the NVIDIA Container Toolkit for Docker with CUDA already running

set -e

echo "Installing NVIDIA Container Toolkit on Ubuntu..."

# Update package index
echo "Updating package index..."
sudo apt-get update

# Install prerequisites
echo "Installing prerequisites..."
sudo apt-get install -y ca-certificates curl gnupg

# Add NVIDIA GPG key and repository
echo "Adding NVIDIA repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package index again
echo "Updating package index with NVIDIA repository..."
sudo apt-get update

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
echo "Configuring Docker to use NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
echo "Restarting Docker daemon..."
sudo systemctl restart docker

# Verify installation
echo "Verifying installation..."
if command -v nvidia-ctk &> /dev/null; then
    echo "✓ NVIDIA Container Toolkit installed successfully"
    nvidia-ctk --version
else
    echo "✗ NVIDIA Container Toolkit installation failed"
    exit 1
fi

# Test Docker with NVIDIA runtime
echo "Testing Docker with NVIDIA runtime..."
if sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✓ Docker with NVIDIA runtime is working correctly"
else
    echo "⚠ Docker test failed - you may need to run: sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
fi

echo ""
echo "Installation complete! You can now run GPU containers with:"
echo "docker run --gpus all <your-gpu-container>"
echo ""
echo "Example usage:"
echo "docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"