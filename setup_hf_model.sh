#!/bin/bash

# HF Model Download and Docker Mount Preparation Script
# This script downloads a HuggingFace model and prepares it for Docker mounting

set -e

# Check if model repository is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model-repo>"
    echo "Example: $0 Qwen/Qwen2.5-0.5B"
    exit 1
fi

MODEL_REPO="$1"

echo "=== HF Model Download and Docker Mount Preparation ==="
echo "Model Repository: $MODEL_REPO"
echo ""

################################################
# 1. Download HF model
################################################
echo "=== Downloading HF model: $MODEL_REPO ==="

# Remove existing model directory
rm -rf /model
mkdir -p /model

# Download directly to /model
hf download "$MODEL_REPO" --local-dir /model

if [ ! -d "/model" ] || [ -z "$(ls -A /model)" ]; then
    echo "ERROR: Model download failed or /model is empty"
    exit 1
fi

echo "=== Model downloaded successfully ==="
echo ""

################################################
# 2. Verify model files
################################################
echo "=== Verifying model files ==="
ls -la /model/
echo ""

# Check for essential model files
ESSENTIAL_FILES=("config.json" "pytorch_model.bin" "tokenizer.json" "tokenizer_config.json")
for file in "${ESSENTIAL_FILES[@]}"; do
    if [ -f "/model/$file" ]; then
        echo "✓ Found: $file"
    else
        echo "⚠ Missing: $file (may be optional)"
    fi
done
echo ""

################################################
# 3. Set permissions for Docker
################################################
echo "=== Setting permissions for Docker ==="
chmod -R 755 /model
echo "✓ Permissions set"
echo ""

################################################
# 4. Display Docker mount command
################################################
echo "=== Docker Mount Command ==="
echo "Use the following command to run your container with the model:"
echo ""
echo "docker run -d \\"
echo "  --name ray-worker \\"
echo "  --gpus all \\"
echo "  --ipc=host \\"
echo "  --ulimit memlock=-1 \\"
echo "  --ulimit stack=67108864 \\"
echo "  -p 8002:8000 \\"
echo "  -v /model:/model:ro \\"
echo "  michaelsigamani/proj-grounded-telescopes:0.1.0"
echo ""

echo "=== Model preparation complete! ==="
echo "Model is ready at: /model"
echo "Total size: $(du -sh /model | cut -f1)"