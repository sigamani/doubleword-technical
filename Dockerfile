FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies
RUN python3 -m pip install \
    ray[data]==2.49.1 \
    vllm==0.10.0 \
    torch>=2.0.0 \
    transformers>=4.30.0 \
    datasets>=2.10.0 \
    pyyaml>=6.0 \
    requests>=2.28.0 \
    nvidia-ml-py>=3.1.0 \
    prometheus-client>=0.14.0 \
    fastapi>=0.68.0 \
    uvicorn[standard]>=0.15.0 \
    pydantic>=1.8.0

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Expose ports
EXPOSE 8000 8265

# Default command
CMD ["python", "app/ray_data_batch_inference.py"]