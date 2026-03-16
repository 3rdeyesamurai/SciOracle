FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Ensure non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Update apt, install python3 and system dependencies, and symlink python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA 11.8 and other requirements
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the SciOracle project
COPY . .

# Ensure storage directories exist
RUN mkdir -p /app/memory

# Command to drop into interactive shell for executing the logic CLI
CMD ["bash"]
