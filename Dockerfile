# Use NVIDIA CUDA base image with Python support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies and add Python 3.12 repository
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install uv (the package manager used by the project)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy dependency files first
COPY pyproject.toml .
COPY uv.lock* .
COPY README.md .

# Copy source code for editable install
COPY src /app/src

# Install Python dependencies
RUN /root/.local/bin/uv pip install --system -e .

# Install additional dependencies for production
RUN /root/.local/bin/uv pip install --system gunicorn

# Create directories for models, cache, and data
RUN mkdir -p /app/models /app/cache /app/inputs /app/outputs /app/config && \
    chown -R appuser:appuser /app

# Copy application code (excluding what's in .dockerignore)
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser config /app/config
COPY --chown=appuser:appuser main.py /app/
COPY --chown=appuser:appuser pyproject.toml /app/
COPY --chown=appuser:appuser README.md /app/
COPY --chown=appuser:appuser CLAUDE.md /app/

# Create directories that will be used as volume mount points
RUN mkdir -p /data/inputs /data/outputs /data/models /data/cache && \
    chown -R appuser:appuser /data

# Switch to non-root user
USER appuser

# Set environment variables for NeMo to use local cache
ENV NEMO_CACHE_DIR=/data/models \
    HF_HOME=/data/models \
    TORCH_HOME=/data/models \
    XDG_CACHE_HOME=/data/cache

# Copy entrypoint script
COPY --chown=appuser:appuser docker-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]