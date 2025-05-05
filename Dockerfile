# Use CUDA base image if GPU is available, else use slim Python
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as gpu
FROM python:3.10-slim as cpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for deployment
RUN pip install --no-cache-dir \
    fastapi==0.68.1 \
    uvicorn==0.15.0 \
    gunicorn==20.1.0 \
    prometheus-client==0.11.0 \
    python-multipart==0.0.5

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create cache directory
RUN mkdir -p /app/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/best_model.pt
ENV CACHE_DIR=/app/cache
ENV NUM_WORKERS=4
ENV MAX_CACHE_SIZE_GB=50

# Expose ports
EXPOSE 8000  # FastAPI
EXPOSE 8001  # Prometheus metrics

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["gunicorn", "src.backend.api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
