# BridgeGuardian AI — Production Koyeb Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_ENV=production

WORKDIR /app

# Install system dependencies required for OpenCV, C++ extensions, and ReportLab rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure required runtime directories exist
RUN mkdir -p backend/static/uploads backend/static/processed backend/static/reports models logs

# Expose service port
EXPOSE 8000

# Production startup using Gunicorn with Uvicorn Workers
CMD exec gunicorn -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120
