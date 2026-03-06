# Use Python 3.10 slim (matches your local)
FROM python:3.10-slim

# Install system dependencies needed by EasyOCR, PyMuPDF, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend code
COPY . .

# Render uses $PORT environment variable automatically
# Use shell form (no brackets) so $PORT expands correctly

CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4