# Use official Python slim image (3.10 - stable & small)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install only necessary system packages for EasyOCR + others
# Removed libgl1-mesa-glx (obsolete) - EasyOCR works without it on CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libsndfile1 \
    git \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Expose port (Render uses $PORT)
EXPOSE $PORT
# Render uses $PORT environment variable automatically
# Use shell form (no brackets) so $PORT expands correctly

CMD uvicorn main:app --host 0.0.0.0 --port $PORT