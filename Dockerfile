# Use Ubuntu 24.04 as the base image for ARM64 compatibility
FROM ubuntu:24.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Python 3.11, Tesseract, OpenCV, and PaddleOCR
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    tesseract-ocr \
    libtesseract-dev \
    libopencv-dev \
    python3-opencv \
    tesseract-ocr-eng \
    build-essential \
    g++ \
    cmake \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Hugging Face CLI to download the YOLO model
RUN pip3 install --no-cache-dir huggingface_hub[cli]

# Download the YOLO model from Hugging Face
RUN huggingface-cli download Moankhaled10/expiry-detection best.pt --local-dir model_weights --cache-dir model_weights/cache

# Copy the entire application code
COPY . .

# Create static directories for uploads and cropped images
RUN mkdir -p static/uploads static/cropped_images

# Set environment variables for Flask and PaddleOCR
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PADDLE_NUM_THREADS=4

# Expose the Flask port
EXPOSE 5000

# Run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=2", "--threads=2", "main:app"]
