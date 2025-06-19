FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONunbuffered=1
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    g++ \
    wget \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install paddlepaddle==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
RUN pip install paddleocr==2.6.0
RUN pip install gunicorn

# Pre-download PaddleOCR model files
RUN mkdir -p /root/.paddleocr/whl/det/en /root/.paddleocr/whl/rec/en
RUN wget -O /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar \
    && tar -xf /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer.tar -C /root/.paddleocr/whl/det/en/ \
    && rm /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer.tar
RUN wget -O /root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar \
    && tar -xf /root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer.tar -C /root/.paddleocr/whl/rec/en/ \
    && rm /root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer.tar

# Ensure cache directory permissions
RUN chmod -R 755 /root/.paddleocr

# Copy the rest of the application
COPY . .

# Create directories for uploads
RUN mkdir -p /app/static/uploads /app/static/cropped_images /app/model_weights
RUN chmod -R 755 /app/static /app/model_weights

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application with increased timeout and single worker
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 240 --log-level debug main:app"]
