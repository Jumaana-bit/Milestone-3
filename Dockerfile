# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install required libraries
RUN pip install --no-cache-dir ultralytics torch torchvision \
    opencv-python numpy matplotlib google-cloud-pubsub apache-beam[gcp]

# Copy local files
COPY pipeline.py .

# Set the entrypoint
CMD ["python", "pipeline.py"]
