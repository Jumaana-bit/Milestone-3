FROM apache/beam_python3.11_sdk:latest
RUN pip install google-cloud-pubsub opencv-python torch torchvision ultralytics numpy
