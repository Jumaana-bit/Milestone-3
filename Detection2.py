import os
import json
import logging
import glob
import cv2
import torch
import numpy as np
import subprocess
import sys

# Ensure required package is installed
subprocess.run([sys.executable, "-m", "pip", "install", "google-cloud-pubsub"], check=True)

# Now import pubsub_v1 after installation
from google.cloud import pubsub_v1

from ultralytics import YOLO
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, GoogleCloudOptions, WorkerOptions, DebugOptions

# Set GOOGLE_APPLICATION_CREDENTIALS environment variable
files = glob.glob("*.json")
if files:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]
else:
    raise FileNotFoundError("Service account key JSON file not found.")

# Define the pipeline options
options = PipelineOptions()

# Set the runner to Dataflow
options.view_as(StandardOptions).runner = 'DataflowRunner'

# Set Google Cloud-specific options
options.view_as(StandardOptions).streaming = True
options.view_as(GoogleCloudOptions).project = 'cloud-451522'
options.view_as(GoogleCloudOptions).staging_location = 'gs://cloud-451522-bucket/staging'
options.view_as(GoogleCloudOptions).temp_location = 'gs://cloud-451522-bucket/temp'
options.view_as(GoogleCloudOptions).region = 'northamerica-northeast2'
options.view_as(GoogleCloudOptions).job_name = 'tria4'

# Increase worker resources
options.view_as(WorkerOptions).max_num_workers = 10
options.view_as(WorkerOptions).machine_type = 'n1-standard-4'
options.view_as(WorkerOptions).autoscaling_algorithm = 'THROUGHPUT_BASED'

options.view_as(WorkerOptions).worker_harness_container_image = 'gcr.io/cloud-451522/dataflow-custom-image:latest'

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Load MiDaS model for depth estimation
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

# Image processing functions
def detect_pedestrians(image):
    """Detect pedestrians using YOLOv8."""
    results = yolo_model(image)
    pedestrians = []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if int(cls) == 0 and conf > 0.5:  # Class 0 is 'person'
                x1, y1, x2, y2 = map(int, box.tolist())
                pedestrians.append((x1, y1, x2, y2, conf.item()))
    return pedestrians

def estimate_depth(image):
    """Generate depth map using MiDaS."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

def process_image(image_path):
    """Detect pedestrians and estimate depth."""
    image = cv2.imread(image_path)
    pedestrians = detect_pedestrians(image)
    depth_map = estimate_depth(image)
    results = []
    for x1, y1, x2, y2, conf in pedestrians:
        bbox_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(bbox_depth) if bbox_depth.size > 0 else 0
        results.append({"bbox": [x1, y1, x2, y2], "depth": avg_depth, "confidence": conf})
    return results

# Beam DoFn classes for processing and publishing
class DetectAndEstimateDepth(beam.DoFn):
    def process(self, element):
        image_data = json.loads(element)  # Assuming the message is in JSON format
        image_path = image_data['image_path']
        logging.info(f"Processing image: {image_path}")
        results = process_image(image_path)
        for res in results:
            yield res

class PublishToPubSub(beam.DoFn):
    def __init__(self, project_id, output_topic):
        self.project_id = project_id
        self.output_topic = output_topic

    def setup(self):
        try:
            logging.info("Setting up Pub/Sub publisher...")
            self.publisher = pubsub_v1.PublisherClient()
            logging.info("Pub/Sub publisher initialized.")
        except Exception as e:
            logging.error(f"Error in setup: {e}")
            raise

    def process(self, element):
        try:
            message = json.dumps(element).encode('utf-8')
            topic_path = f"projects/{self.project_id}/topics/{self.output_topic}"
            future = self.publisher.publish(topic_path, message)
            future.result()
        except Exception as e:
            logging.error(f"Error publishing to Pub/Sub: {e}")

# Main pipeline function
def run_pipeline(input_subscription, output_topic, project_id):
    with beam.Pipeline(options=options) as pipeline:
        input_data = (
            pipeline
            | "Read from Pub/Sub" >> beam.io.ReadFromPubSub(subscription=input_subscription)
            | "Process images" >> beam.ParDo(DetectAndEstimateDepth())
            | "Publish results to Pub/Sub" >> beam.ParDo(PublishToPubSub(project_id, output_topic))
        )

if __name__ == "__main__":
    input_subscription = "projects/cloud-451522/subscriptions/pedestrian-detection-sub"
    output_topic = "projects/cloud-451522/topics/pedestrian-results"
    project_id = "cloud-451522"
    run_pipeline(input_subscription, output_topic, project_id)