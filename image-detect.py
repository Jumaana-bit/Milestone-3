import cv2
import torch
import numpy as np
import json
import base64
from google.cloud import pubsub_v1
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import logging

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Initialize Google Cloud Pub/Sub
PROJECT_ID = "cloud-451522"
SUBSCRIPTION_NAME = "pedestrian-detection-sub"

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)

# Load YOLO model for pedestrian detection
yolo_model = YOLO("yolov8n.pt")

# Load MiDaS model for depth estimation
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load MiDaS transformation
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

def detect_pedestrians(image):
    """Detects pedestrians using YOLO."""
    results = yolo_model(image)
    pedestrians = []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if int(cls) == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.tolist())
                pedestrians.append((x1, y1, x2, y2, conf.item()))
    return pedestrians

def estimate_depth(image):
    """Estimates depth using MiDaS."""
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

def process_image(image_url):
    """Processes image from URL, detecting pedestrians and estimating depth."""
    image_data = cv2.imread(image_url)
    pedestrians = detect_pedestrians(image_data)
    depth_map = estimate_depth(image_data)

    results = []
    for x1, y1, x2, y2, conf in pedestrians:
        bbox_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(bbox_depth) if bbox_depth.size > 0 else 0

        results.append({
            "bbox": [x1, y1, x2, y2],
            "depth": avg_depth,
            "confidence": conf
        })

    return json.dumps(results)

def callback(message):
    """Processes messages from Pub/Sub."""
    try:
        image_url = base64.b64decode(message.data).decode("utf-8")
        print(f"Processing image: {image_url}")

        result = process_image(image_url)
        print(f"Result: {result}")

        message.ack()  # Acknowledge message

    except Exception as e:
        print(f"Error processing message: {e}")
        message.nack()

# Listen to messages
subscriber.subscribe(subscription_path, callback=callback)
print("Listening for messages...")
