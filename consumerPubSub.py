import os
import json
from google.cloud import pubsub_v1  # pip install google-cloud-pubsub
import glob
import cv2  # pip install opencv-python
import torch
import numpy as np

# Set GOOGLE_APPLICATION_CREDENTIALS environment variable
files = glob.glob("*.json")
if files:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]
else:
    raise FileNotFoundError("Service account key JSON file not found.")

# TODO: Replace with your Google Cloud project ID and subscription ID
project_id = "cloud-451522"
subscription_id = "pedestrian-detection-sub"

# Initialize Pub/Sub Subscriber client
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

# Load YOLO model for pedestrian detection
from ultralytics import YOLO
yolo_model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 model

# Load MiDaS model for depth estimation
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

def detect_pedestrians(image):
    """Detect pedestrians in the image using YOLO."""
    results = yolo_model(image)
    pedestrians = []

    # YOLOv8 class ID for person (pedestrian) might be 0, verify this
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if int(cls) == 0 and conf > 0.5:  # Class 0 is 'person'
                x1, y1, x2, y2 = map(int, box.tolist())
                pedestrians.append((x1, y1, x2, y2, conf.item()))

    return pedestrians

def estimate_depth(image):
    """Estimate depth map using MiDaS."""
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
    """Process the image: Detect pedestrians and estimate depth."""
    image = cv2.imread(image_path)

    # Detect pedestrians in the image
    pedestrians = detect_pedestrians(image)

    # Estimate depth map
    depth_map = estimate_depth(image)

    # Draw bounding boxes and depth values
    image_with_bboxes = image.copy()

    results = []
    for x1, y1, x2, y2, conf in pedestrians:
        # Extract depth values within the bounding box
        bbox_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(bbox_depth) if bbox_depth.size > 0 else 0

        # Draw bounding box and the average depth inside the box
        cv2.rectangle(image_with_bboxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_with_bboxes, f"Depth: {avg_depth:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        results.append({
            "bbox": [x1, y1, x2, y2],
            "depth": avg_depth,
            "confidence": conf
        })

    return image_with_bboxes, results, depth_map

def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    """Process each message from Pub/Sub."""
    data = json.loads(message.data)
    image_path = data['image_path']

    # Process the image and print the results
    image_with_bboxes, results, depth_map = process_image(image_path)
    print(f"Processed image: {image_path}")
    for res in results:
        print(res)

    # Save or display the processed image with bounding boxes and depth
    cv2.imwrite(f"processed_{os.path.basename(image_path)}", image_with_bboxes)

    # Acknowledge the message
    message.ack()

# Start subscribing and processing messages
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
print(f"Listening for messages on {subscription_path}..\n")

# Block the main thread while listening for messages
with subscriber:
    streaming_pull_future.result()
