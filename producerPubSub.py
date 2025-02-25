import os
import time
import json
import glob
from google.cloud import pubsub_v1  # pip install google-cloud-pubsub

# Set GOOGLE_APPLICATION_CREDENTIALS environment variable
files = glob.glob("*.json")
if files:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]
else:
    raise FileNotFoundError("Service account key JSON file not found.")

# TODO: Replace with your Google Cloud project ID and Pub/Sub topic ID
project_id = "cloud-451522"
topic_id = "pedestrian-detection"

# Initialize Pub/Sub Publisher client
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

# Define directory containing your images
# Define directory containing your images
image_directory = r"C:\Users\jumia\Downloads\Milestone2"  # Use raw string (r)
  # Replace with your directory containing images

# Get all image files (assuming jpg or png files)
image_files = glob.glob(os.path.join(image_directory, "*.jpg")) + glob.glob(os.path.join(image_directory, "*.png"))

# Publish each image path to Pub/Sub
for image_path in image_files:
    value = {"image_path": image_path}  # Create a dictionary with the image path

    # Convert the dictionary to JSON and encode it before publishing
    message = json.dumps(value).encode('utf-8')

    # Publish to Pub/Sub
    future = publisher.publish(topic_path, message)
    print(f"Image at {image_path} is sent to Pub/Sub.")
    time.sleep(0.1)  # Throttle the publishing to avoid rate limiting
