import os 
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import detect

import random
import cv2
model_path = 'best.pt'
model = YOLO(model_path)
res = model.predict()
import numpy as np
from sklearn.cluster import DBSCAN

# def generate_colors(n):
#     # """Generate a list of random colors."""
#     # return [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(n)]
#     colors = ['#FF0000','#FFFF00','#00FF00','#FF00FF','#FFC0CB','#00FFFF','#FFA500','#008000','#0000FF','#A52A2A','#FF0000','#FFFF00','#00FF00','#FF00FF','#FFC0CB','#00FFFF','#FFA500','#008000','#0000FF','#A52A2A']
#     return [ colors[i] for i in range(n)]
COLOR_MAP = [(0, 0, 255),(0,255, 255),(0, 255, 0),(255, 0, 255),(255, 255, 0),(0, 165, 255),(0, 128, 0)]
def cluster_boxes(boxes, eps=0.5, min_samples=1):
    """Cluster bounding boxes using DBSCAN."""
    centers = np.array([[0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])] for box in boxes])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    return clustering.labels_

def process_image(image_path, output_path, model):
    """Process a single image for kidney stone detection and color coding."""
    # Detect kidney stones in the image
    results = model.predict(source=image_path)

    # Extract bounding boxes from the results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format

    # Cluster bounding boxes
    labels = cluster_boxes(boxes, eps=20, min_samples=1)
    unique_labels = set(labels)

    # Load the image
    image = cv2.imread(image_path)

    # Draw each bounding box with its corresponding color
    c = 0
    for box, label in zip(boxes, labels):
        color = COLOR_MAP[0]#label%len(COLOR_MAP)
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(image, f'stone', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        c+=1
    c=0
    # Save the processed image
    cv2.imwrite(output_path, image)

# Define the output path
output_path = 'image_path'  #image path is needed to write
input_dir  = 'color.jpg'

process_image(input_dir, output_path, model)

# Display the processed image
processed_image = cv2.imread(output_path)
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
