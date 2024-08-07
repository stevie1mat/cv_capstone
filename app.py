import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('yolov8n.pt')  # Replace with your trained model path

def predict(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Perform detection
    results = model(img_array)
    
    # Extract bounding boxes and class labels
    boxes = results[0].boxes.xyxy.cpu().numpy()  # xyxy format
    scores = results[0].boxes.conf.cpu().numpy()  # confidence scores
    classes = results[0].boxes.cls.cpu().numpy()  # class labels
    
    return boxes, scores, classes

st.title("Grocery Detection App")
st.write("Upload an image to detect groceries")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Perform prediction
    boxes, scores, classes = predict(image)
    
    # Draw bounding boxes on the image
    img_with_boxes = np.array(image)
    for box, score, cls in zip(boxes, scores, classes):
        cv2.rectangle(img_with_boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        label = f'Class: {int(cls)}, Score: {score:.2f}'
        cv2.putText(img_with_boxes, label, (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the image with bounding boxes
    st.image(img_with_boxes, caption='Detected Groceries', use_column_width=True)
