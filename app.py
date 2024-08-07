import streamlit as st
import cv2
import numpy as np
from PIL import Image
import yolov5

# Load YOLOv5 model (assuming YOLOv5 is being used)
model = yolov5.load('yolov5s')  # You can replace 'yolov5s' with your custom model path if needed

def predict(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Perform detection
    results = model(img_array)
    
    # Convert results to a pandas DataFrame for easy display and processing
    df_results = results.pandas().xyxy[0]
    
    return df_results

st.title("Grocery Detection App")
st.write("Upload an image to detect groceries")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Perform prediction
    results = predict(image)
    
    # Display predictions
    st.write("Predictions:")
    st.write(results)

