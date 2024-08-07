import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load YOLO model (for PyTorch)
model = torch.load('yolov8n.pt')
model.eval()

def predict(image):
    # Preprocess the image
    img = np.array(image)
    img = torch.tensor(img).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img)
    
    return predictions

st.title("Grocery Detection App")
st.write("Upload an image to detect groceries")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Perform prediction
    predictions = predict(image)
    
    # Display predictions
    st.write("Predictions:")
    st.write(predictions)
