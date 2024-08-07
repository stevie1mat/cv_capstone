import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved YOLO model
model = tf.keras.models.load_model('yolov8n.pt')

def preprocess_image(image):
    # Preprocess the image to match the model input
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (416, 416))  # Assuming YOLO input size is 416x416
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    return predictions

st.title("Grocery Detection App")
st.write("Upload an image to detect groceries")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Perform prediction
    predictions = predict(image)
    
    # Display predictions (you might want to add your own post-processing to visualize results)
    st.write("Predictions:")
    st.write(predictions)

