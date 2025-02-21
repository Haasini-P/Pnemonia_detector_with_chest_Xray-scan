import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Function to load and preprocess the image for prediction
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict pneumonia
def predict_pneumonia(img_path, model):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        result = "PNEUMONIA"
        confidence = prediction[0][0] * 100
    else:
        result = "NORMAL"
        confidence = (1 - prediction[0][0]) * 100
    return result, confidence

# Streamlit UI elements
st.title('Pneumonia Detection from Chest X-rays')
st.write("This application uses a trained deep learning model to detect pneumonia in chest X-ray images.")

# Upload image file
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file
    img_path = os.path.join("uploaded_images", uploaded_file.name)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)
    
    # Load the model (make sure it's in the same directory or specify path)
    model = load_model('my_model.keras')  # Adjust path as needed
    
    # Make prediction
    result, confidence = predict_pneumonia(img_path, model)
    
    # Show result and confidence
    st.write(f"*Prediction:* {result}")
    st.write(f"*Confidence:* {confidence:.2f}%")
    
    # Provide an explanation about the result
    if result == "PNEUMONIA":
        st.write("The model detected signs of pneumonia. Please consult a healthcare professional for further evaluation.")
    else:
        st.write("The model did not detect pneumonia. However, it is always advisable to consult a healthcare professional for a more accurate diagnosis.")

# Add a footer or additional information
st.sidebar.title("About")
st.sidebar.info("This web app is developed using Streamlit and a deep learning model trained on chest X-ray images. The model is capable of detecting pneumonia in the X-ray images.")