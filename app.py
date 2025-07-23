import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set Streamlit page configuration
st.set_page_config(page_title="Blood Group Finder", layout="centered")
st.title("🩸 Blood Group Detection from Blood Sample Image")

# Load your trained model (use raw string or escape backslashes)
MODEL_PATH = r'C:\Users\karan\OneDrive\Desktop\bdtask\blood_group_model.h5'

# Load model once and cache it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Define class labels (make sure the order matches your training dataset)
class_names = ['A', 'AB', 'B', 'O']

# File uploader
uploaded_file = st.file_uploader("Choose a blood sample image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img_resized = image.resize((128, 128))  # Same size used during training
        img_array = np.array(img_resized) / 255.0  # Normalize to 0–1
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(img_array)[0]
        confidence = np.max(prediction)
        predicted_class = class_names[np.argmax(prediction)]

        # Display result
        st.success(f"🧬 **Predicted Blood Group:** {predicted_class}")
        st.info(f"🔍 **Confidence:** {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")

else:
    st.info("Upload a blood sample image to detect the blood group.")
