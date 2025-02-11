import streamlit as st
from PIL import Image
import cv2
import numpy as np
from model import GenderAgeDetector
import os

# Set page config
st.set_page_config(
    page_title="Gender and Age Detection",
    page_icon="👤",
    layout="wide"
)

# Title and description
st.title("Gender and Age Detection")
st.markdown("""
This application uses deep learning to detect faces and predict gender and age from images.
Upload an image to get started!
""")

# Initialize the detector
@st.cache_resource
def load_detector():
    return GenderAgeDetector()  # Will use default path resolution

try:
    detector = load_detector()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert uploaded file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Create columns for before/after
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Process image
    try:
        result_image, predictions = detector.process_image(image)
        
        with col2:
            st.subheader("Detected Faces")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        # Display results
        st.subheader("Detection Results")
        if predictions:
            for i, pred in enumerate(predictions, 1):
                st.markdown(f"""
                **Face #{i}**
                - Gender: {pred['gender']}
                - Age Range: {pred['age']} years
                """)
        else:
            st.warning("No faces detected in the image.")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Add information about the project
st.markdown("""
---
### About
This application uses:
- OpenCV's DNN face detector
- Deep learning models for gender and age prediction
- Streamlit for the web interface

The age predictions are grouped into ranges:
- 0-2 years
- 4-6 years
- 8-12 years
- 15-20 years
- 25-32 years
- 38-43 years
- 48-53 years
- 60-100 years
""")

# Footer
st.markdown("""
---
Made with ❤️ using OpenCV and Streamlit
""") 