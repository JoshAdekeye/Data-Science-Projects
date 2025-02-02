"""
Streamlit web app for breast cancer prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from model import BreastCancerModel
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🏥",
    layout="wide"
)

# Initialize model
@st.cache_resource
def load_model():
    model = BreastCancerModel()
    model.load_models()
    return model

# Feature input function
def get_feature_input():
    features = {}
    
    st.write("### Enter Patient's Measurements")
    
    # Create three columns for better organization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("#### Mean Values")
        features['radius_mean'] = st.number_input('Radius Mean', value=14.0)
        features['texture_mean'] = st.number_input('Texture Mean', value=19.0)
        features['perimeter_mean'] = st.number_input('Perimeter Mean', value=92.0)
        features['area_mean'] = st.number_input('Area Mean', value=654.0)
        features['smoothness_mean'] = st.number_input('Smoothness Mean', value=0.1)
        features['compactness_mean'] = st.number_input('Compactness Mean', value=0.1)
        features['concavity_mean'] = st.number_input('Concavity Mean', value=0.1)
        features['concave points_mean'] = st.number_input('Concave Points Mean', value=0.05)
        features['symmetry_mean'] = st.number_input('Symmetry Mean', value=0.2)
        features['fractal_dimension_mean'] = st.number_input('Fractal Dimension Mean', value=0.06)
        
    with col2:
        st.write("#### Standard Error Values")
        features['radius_se'] = st.number_input('Radius SE', value=0.3)
        features['texture_se'] = st.number_input('Texture SE', value=1.3)
        features['perimeter_se'] = st.number_input('Perimeter SE', value=2.3)
        features['area_se'] = st.number_input('Area SE', value=24.0)
        features['smoothness_se'] = st.number_input('Smoothness SE', value=0.01)
        features['compactness_se'] = st.number_input('Compactness SE', value=0.02)
        features['concavity_se'] = st.number_input('Concavity SE', value=0.02)
        features['concave points_se'] = st.number_input('Concave Points SE', value=0.01)
        features['symmetry_se'] = st.number_input('Symmetry SE', value=0.02)
        features['fractal_dimension_se'] = st.number_input('Fractal Dimension SE', value=0.003)
        
    with col3:
        st.write("#### Worst Values")
        features['radius_worst'] = st.number_input('Radius Worst', value=16.0)
        features['texture_worst'] = st.number_input('Texture Worst', value=25.0)
        features['perimeter_worst'] = st.number_input('Perimeter Worst', value=104.0)
        features['area_worst'] = st.number_input('Area Worst', value=820.0)
        features['smoothness_worst'] = st.number_input('Smoothness Worst', value=0.12)
        features['compactness_worst'] = st.number_input('Compactness Worst', value=0.15)
        features['concavity_worst'] = st.number_input('Concavity Worst', value=0.15)
        features['concave points_worst'] = st.number_input('Concave Points Worst', value=0.08)
        features['symmetry_worst'] = st.number_input('Symmetry Worst', value=0.25)
        features['fractal_dimension_worst'] = st.number_input('Fractal Dimension Worst', value=0.07)
    
    return features

def plot_probability(probabilities):
    """Create a bar plot of prediction probabilities."""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ['Benign', 'Malignant']
    probs = probabilities[0] * 100
    
    # Create bar plot
    bars = ax.bar(labels, probs, color=['green', 'red'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Probability (%)')
    ax.set_title('Prediction Probabilities')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def main():
    st.title("Breast Cancer Prediction App 🏥")
    st.write("""
    This app predicts whether a breast mass is benign or malignant based on measurements from a digital image
    of a fine needle aspirate (FNA) of the breast mass.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.write("""
    This application uses machine learning to predict breast cancer diagnosis.
    It implements two models:
    - Logistic Regression (98.25% accuracy)
    - Decision Tree (94.15% accuracy)
    """)
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree"],
        index=0
    )
    
    # Get feature input
    features = get_feature_input()
    
    # Create features DataFrame
    if features:
        X = pd.DataFrame([features])
        
        if st.button("Make Prediction"):
            # Make prediction
            model_name = 'logistic_regression' if model_type == "Logistic Regression" else 'decision_tree'
            prediction, probability = model.predict(X, model=model_name)
            
            # Display results
            st.write("### Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Diagnosis")
                if prediction[0] == 'M':
                    st.error("Malignant")
                else:
                    st.success("Benign")
                
            with col2:
                st.write("#### Probability")
                fig = plot_probability(probability)
                st.pyplot(fig)
            
            # Additional information
            st.write("### Understanding the Results")
            st.write("""
            - **Benign**: The mass is not cancerous
            - **Malignant**: The mass is cancerous
            
            Please note that this is a diagnostic aid and should not be used as the sole basis for diagnosis.
            Always consult with healthcare professionals for proper medical diagnosis and treatment.
            """)

if __name__ == "__main__":
    main() 