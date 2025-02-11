"""
Streamlit web application for Osteoporosis Risk Prediction.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import OsteoporosisModel

# Page config
st.set_page_config(
    page_title="Osteoporosis Risk Prediction",
    page_icon="🦴",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    """Load the trained model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'models', 'model.joblib')
    preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.joblib')
    label_encoder_path = os.path.join(project_root, 'models', 'label_encoder.joblib')
    
    return OsteoporosisModel.load(model_path, preprocessor_path, label_encoder_path)

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🦴 Osteoporosis Risk Prediction")
    st.write("""
    This application helps predict the risk of osteoporosis based on various health factors.
    Please input your information below to get a risk assessment.
    """)
    
    try:
        model = load_model()
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("Patient Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=50)
                gender = st.selectbox("Gender", ["Female", "Male"])
                hormonal_changes = st.selectbox("Hormonal Changes", ["Normal", "Postmenopausal"])
                family_history = st.selectbox("Family History of Osteoporosis", ["No", "Yes"])
                race_ethnicity = st.selectbox("Race/Ethnicity", ["Asian", "African American", "Caucasian"])
                
            with col2:
                weight = st.selectbox("Body Weight", ["Underweight", "Normal"])
                calcium = st.selectbox("Calcium Intake", ["Low", "Adequate"])
                vitamin_d = st.selectbox("Vitamin D Intake", ["Insufficient", "Sufficient"])
                physical_activity = st.selectbox("Physical Activity Level", ["Active", "Sedentary"])
                smoking = st.selectbox("Smoking Status", ["No", "Yes"])
                
            with col3:
                alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate"])
                medical_conditions = st.selectbox(
                    "Medical Conditions",
                    ["None", "Rheumatoid Arthritis", "Hyperthyroidism"]
                )
                medications = st.selectbox(
                    "Medications",
                    ["None", "Corticosteroids"]
                )
                prior_fractures = st.selectbox("Prior Fractures", ["No", "Yes"])
            
            submit_button = st.form_submit_button("Predict Risk")
            
            if submit_button:
                # Prepare input data
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender': [gender],
                    'Hormonal Changes': [hormonal_changes],
                    'Family History': [family_history],
                    'Race/Ethnicity': [race_ethnicity],
                    'Body Weight': [weight],
                    'Calcium Intake': [calcium],
                    'Vitamin D Intake': [vitamin_d],
                    'Physical Activity': [physical_activity],
                    'Smoking': [smoking],
                    'Alcohol Consumption': [alcohol],
                    'Medical Conditions': [medical_conditions],
                    'Medications': [medications],
                    'Prior Fractures': [prior_fractures]
                })
                
                # Make prediction
                prediction_proba = model.predict(input_data)[0]
                
                # Display results
                st.subheader("Prediction Results")
                
                # Create columns for visualization
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    # Display risk probability
                    risk_percentage = prediction_proba[1] * 100
                    st.metric(
                        label="Risk of Osteoporosis",
                        value=f"{risk_percentage:.1f}%"
                    )
                    
                    # Risk level interpretation
                    if risk_percentage < 30:
                        risk_level = "Low Risk"
                        color = "green"
                    elif risk_percentage < 60:
                        risk_level = "Moderate Risk"
                        color = "orange"
                    else:
                        risk_level = "High Risk"
                        color = "red"
                    
                    st.markdown(f"<h3 style='color: {color}'>{risk_level}</h3>", 
                              unsafe_allow_html=True)
                
                with result_col2:
                    # Create gauge chart
                    fig = px.pie(
                        values=[prediction_proba[1], prediction_proba[0]],
                        names=['Risk', 'Safe'],
                        hole=0.7,
                        color_discrete_sequence=['red', 'green']
                    )
                    fig.update_layout(
                        showlegend=False,
                        annotations=[dict(text=f"{risk_percentage:.1f}%", 
                                       x=0.5, y=0.5, 
                                       font_size=20, 
                                       showarrow=False)]
                    )
                    st.plotly_chart(fig)
                
                # Recommendations
                st.subheader("Recommendations")
                recommendations = []
                
                if age > 50:
                    recommendations.append("Consider regular bone density screenings due to age-related risk.")
                
                if calcium == "Low":
                    recommendations.append("Increase calcium intake through diet or supplements (recommended: Adequate intake).")
                
                if vitamin_d == "Insufficient":
                    recommendations.append("Consider vitamin D supplementation (recommended: Sufficient intake).")
                
                if physical_activity == "Sedentary":
                    recommendations.append("Incorporate more weight-bearing exercises into your routine.")
                
                if smoking == "Yes":
                    recommendations.append("Consider smoking cessation to improve bone health.")
                
                if alcohol == "Moderate":
                    recommendations.append("Limit alcohol consumption to reduce osteoporosis risk.")
                
                if prior_fractures == "Yes":
                    recommendations.append("Follow up with healthcare provider for fracture risk management.")
                
                if medical_conditions != "None":
                    recommendations.append(f"Regularly monitor and manage your medical condition with your healthcare provider: {medical_conditions}.")
                
                if medications != "None":
                    recommendations.append(f"Discuss medication effects on bone health with your healthcare provider: {medications}.")
                
                if not recommendations:
                    recommendations.append("Continue maintaining your current healthy lifestyle.")
                
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                st.write("""
                **Note**: These recommendations are general guidelines. Always consult with your healthcare 
                provider for personalized medical advice.
                """)
        
        # Additional information
        with st.expander("About the Model"):
            st.write("""
            This model uses machine learning to predict osteoporosis risk based on various health factors.
            The prediction is based on statistical patterns found in historical data and should be used
            as a screening tool only. Always consult with healthcare professionals for medical advice.
            
            Key factors considered in the prediction:
            - Demographics: Age, Gender, Race/Ethnicity
            - Medical History: Family History, Prior Fractures
            - Lifestyle: Physical Activity, Smoking, Alcohol Consumption
            - Health Indicators: Body Weight, Calcium Intake, Vitamin D Intake
            - Medical Conditions and Medications
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure the model is properly trained and saved before using the application.")

if __name__ == "__main__":
    main() 