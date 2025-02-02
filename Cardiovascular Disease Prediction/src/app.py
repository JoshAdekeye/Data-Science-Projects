import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from model import CardiovascularModel

# Medical term explanations
MEDICAL_TERMS = {
    'Blood Pressure': """
    **Blood Pressure (BP)** is the pressure of circulating blood against vessel walls.
    - Normal: < 120 mmHg
    - Prehypertension: 120-139 mmHg
    - Stage 1 Hypertension: 140-159 mmHg
    - Stage 2 Hypertension: ≥ 160 mmHg
    """,
    
    'Cholesterol': """
    **Serum Cholesterol** is the amount of cholesterol in your blood.
    - Desirable: < 200 mg/dL
    - Borderline High: 200-239 mg/dL
    - High: ≥ 240 mg/dL
    """,
    
    'ECG': """
    **Electrocardiogram (ECG/EKG)** measures heart's electrical activity:
    - Normal: No notable deviations
    - ST-T Wave Abnormality: Changes in heart's repolarization
    - Left Ventricular Hypertrophy: Enlarged heart muscle
    """,
    
    'ST Depression': """
    **ST Depression (oldpeak)** indicates reduced blood flow to heart:
    - Normal: ≤ 1.0 mm
    - Borderline: 1.0-2.0 mm
    - Abnormal: > 2.0 mm
    """,
    
    'Angina': """
    **Angina** is chest pain due to reduced blood flow to heart:
    - Typical: Classic heart-related chest pain
    - Atypical: Unusual presentation
    - Non-anginal: Not heart-related
    - Asymptomatic: No pain
    """
}

def show_medical_info():
    """Display medical term explanations."""
    st.sidebar.markdown("### Medical Terms Guide")
    term = st.sidebar.selectbox("Select term to learn more:", list(MEDICAL_TERMS.keys()))
    if term:
        st.sidebar.markdown(MEDICAL_TERMS[term])

def load_model():
    """Load the trained model."""
    model = CardiovascularModel()
    model.load('models/cardio_model.pkl')
    return model

def create_feature_input(key_suffix=""):
    """Create input fields for all features."""
    st.sidebar.header(f'Patient Information {key_suffix}')
    
    # Numeric inputs with tooltips
    age = st.sidebar.number_input('Age', min_value=20, max_value=80, value=40, key=f'age_{key_suffix}',
                                help="Patient's age in years")
    resting_bp = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, value=120, 
                                       key=f'bp_{key_suffix}', help="Blood pressure when at rest")
    cholesterol = st.sidebar.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=600, value=200,
                                        key=f'chol_{key_suffix}', help="Total cholesterol level in blood")
    max_hr = st.sidebar.number_input('Maximum Heart Rate', min_value=70, max_value=202, value=150,
                                   key=f'hr_{key_suffix}', help="Maximum heart rate achieved during exercise")
    oldpeak = st.sidebar.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=6.2, value=0.0,
                                    key=f'oldpeak_{key_suffix}', help="ST depression induced by exercise relative to rest")
    
    # Categorical inputs with tooltips
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'], key=f'gender_{key_suffix}')
    chest_pain = st.sidebar.selectbox('Chest Pain Type', 
                                    ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
                                    key=f'cp_{key_suffix}',
                                    help="Type of chest pain experienced")
    fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'],
                                    key=f'fbs_{key_suffix}',
                                    help="Blood sugar level after fasting")
    rest_ecg = st.sidebar.selectbox('Resting ECG Results',
                                  ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'],
                                  key=f'ecg_{key_suffix}',
                                  help="ECG results when at rest")
    exercise_angina = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'],
                                         key=f'angina_{key_suffix}',
                                         help="Chest pain induced by exercise")
    slope = st.sidebar.selectbox('ST Segment Slope', ['Upsloping', 'Flat', 'Downsloping'],
                               key=f'slope_{key_suffix}',
                               help="Slope of peak exercise ST segment")
    vessels = st.sidebar.selectbox('Number of Major Vessels', [0, 1, 2, 3],
                                 key=f'vessels_{key_suffix}',
                                 help="Number of major blood vessels colored by fluoroscopy")
    
    # Convert categorical inputs to numeric
    feature_dict = {
        'age': age,
        'gender': 1 if gender == 'Male' else 0,
        'chestpain': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(chest_pain),
        'restingBP': resting_bp,
        'serumcholestrol': cholesterol,
        'fastingbloodsugar': 1 if fasting_bs == 'Yes' else 0,
        'restingrelectro': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(rest_ecg),
        'maxheartrate': max_hr,
        'exerciseangia': 1 if exercise_angina == 'Yes' else 0,
        'oldpeak': oldpeak,
        'slope': ['Upsloping', 'Flat', 'Downsloping'].index(slope),
        'noofmajorvessels': vessels
    }
    
    return feature_dict

def plot_risk_factors(features: dict, title_suffix=""):
    """Plot key risk factors and their ranges."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Key Risk Factors Analysis {title_suffix}', fontsize=16)
    
    # Blood Pressure
    bp_categories = ['Normal', 'Prehypertension', 'Stage 1', 'Stage 2']
    bp_ranges = [120, 140, 160, 200]
    bp_value = features['restingBP']
    bp_category = pd.cut([bp_value], bins=[0] + bp_ranges, labels=bp_categories)[0]
    
    axes[0, 0].barh(bp_categories, [1]*4, alpha=0.3)
    axes[0, 0].axhline(y=bp_category, color='r', linestyle='--')
    axes[0, 0].set_title('Blood Pressure Category')
    
    # Cholesterol
    chol_categories = ['Desirable', 'Borderline High', 'High']
    chol_ranges = [200, 240, 600]
    chol_value = features['serumcholestrol']
    chol_category = pd.cut([chol_value], bins=[0] + chol_ranges, labels=chol_categories)[0]
    
    axes[0, 1].barh(chol_categories, [1]*3, alpha=0.3)
    axes[0, 1].axhline(y=chol_category, color='r', linestyle='--')
    axes[0, 1].set_title('Cholesterol Level')
    
    # Heart Rate
    hr_categories = ['Low', 'Normal', 'High']
    hr_ranges = [100, 170, 202]
    hr_value = features['maxheartrate']
    hr_category = pd.cut([hr_value], bins=[0] + hr_ranges, labels=hr_categories)[0]
    
    axes[1, 0].barh(hr_categories, [1]*3, alpha=0.3)
    axes[1, 0].axhline(y=hr_category, color='r', linestyle='--')
    axes[1, 0].set_title('Maximum Heart Rate')
    
    # Age Groups
    age_categories = ['Young Adult', 'Middle Age', 'Senior']
    age_ranges = [40, 60, 80]
    age_value = features['age']
    age_category = pd.cut([age_value], bins=[0] + age_ranges, labels=age_categories)[0]
    
    axes[1, 1].barh(age_categories, [1]*3, alpha=0.3)
    axes[1, 1].axhline(y=age_category, color='r', linestyle='--')
    axes[1, 1].set_title('Age Group')
    
    plt.tight_layout()
    return fig

def compare_profiles(model, profile1: dict, profile2: dict):
    """Compare two patient profiles."""
    df1 = pd.DataFrame([profile1])
    df2 = pd.DataFrame([profile2])
    
    # Get predictions
    pred1 = model.predict(df1)[0]
    prob1 = model.predict_proba(df1)[0]
    pred2 = model.predict(df2)[0]
    prob2 = model.predict_proba(df2)[0]
    
    # Display comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profile 1")
        if pred1 == 1:
            st.error(f'⚠️ High Risk (Probability: {prob1[1]:.2%})')
        else:
            st.success(f'✅ Low Risk (Probability: {prob1[0]:.2%})')
        risk_plot1 = plot_risk_factors(profile1, "(Profile 1)")
        st.pyplot(risk_plot1)
    
    with col2:
        st.subheader("Profile 2")
        if pred2 == 1:
            st.error(f'⚠️ High Risk (Probability: {prob2[1]:.2%})')
        else:
            st.success(f'✅ Low Risk (Probability: {prob2[0]:.2%})')
        risk_plot2 = plot_risk_factors(profile2, "(Profile 2)")
        st.pyplot(risk_plot2)
    
    # Compare key metrics
    st.subheader("Key Metrics Comparison")
    comparison_data = {
        'Metric': ['Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Depression'],
        'Profile 1': [profile1['restingBP'], profile1['serumcholestrol'], 
                     profile1['maxheartrate'], profile1['oldpeak']],
        'Profile 2': [profile2['restingBP'], profile2['serumcholestrol'], 
                     profile2['maxheartrate'], profile2['oldpeak']],
        'Difference': [
            profile2['restingBP'] - profile1['restingBP'],
            profile2['serumcholestrol'] - profile1['serumcholestrol'],
            profile2['maxheartrate'] - profile1['maxheartrate'],
            profile2['oldpeak'] - profile1['oldpeak']
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

def main():
    st.title('Cardiovascular Disease Prediction')
    st.write("""
    This app predicts the likelihood of cardiovascular disease based on patient information.
    Use the sidebar to input patient details and get predictions.
    """)
    
    # Add medical terms guide
    show_medical_info()
    
    # Load model
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Choose mode
    mode = st.radio("Select Mode:", ["Single Profile", "Compare Profiles"])
    
    if mode == "Single Profile":
        # Get feature inputs
        features = create_feature_input()
        input_df = pd.DataFrame([features])
        
        # Make prediction
        if st.sidebar.button('Predict'):
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            st.header('Prediction Results')
            if prediction == 1:
                st.error(f'⚠️ High Risk of Heart Disease (Probability: {probability[1]:.2%})')
            else:
                st.success(f'✅ Low Risk of Heart Disease (Probability: {probability[0]:.2%})')
            
            st.header('Risk Factor Analysis')
            risk_plot = plot_risk_factors(features)
            st.pyplot(risk_plot)
            
            if model.feature_importance is not None:
                st.header('Feature Importance')
                importance_df = pd.DataFrame(
                    list(model.feature_importance.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df['Feature'], importance_df['Importance'])
                ax.set_title('Feature Importance')
                st.pyplot(fig)
    
    else:  # Compare Profiles
        st.sidebar.markdown("### Profile 1")
        profile1 = create_feature_input("1")
        
        st.sidebar.markdown("### Profile 2")
        profile2 = create_feature_input("2")
        
        if st.sidebar.button('Compare Profiles'):
            compare_profiles(model, profile1, profile2)
    
    # Display information about the model
    st.sidebar.markdown("""
    ### About
    This model uses Random Forest classification to predict cardiovascular disease risk.
    The prediction is based on:
    - Personal factors (age, gender)
    - Clinical measurements (BP, cholesterol)
    - Test results (ECG, heart rate)
    - Symptoms (chest pain, angina)
    """)

if __name__ == "__main__":
    main() 