# Osteoporosis Risk Prediction

## Overview
This project implements a machine learning model to predict osteoporosis risk based on various health factors and demographic information. Osteoporosis is a condition that weakens bones, making them fragile and more likely to break. It develops slowly over several years and is often only diagnosed when a minor fall or sudden impact causes a bone fracture. The model helps healthcare providers identify patients who may be at risk of developing osteoporosis, enabling early intervention and prevention strategies.

## Project Structure
```
├── data/               # Data directory containing datasets
│   └── osteoporosis.csv    # Main dataset
├── src/               # Source code
│   ├── eda.py        # Exploratory Data Analysis scripts
│   ├── model.py      # Core model logic and training functions
│   ├── train.py      # Training script
│   └── app.py        # Streamlit web interface
├── models/           # Trained model files
│   ├── model.joblib         # Trained model
│   ├── preprocessor.joblib  # Data preprocessor
│   └── label_encoder.joblib # Label encoder
├── reports/          # Generated analysis reports
│   ├── figures/      # Generated visualizations
│   └── summary_statistics.csv  # Dataset statistics
├── requirements.txt   # Project dependencies
├── README.md         # This documentation
└── .gitignore        # Git ignore file
```

## Features
- Machine learning model for osteoporosis risk prediction
- Comprehensive Exploratory Data Analysis (EDA)
  - Feature distribution analysis
  - Correlation analysis
  - Missing value analysis
  - Feature importance visualization
  - Target distribution analysis
- Data preprocessing and feature engineering
- Model training and evaluation
- Interactive web interface using Streamlit
  - User-friendly form for input
  - Risk prediction with probability score
  - Visual risk representation
  - Personalized health recommendations
- Comprehensive documentation

## Model Performance
The Random Forest model achieves strong performance metrics:
- Training Accuracy: 99.87%
- Validation Accuracy: 84.95%
- Precision: 86.31%
- Recall: 85.10%
- F1-Score: 84.84%

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/osteoporosis-risk-prediction.git
cd osteoporosis-risk-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Unix or MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run Exploratory Data Analysis:
```bash
python src/eda.py
```
This will generate visualizations and statistics in the `reports` directory.

2. Train the model:
```bash
python src/train.py
```
This will create trained model files in the `models` directory.

3. Run the web application:
```bash
streamlit run src/app.py
```
The application will open in your default web browser.

## Input Features

The model considers the following factors:

### Demographics
- Age (numeric)
- Gender (Female/Male)
- Race/Ethnicity (Asian/African American/Caucasian)

### Clinical Factors
- Hormonal Changes (Normal/Postmenopausal)
- Family History (Yes/No)
- Prior Fractures (Yes/No)
- Body Weight (Underweight/Normal)

### Lifestyle Factors
- Physical Activity (Active/Sedentary)
- Smoking Status (Yes/No)
- Alcohol Consumption (None/Moderate)

### Health Indicators
- Calcium Intake (Low/Adequate)
- Vitamin D Intake (Insufficient/Sufficient)

### Medical History
- Medical Conditions (None/Rheumatoid Arthritis/Hyperthyroidism)
- Medications (None/Corticosteroids)

## Model Details
- Algorithm: Random Forest Classifier
- Features: 14 input variables (mix of categorical and numerical)
- Preprocessing: 
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
  - Handles unknown categories gracefully
- Performance metrics:
  - Training Accuracy: 99.87%
  - Validation Accuracy: 84.95%
  - Precision: 86.31%
  - Recall: 85.10%
  - F1-Score: 84.84%

## Clinical Impact
The analysis provides valuable insights for healthcare professionals to:
- Identify high-risk individuals early
- Implement targeted interventions
- Develop personalized prevention strategies
- Monitor risk factors effectively
- Guide patient education and lifestyle modifications

## Limitations and Considerations
- The model should be used as a screening tool only
- Always consult healthcare professionals for medical advice
- Predictions are based on statistical patterns in historical data
- Regular model updates and validation are recommended
- Some feature combinations may not be represented in the training data

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset source: [Source information]
- Contributors and maintainers
- Healthcare professionals who provided domain expertise

## Data Dictionary
| Column                    | Description                                     |
|---------------------------|-------------------------------------------------|
| ID                        | Unique identifier for each patient              |
| Age                       | Age of the patient                              |
| Gender                    | Gender of the patient                           |
| Hormonal Changes          | Whether the patient has undergone hormonal changes |
| Family History            | Whether the patient has a family history of osteoporosis |
| Race/Ethnicity            | Race or ethnicity of the patient                |
| Body Weight               | Weight details of the patient                   |
| Calcium                   | Calcium levels in the patient's body            |
| Vitamin D                 | Vitamin D levels in the patient's body          |
| Physical Activity         | Physical activity details of the patient        |
| Smoking                   | Whether the patient smokes                      |
| Alcohol Consumption       | Whether the patient consumes alcohol            |
| Medical Conditions        | Medical conditions of the patient               |
| Medication                | Medication details of the patient               |
| Prior Fracture           | Whether the patient has had a prior fracture    |
| Osteoporosis             | Whether the patient has osteoporosis            |

## Key Insights

1. **Age and Osteoporosis**: Significant association between age and osteoporosis risk, with patients aged 20-40 showing lower risk compared to older individuals.

2. **Hormonal Changes**: Postmenopausal individuals and those with hormonal changes exhibit higher osteoporosis risk.

3. **Nutrition Factors**: Lower levels of calcium and vitamin D correlate with increased osteoporosis risk.

4. **Physical Activity**: Active individuals show lower osteoporosis risk compared to those with sedentary lifestyles.

5. **Body Weight**: Lower body weight correlates with higher osteoporosis risk.

6. **Medical Conditions**: Conditions like hyperthyroidism and medications like corticosteroids increase osteoporosis risk. 