# Cardiovascular Disease Prediction

A machine learning web application that predicts cardiovascular disease risk using patient health data from a multispecialty hospital in India.

## Overview

This project implements a Random Forest classifier to predict cardiovascular disease risk based on various patient health metrics. The project includes both analysis scripts and an interactive web interface with medical term explanations and profile comparison capabilities.

### Model Implemented
- Random Forest Classifier (98% accuracy)

### Model Selection Rationale

We chose Random Forest for the following reasons:

1. **Random Forest Classifier**:
   - Excellent performance for medical classification tasks
   - Provides probability scores for risk assessment
   - Feature importance ranking for medical insights
   - Robust to outliers and non-linear relationships
   - Reduces overfitting through ensemble learning
   - Handles both numerical and categorical features effectively
   - Suitable for medical applications where reliability is crucial

The model was selected with a focus on:
- Prediction accuracy
- Robust performance
- Feature importance insights
- Handling mixed data types
- Reliable probability estimates

## Results and Performance

### Model Performance
- Accuracy: 98%
- ROC AUC Score: 0.9993
- Cross-validation ROC AUC: 0.9973 (±0.0051)
- Excellent precision and recall for both classes
- Consistent performance across different data splits

### Key Findings
- Model significantly outperforms baseline
- Strong predictive power for both low and high risk cases
- Feature importance analysis reveals key risk factors:
  - Slope of peak exercise ST segment
  - Chest pain type
  - Number of major vessels
  - Resting blood pressure
  are the most significant predictors

### Clinical Implications
- High accuracy suitable for clinical screening
- Probability scores for risk assessment
- Visual analysis of risk factors
- Comparative analysis between patients
- Medical term explanations for better understanding

## Project Structure

```
├── data/                # Dataset directory
│   └── Cardiovascular_Disease_Dataset.csv
├── src/                # Source code
│   ├── analysis.py     # Data analysis and visualization
│   ├── app.py         # Streamlit web application
│   ├── model.py       # ML model implementation
│   └── train.py       # Model training script
├── models/            # Saved model files
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Features

### Dataset Features
Each patient record includes:
- Age
- Gender
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise induced angina
- ST depression
- Slope of ST segment
- Number of major vessels
- Target (disease presence)

### Web Application Features
1. **Single Profile Analysis**:
   - Interactive input form for all measurements
   - Real-time predictions with probability scores
   - Risk factor visualization
   - Feature importance display

2. **Profile Comparison**:
   - Side-by-side comparison of two patients
   - Comparative risk analysis
   - Key metrics comparison
   - Visual risk factor comparison

3. **Medical Information**:
   - Detailed explanations of medical terms
   - Reference ranges for measurements
   - Risk factor categorization
   - Interactive medical guide

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure the dataset is in the `data` directory:
   - `data/Cardiovascular_Disease_Dataset.csv`

## Usage

1. Train the model:
```bash
python src/train.py
```
This will:
- Load and preprocess the data
- Train the Random Forest model
- Save the trained model
- Generate performance metrics

2. Run the analysis script (optional):
```bash
python src/analyze_cardio.py
```
This will:
- Perform exploratory data analysis
- Generate visualizations
- Display detailed statistics

3. Launch the web application:
```bash
streamlit run src/app.py
```
This will:
- Start the Streamlit server
- Open the web interface
- Enable predictions and comparisons

## Web Application Usage

1. Single Profile Mode:
   - Enter patient information in the sidebar
   - Click "Predict" to get risk assessment
   - View risk factor analysis and feature importance

2. Compare Profiles Mode:
   - Enter information for two patients
   - Click "Compare Profiles"
   - View side-by-side comparison and analysis

3. Medical Terms Guide:
   - Select terms from the dropdown
   - View detailed explanations and ranges
   - Use tooltips for quick reference

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Cardiovascular Disease Dataset from Mendeley Data
- Scikit-learn documentation and community
- Streamlit for the web application framework
- Medical references for term explanations 