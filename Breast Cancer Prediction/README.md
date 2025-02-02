# Breast Cancer Prediction

A machine learning web application that predicts breast cancer diagnosis (malignant or benign) using the Wisconsin Breast Cancer dataset.

## Overview

This project implements machine learning models to classify breast tumors as either malignant or benign based on features extracted from digitized images of fine needle aspirates (FNA) of breast masses. The project includes both analysis scripts and an interactive web interface.

### Models Implemented
- Logistic Regression (98.25% accuracy)
- Decision Tree Classifier (94.15% accuracy)

## Project Structure

```
├── data/               # Dataset directory (create this directory)
├── src/               # Source code
│   ├── analysis.py    # Data analysis and visualization
│   ├── app.py        # Streamlit web application
│   ├── model.py      # ML model implementation
│   └── train.py      # Model training script
├── models/           # Saved model files (created after training)
├── .gitignore        # Git ignore file
├── LICENSE           # MIT License
└── README.md         # Project documentation
```

## Features

### Dataset Features
Each cell nucleus has the following features:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

Each feature has three measurements:
- Mean
- Standard Error (SE)
- "Worst" or largest (mean of the three largest values)

### Web Application Features
- Interactive input form for all measurements
- Real-time predictions
- Probability visualization
- Support for multiple models
- Detailed result explanation

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/breast-cancer-prediction.git
cd breast-cancer-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Create a `data` directory in the project root
   - Download the dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
   - Save it as `data.csv` in the `data` directory

## Usage

1. Train the models:
```bash
python src/train.py
```
This will:
- Load and preprocess the data
- Train both models (Logistic Regression and Decision Tree)
- Save the trained models in the `models` directory

2. Run the analysis script (optional):
```bash
python src/analysis.py
```
This will:
- Perform exploratory data analysis
- Generate visualizations
- Display model performance metrics

3. Launch the web application:
```bash
streamlit run src/app.py
```
This will:
- Start the Streamlit server
- Open the web interface in your default browser
- Allow you to make predictions using the trained models

## Web Application Usage

1. Input measurements:
   - Enter values for all 30 features (mean, SE, and worst measurements)
   - Use the default values as a starting point

2. Make predictions:
   - Select your preferred model (Logistic Regression or Decision Tree)
   - Click "Make Prediction" to get results
   - View the diagnosis and probability scores

## Results

### Model Performance
- Logistic Regression:
  - Accuracy: 98.25%
  - Cross-validation score: 97.48% (±2.77%)
  - Excellent precision and recall for both classes

- Decision Tree:
  - Accuracy: 94.15%
  - Cross-validation score: 90.96% (±3.26%)
  - Good performance but less stable than Logistic Regression

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Wisconsin Breast Cancer dataset from UCI Machine Learning Repository
- Scikit-learn documentation and community
- Streamlit for the web application framework