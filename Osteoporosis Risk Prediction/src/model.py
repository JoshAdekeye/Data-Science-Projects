"""
Core model implementation for Osteoporosis Risk Prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OsteoporosisModel:
    """Main model class for Osteoporosis Risk Prediction."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        """
        Preprocess the input data.
        
        Args:
            data (pd.DataFrame): Raw input data
            
        Returns:
            tuple: Preprocessed features and labels
        """
        # Remove ID column if it exists
        if 'Id' in data.columns:
            data = data.drop('Id', axis=1)
        
        # Separate features and target
        X = data.drop('Osteoporosis', axis=1)
        y = data['Osteoporosis']
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing steps for both numeric and categorical data
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Fit and transform the data
        X_transformed = self.preprocessor.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_transformed, y_encoded
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the model.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            
        Returns:
            dict: Training metrics
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        # Generate detailed metrics
        y_pred = self.model.predict(X_val)
        metrics = classification_report(y_val, y_pred, output_dict=True)
        
        return {
            'train_accuracy': train_score,
            'validation_accuracy': val_score,
            'detailed_metrics': metrics
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_transformed)
    
    def save(self, model_path: str, preprocessor_path: str, label_encoder_path: str):
        """
        Save the model and preprocessor.
        
        Args:
            model_path (str): Path to save the model
            preprocessor_path (str): Path to save the preprocessor
            label_encoder_path (str): Path to save the label encoder
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        logger.info(f"Label encoder saved to {label_encoder_path}")
    
    @classmethod
    def load(cls, model_path: str, preprocessor_path: str, label_encoder_path: str) -> 'OsteoporosisModel':
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model
            preprocessor_path (str): Path to the saved preprocessor
            label_encoder_path (str): Path to the saved label encoder
            
        Returns:
            OsteoporosisModel: Loaded model instance
        """
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.preprocessor = joblib.load(preprocessor_path)
        instance.label_encoder = joblib.load(label_encoder_path)
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        logger.info(f"Label encoder loaded from {label_encoder_path}")
        return instance 