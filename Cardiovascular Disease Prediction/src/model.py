import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from typing import Tuple, Dict, Any

class CardiovascularModel:
    """Machine learning model for cardiovascular disease prediction."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def preprocess_data(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features for model training/prediction.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Preprocessed features as numpy array
        """
        # Scale numeric features
        numeric_features = [
            'age', 'restingBP', 'serumcholestrol',
            'maxheartrate', 'oldpeak'
        ]
        X_numeric = self.scaler.fit_transform(X[numeric_features])
        
        # Get categorical features
        categorical_features = [
            'gender', 'chestpain', 'fastingbloodsugar',
            'restingrelectro', 'exerciseangia', 'slope',
            'noofmajorvessels'
        ]
        X_categorical = X[categorical_features].values
        
        # Combine features
        return np.hstack([X_numeric, X_categorical])
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train the model and evaluate performance.
        
        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Proportion of test split
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        numeric_features = [
            'age', 'restingBP', 'serumcholestrol',
            'maxheartrate', 'oldpeak'
        ]
        categorical_features = [
            'gender', 'chestpain', 'fastingbloodsugar',
            'restingrelectro', 'exerciseangia', 'slope',
            'noofmajorvessels'
        ]
        feature_names = numeric_features + categorical_features
        self.feature_importance = dict(zip(
            feature_names,
            self.model.feature_importances_
        ))
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_processed, y,
            cv=5, scoring='roc_auc'
        )
        
        # Compile metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba),
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std(),
            'feature_importance': self.feature_importance
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities for new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        X_processed = self.preprocess_data(X)
        return self.model.predict_proba(X_processed)
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to saved model
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.random_state = model_data['random_state'] 