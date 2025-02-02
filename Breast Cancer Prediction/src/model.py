import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import os

class BreastCancerModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.log_reg = LogisticRegression(random_state=42)
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        
    def load_data(self, data_path):
        """Load and preprocess the breast cancer dataset."""
        # Read the data
        df = pd.read_csv(data_path)
        
        # Handle missing values
        print("Shape before cleaning:", df.shape)
        
        # Remove the empty column and ID column
        df = df.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
        
        print("\nColumns in dataset:")
        print(df.columns.tolist())
        
        # Remove any duplicate rows
        df = df.drop_duplicates()
        
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Separate features and target
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        # Handle missing values in features using mean imputation
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        print("\nShape after cleaning:", X.shape)
        print("\nFeatures:", X.columns.tolist())
        
        return X, y
    
    def preprocess_data(self, X, y):
        """Split and scale the data."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train both logistic regression and decision tree models."""
        self.log_reg.fit(X_train, y_train)
        self.decision_tree.fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models and return their performance metrics."""
        results = {}
        
        # Logistic Regression evaluation
        log_pred = self.log_reg.predict(X_test)
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, log_pred),
            'classification_report': classification_report(y_test, log_pred),
            'confusion_matrix': confusion_matrix(y_test, log_pred)
        }
        
        # Decision Tree evaluation
        tree_pred = self.decision_tree.predict(X_test)
        results['decision_tree'] = {
            'accuracy': accuracy_score(y_test, tree_pred),
            'classification_report': classification_report(y_test, tree_pred),
            'confusion_matrix': confusion_matrix(y_test, tree_pred)
        }
        
        return results
    
    def predict(self, X, model='logistic_regression'):
        """Make predictions using the specified model."""
        # Handle missing values in input data
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        X_scaled = self.scaler.transform(X)
        
        if model == 'logistic_regression':
            return self.log_reg.predict(X_scaled), self.log_reg.predict_proba(X_scaled)
        elif model == 'decision_tree':
            return self.decision_tree.predict(X_scaled), self.decision_tree.predict_proba(X_scaled)
        else:
            raise ValueError("Model must be either 'logistic_regression' or 'decision_tree'")
    
    def save_models(self, output_dir='models'):
        """Save trained models and preprocessors."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save models
        joblib.dump(self.log_reg, os.path.join(output_dir, 'logistic_regression.joblib'))
        joblib.dump(self.decision_tree, os.path.join(output_dir, 'decision_tree.joblib'))
        
        # Save preprocessors
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.joblib'))
        joblib.dump(self.imputer, os.path.join(output_dir, 'imputer.joblib'))
    
    def load_models(self, model_dir='models'):
        """Load trained models and preprocessors."""
        self.log_reg = joblib.load(os.path.join(model_dir, 'logistic_regression.joblib'))
        self.decision_tree = joblib.load(os.path.join(model_dir, 'decision_tree.joblib'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        self.imputer = joblib.load(os.path.join(model_dir, 'imputer.joblib')) 