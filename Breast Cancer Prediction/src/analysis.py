"""
Breast Cancer Prediction Analysis
This script implements a machine learning solution for breast cancer prediction
using the Wisconsin Breast Cancer dataset following the PACE framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set up plotting style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

def load_and_clean_data():
    """Load and clean the dataset."""
    # Load data
    df = pd.read_csv('data/data.csv')
    print("Dataset Shape:", df.shape)
    
    # Remove unnecessary columns
    df = df.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
    
    # Encode diagnosis (M=1, B=0)
    df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Display class distribution
    print("\nClass distribution:")
    print(df['diagnosis'].value_counts(normalize=True))
    print("0: Benign, 1: Malignant")
    
    return df

def perform_eda(df):
    """Perform exploratory data analysis."""
    # Correlation matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.show()
    
    # Distribution of key features
    key_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for idx, feature in enumerate(key_features):
        sns.boxplot(x='diagnosis', y=feature, data=df, ax=axes[idx//2, idx%2])
        axes[idx//2, idx%2].set_title(f'Distribution of {feature} by Diagnosis')
        axes[idx//2, idx%2].set_xlabel('Diagnosis (0: Benign, 1: Malignant)')
    plt.tight_layout()
    plt.show()
    
    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for idx, feature in enumerate(key_features):
        sns.histplot(data=df, x=feature, hue='diagnosis', multiple="stack", ax=axes[idx//2, idx%2])
        axes[idx//2, idx%2].set_title(f'Distribution of {feature} by Diagnosis')
        axes[idx//2, idx%2].set_xlabel(f'{feature} (0: Benign, 1: Malignant)')
    plt.tight_layout()
    plt.show()

def prepare_data(df):
    """Prepare data for modeling."""
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate models."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{'-'*50}")
        print(f"Training and evaluating {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        results[name]['cv_scores'] = cv_scores
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\nClassification Report:")
        print(results[name]['classification_report'])
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results[name]['confusion_matrix'], 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant']
        )
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    return models, results

def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def main():
    """Main function to run the analysis."""
    print("Starting Breast Cancer Prediction Analysis...")
    print("="*50)
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Perform EDA
    print("\nPerforming Exploratory Data Analysis...")
    perform_eda(df)
    
    # Prepare data
    print("\nPreparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Plot ROC curves
    print("\nPlotting ROC curves...")
    plot_roc_curves(models, X_test, y_test)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 