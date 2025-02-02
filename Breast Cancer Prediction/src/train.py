"""
Train and save breast cancer prediction models.
"""

from model import BreastCancerModel
import os

def main():
    # Initialize the model
    model = BreastCancerModel()
    
    # Load data
    data_path = os.path.join('data', 'data.csv')
    X, y = model.load_data(data_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = model.preprocess_data(X, y)
    
    # Train models
    print("\nTraining models...")
    model.train_models(X_train, y_train)
    
    # Evaluate models
    results = model.evaluate_models(X_test, y_test)
    
    # Print results
    print("\nLogistic Regression Results:")
    print(f"Accuracy: {results['logistic_regression']['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['logistic_regression']['classification_report'])
    
    print("\nDecision Tree Results:")
    print(f"Accuracy: {results['decision_tree']['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['decision_tree']['classification_report'])
    
    # Save models
    print("\nSaving models...")
    model.save_models()
    print("Models saved successfully!")

if __name__ == "__main__":
    main() 