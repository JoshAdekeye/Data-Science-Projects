import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.model import CardiovascularModel
from src.utils import setup_logging

def plot_feature_importance(importance_dict: dict, output_path: str) -> None:
    """Plot feature importance.
    
    Args:
        importance_dict: Dictionary of feature importance values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame(
        list(importance_dict.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(cm, output_path: str) -> None:
    """Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Train and evaluate the cardiovascular disease model."""
    # Setup
    logger = setup_logging()
    logger.info("Starting model training")
    
    # Create output directories
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)
    
    plots_dir = Path('training_results')
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading data")
    df = pd.read_csv('data/Cardiovascular_Disease_Dataset.csv')
    
    # Prepare features
    X = df.drop(['patientid', 'target'], axis=1)
    y = df['target']
    
    # Initialize and train model
    logger.info("Training model")
    model = CardiovascularModel()
    metrics = model.train(X, y)
    
    # Save model
    logger.info("Saving model")
    model.save('models/cardio_model.pkl')
    
    # Plot results
    logger.info("Generating training result plots")
    plot_feature_importance(
        metrics['feature_importance'],
        'training_results/feature_importance.png'
    )
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        'training_results/confusion_matrix.png'
    )
    
    # Log metrics
    logger.info("\n=== Model Performance ===")
    logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
    logger.info(f"\nROC AUC Score: {metrics['roc_auc_score']:.4f}")
    logger.info(f"\nCross-validation ROC AUC: {metrics['cv_scores_mean']:.4f} (+/- {metrics['cv_scores_std']*2:.4f})")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main() 