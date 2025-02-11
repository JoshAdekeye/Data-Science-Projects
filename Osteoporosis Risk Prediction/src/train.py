"""
Training script for Osteoporosis Risk Prediction model.
"""

import os
import pandas as pd
import logging
from model import OsteoporosisModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate the dataset.
    
    Args:
        data_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded and validated dataset
    """
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Basic validation
    required_columns = [
        'Age', 'Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity',
        'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity',
        'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications',
        'Prior Fractures', 'Osteoporosis'
    ]
    
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return data

def main():
    """Main training function."""
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'osteoporosis.csv')
    model_dir = os.path.join(project_root, 'models')
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Load data
        data = load_data(data_path)
        logger.info(f"Loaded dataset with shape: {data.shape}")
        
        # Initialize and train model
        model = OsteoporosisModel()
        X, y = model.preprocess_data(data)
        
        # Train the model
        metrics = model.train(X, y)
        
        # Log metrics
        logger.info("Training completed. Metrics:")
        logger.info(f"Training accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"Validation accuracy: {metrics['validation_accuracy']:.4f}")
        logger.info("\nDetailed metrics:")
        for metric, value in metrics['detailed_metrics'].items():
            if isinstance(value, dict):
                logger.info(f"\n{metric}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"{metric}: {value:.4f}")
        
        # Save model
        model_path = os.path.join(model_dir, 'model.joblib')
        preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
        label_encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
        model.save(model_path, preprocessor_path, label_encoder_path)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 