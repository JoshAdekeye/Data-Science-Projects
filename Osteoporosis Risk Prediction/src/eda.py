"""
Exploratory Data Analysis for Osteoporosis Risk Prediction.
This script generates comprehensive visualizations and statistical analysis of the dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Class for performing exploratory data analysis."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        self.data = data
        self.target_col = 'Osteoporosis'
        # Exclude ID column from numeric columns
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.drop('ID') if 'ID' in data.columns else data.select_dtypes(include=[np.number]).columns
        self.categorical_columns = data.select_dtypes(exclude=[np.number]).columns
        
    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate basic summary statistics for numeric columns.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        logger.info("Generating summary statistics...")
        stats = self.data[self.numeric_columns].describe()
        stats.loc['missing'] = self.data[self.numeric_columns].isnull().sum()
        stats.loc['missing_percentage'] = (self.data[self.numeric_columns].isnull().sum() / len(self.data)) * 100
        return stats
    
    def analyze_target_distribution(self):
        """Analyze the distribution of the target variable."""
        if self.target_col not in self.data.columns:
            logger.warning(f"Target column '{self.target_col}' not found in dataset")
            return
            
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data, x=self.target_col)
        plt.title('Distribution of Osteoporosis Cases')
        plt.xlabel('Has Osteoporosis')
        plt.ylabel('Count')
        plt.savefig('reports/figures/target_distribution.png')
        plt.close()
        
        # Calculate class proportions
        proportions = self.data[self.target_col].value_counts(normalize=True)
        logger.info(f"Target class proportions:\n{proportions}")
        
    def plot_correlation_matrix(self):
        """Generate and save correlation matrix heatmap."""
        logger.info("Generating correlation matrix...")
        
        # Calculate correlations for numeric columns
        corr = self.data[self.numeric_columns].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('reports/figures/correlation_matrix.png')
        plt.close()
        
    def plot_feature_distributions(self):
        """Generate distribution plots for all numeric features."""
        logger.info("Plotting feature distributions...")
        
        n_features = len(self.numeric_columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 5 * n_rows))
        
        for idx, col in enumerate(self.numeric_columns, 1):
            plt.subplot(n_rows, n_cols, idx)
            sns.histplot(data=self.data, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            
        plt.tight_layout()
        plt.savefig('reports/figures/feature_distributions.png')
        plt.close()
        
    def plot_boxplots(self):
        """Generate boxplots for numeric features grouped by target."""
        logger.info("Generating boxplots...")
        
        numeric_cols = [col for col in self.numeric_columns if col != self.target_col]
        n_features = len(numeric_cols)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 6 * n_rows))
        
        for idx, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, n_cols, idx)
            sns.boxplot(data=self.data, x=self.target_col, y=col)
            plt.title(f'{col} by Osteoporosis Status')
            
        plt.tight_layout()
        plt.savefig('reports/figures/feature_boxplots.png')
        plt.close()
        
    def analyze_missing_values(self):
        """Analyze and visualize missing values."""
        logger.info("Analyzing missing values...")
        
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        }).sort_values('Percentage', ascending=False)
        
        # Plot missing values
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_df.index, y='Percentage', data=missing_df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Missing Values by Feature')
        plt.tight_layout()
        plt.savefig('reports/figures/missing_values.png')
        plt.close()
        
        return missing_df
    
    def generate_feature_importance_plot(self):
        """Generate feature importance plot using random forest."""
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("Generating feature importance plot...")
        
        # Prepare data
        numeric_cols = [col for col in self.numeric_columns if col != self.target_col]
        X = self.data[numeric_cols]
        y = self.data[self.target_col]
        
        # Train a simple random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Plot feature importances
        importance_df = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png')
        plt.close()
        
        return importance_df

def main():
    """Main function to run the EDA."""
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'osteoporosis.csv')
    reports_dir = os.path.join(project_root, 'reports', 'figures')
    
    # Create reports directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Initialize analyzer
        analyzer = DataAnalyzer(data)
        
        # Generate all analyses
        logger.info("Starting exploratory data analysis...")
        
        # Basic statistics
        stats = analyzer.generate_summary_statistics()
        stats.to_csv(os.path.join(project_root, 'reports', 'summary_statistics.csv'))
        
        # Generate all plots
        analyzer.analyze_target_distribution()
        analyzer.plot_correlation_matrix()
        analyzer.plot_feature_distributions()
        analyzer.plot_boxplots()
        analyzer.analyze_missing_values()
        analyzer.generate_feature_importance_plot()
        
        logger.info("EDA completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during EDA: {str(e)}")
        raise

if __name__ == '__main__':
    main()