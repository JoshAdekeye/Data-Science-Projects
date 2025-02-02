import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis import CardioAnalyzer
from src.utils import setup_logging

def main():
    """Run comprehensive cardiovascular data analysis."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting cardiovascular data analysis")
    
    # Load data
    logger.info("Loading dataset")
    df = pd.read_csv('data/Cardiovascular_Disease_Dataset.csv')
    analyzer = CardioAnalyzer(df)
    
    # 1. Basic Dataset Information
    logger.info("Analyzing basic dataset information")
    print("\n=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print("\nFeature Descriptions:")
    print(analyzer.get_feature_descriptions())
    
    # 2. Data Quality Check
    print("\n=== Data Quality Analysis ===")
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
    
    # 3. Target Distribution
    print("\n=== Target Distribution ===")
    target_dist = df['target'].value_counts(normalize=True)
    print(target_dist)
    
    # 4. Risk Factor Analysis
    print("\n=== Risk Factor Analysis ===")
    risk_factors = analyzer.analyze_risk_factors()
    print(risk_factors)
    
    # 5. Outlier Analysis
    print("\n=== Outlier Analysis ===")
    numeric_cols = ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak']
    outliers = analyzer.identify_outliers(numeric_cols)
    print(outliers)
    
    # 6. Visualizations
    logger.info("Generating visualizations")
    
    # Age Distribution
    plt.figure(figsize=(10, 6))
    analyzer.plot_age_distribution_by_target()
    plt.savefig('analysis_results/age_distribution.png')
    plt.close()
    
    # Categorical Features
    analyzer.plot_categorical_features()
    plt.savefig('analysis_results/categorical_features.png')
    plt.close()
    
    # Correlation Matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.savefig('analysis_results/correlation_matrix.png')
    plt.close()
    
    # 7. Feature Importance Analysis
    print("\n=== Feature Correlations with Target ===")
    target_correlations = df.corr()['target'].sort_values(ascending=False)
    print(target_correlations)
    
    # 8. Age Group Analysis
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['<30', '30-45', '45-60', '>60'])
    age_group_analysis = df.groupby('age_group')['target'].mean()
    print("\n=== Disease Prevalence by Age Group ===")
    print(age_group_analysis)
    
    # 9. Blood Pressure Analysis
    df['bp_category'] = pd.cut(df['restingBP'], 
                              bins=[0, 120, 140, 160, 300], 
                              labels=['Normal', 'Prehypertension', 'Stage 1', 'Stage 2'])
    bp_analysis = df.groupby('bp_category')['target'].mean()
    print("\n=== Disease Prevalence by Blood Pressure Category ===")
    print(bp_analysis)
    
    # 10. Summary Statistics
    print("\n=== Summary Statistics ===")
    print(df[numeric_cols].describe())
    
    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    import os
    os.makedirs('analysis_results', exist_ok=True)
    
    # Run analysis
    main() 