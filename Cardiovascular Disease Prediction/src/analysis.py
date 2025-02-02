import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataAnalyzer:
    """Class for data analysis and visualization."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the analyzer with a DataFrame.
        
        Args:
            df: Input DataFrame for analysis
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        
    def get_basic_stats(self) -> Dict:
        """Get basic statistics of the dataset.
        
        Returns:
            Dictionary containing basic statistics
        """
        stats = {
            "n_rows": len(self.df),
            "n_cols": len(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "dtypes": self.df.dtypes.to_dict(),
            "numeric_summary": self.df[self.numeric_cols].describe().to_dict()
        }
        return stats
    
    def plot_distributions(
        self,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """Plot distributions of numeric columns.
        
        Args:
            columns: List of columns to plot (defaults to all numeric)
            figsize: Figure size (width, height)
        """
        if columns is None:
            columns = self.numeric_cols
            
        plt.figure(figsize=figsize)
        for i, col in enumerate(columns, 1):
            plt.subplot(len(columns), 1, i)
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        
    def plot_correlation_matrix(
        self,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Plot correlation matrix heatmap.
        
        Args:
            figsize: Figure size (width, height)
        """
        corr = self.df[self.numeric_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        plt.title('Correlation Matrix')
        
    def plot_pca_analysis(
        self,
        n_components: int = 2,
        target_col: Optional[str] = None
    ) -> Tuple[PCA, np.ndarray]:
        """Perform and plot PCA analysis.
        
        Args:
            n_components: Number of PCA components
            target_col: Target column for color coding
            
        Returns:
            Tuple of (PCA object, transformed data)
        """
        # Prepare data
        X = self.df[self.numeric_cols]
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
            
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        if n_components >= 2:
            if target_col and target_col in self.df.columns:
                scatter = plt.scatter(
                    X_pca[:, 0],
                    X_pca[:, 1],
                    c=self.df[target_col],
                    cmap='viridis'
                )
                plt.colorbar(scatter, label=target_col)
            else:
                plt.scatter(X_pca[:, 0], X_pca[:, 1])
                
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA Analysis')
            
        # Plot explained variance ratio
        plt.figure(figsize=(8, 4))
        plt.plot(
            range(1, n_components + 1),
            np.cumsum(pca.explained_variance_ratio_),
            'bo-'
        )
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        
        return pca, X_pca
    
    def identify_outliers(
        self,
        columns: Optional[List[str]] = None,
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """Identify outliers using IQR method.
        
        Args:
            columns: List of columns to check (defaults to all numeric)
            threshold: IQR multiplier for outlier detection
            
        Returns:
            DataFrame with outlier information
        """
        if columns is None:
            columns = self.numeric_cols
            
        outliers = {}
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_outliers': len(self.df[
                    (self.df[col] < lower_bound) |
                    (self.df[col] > upper_bound)
                ]),
                'outlier_percentage': len(self.df[
                    (self.df[col] < lower_bound) |
                    (self.df[col] > upper_bound)
                ]) / len(self.df) * 100
            }
            
        return pd.DataFrame.from_dict(outliers, orient='index')

class CardioAnalyzer(DataAnalyzer):
    """Specialized analyzer for cardiovascular disease dataset."""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.feature_descriptions = {
            'age': 'Age of the patient',
            'gender': 'Gender (1: Male, 0: Female)',
            'chestpain': 'Chest pain type (0-3)',
            'restingBP': 'Resting blood pressure (mm Hg)',
            'serumcholestrol': 'Serum cholesterol (mg/dl)',
            'fastingbloodsugar': 'Fasting blood sugar > 120 mg/dl (1: True, 0: False)',
            'restingrelectro': 'Resting electrocardiographic results (0-2)',
            'maxheartrate': 'Maximum heart rate achieved',
            'exerciseangia': 'Exercise induced angina (1: Yes, 0: No)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of the peak exercise ST segment (0-3)',
            'noofmajorvessels': 'Number of major vessels colored by fluoroscopy (0-3)',
            'target': 'Heart disease diagnosis (1: Present, 0: Absent)'
        }
    
    def get_feature_descriptions(self) -> pd.DataFrame:
        """Get descriptions of all features.
        
        Returns:
            DataFrame with feature descriptions
        """
        return pd.DataFrame.from_dict(
            self.feature_descriptions,
            orient='index',
            columns=['Description']
        )
    
    def plot_age_distribution_by_target(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot age distribution by disease presence.
        
        Args:
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        sns.boxplot(x='target', y='age', data=self.df)
        plt.title('Age Distribution by Heart Disease Presence')
        plt.xlabel('Heart Disease Present')
        plt.ylabel('Age')
        
    def plot_categorical_features(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot distribution of categorical features.
        
        Args:
            figsize: Figure size (width, height)
        """
        categorical_cols = [
            'gender', 'chestpain', 'fastingbloodsugar',
            'restingrelectro', 'exerciseangia', 'slope',
            'noofmajorvessels'
        ]
        
        plt.figure(figsize=figsize)
        for i, col in enumerate(categorical_cols, 1):
            plt.subplot(3, 3, i)
            sns.countplot(data=self.df, x=col, hue='target')
            plt.title(f'{col} Distribution')
            plt.xlabel(col)
            plt.ylabel('Count')
        plt.tight_layout()
    
    def analyze_risk_factors(self) -> pd.DataFrame:
        """Analyze key risk factors for heart disease.
        
        Returns:
            DataFrame with risk factor analysis
        """
        risk_factors = {}
        
        # Analyze age
        risk_factors['age_mean_healthy'] = self.df[self.df['target'] == 0]['age'].mean()
        risk_factors['age_mean_disease'] = self.df[self.df['target'] == 1]['age'].mean()
        
        # Analyze blood pressure
        risk_factors['bp_mean_healthy'] = self.df[self.df['target'] == 0]['restingBP'].mean()
        risk_factors['bp_mean_disease'] = self.df[self.df['target'] == 1]['restingBP'].mean()
        
        # Analyze cholesterol
        risk_factors['chol_mean_healthy'] = self.df[self.df['target'] == 0]['serumcholestrol'].mean()
        risk_factors['chol_mean_disease'] = self.df[self.df['target'] == 1]['serumcholestrol'].mean()
        
        return pd.DataFrame.from_dict(
            risk_factors,
            orient='index',
            columns=['Value']
        )
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling.
        
        Returns:
            Tuple of (X, y) where X is features DataFrame and y is target Series
        """
        # Remove ID column if present
        X = self.df.drop(['patientid', 'target'], axis=1, errors='ignore')
        y = self.df['target']
        
        return X, y 