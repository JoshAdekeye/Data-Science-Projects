# Bank Marketing Campaign Analysis: Term Deposit Prediction

## Project Description

This project analyzes a Portuguese banking institution's direct marketing campaigns (phone calls) to predict whether clients will subscribe to a term deposit. The goal is to help the bank optimize its marketing strategies by identifying potential customers most likely to subscribe, thereby increasing campaign efficiency and reducing costs. By leveraging machine learning techniques, we can uncover patterns and factors that influence customers' decisions to subscribe to term deposits.

## Results

### Model Performance

- **Accuracy**: 89.7% overall prediction accuracy
- **Precision for "Yes" class**: 67.8% (When our model predicts a client will subscribe, it's correct about 68% of the time)
- **Recall for "Yes" class**: 43.2% (Our model identifies 43% of all clients who would subscribe)
- **F1-Score for "Yes" class**: 52.6% (Harmonic mean of precision and recall)
- **AUC-ROC**: 0.91 (Excellent discriminatory ability between classes)

### Key Insights

- The imbalanced nature of the dataset (only 11.7% of clients subscribed) makes recall a critical metric
- The model provides significant improvement over random selection (which would yield only 11.7% success rate)
- Using the model could potentially reduce marketing costs by targeting only high-probability clients

## Visual Summary

The analysis includes extensive data visualization to uncover patterns:

- **Correlation Matrix**: Revealed strong relationships between economic indicators and subscription likelihood
- **Distribution Analysis**: Showed higher subscription rates among certain demographics (education level, job types)
- **Feature Importance Plots**: Visualized the most influential predictors for term deposit subscription
- **Confusion Matrix**: Illustrated model performance with focus on minimizing false negatives
- **ROC & Precision-Recall Curves**: Demonstrated model's discriminative power despite class imbalance

## Model Selection and Methodology

### Models Evaluated

- **Logistic Regression**: Selected as primary model for its interpretability and good performance
- **Decision Tree**: Used for feature importance analysis
- **Random Forest**: Provided robust performance with ensemble learning

### Why Logistic Regression?

We selected logistic regression as our final model because:

1. It provides excellent interpretability, allowing us to understand the factors affecting subscription decisions
2. It achieved good performance metrics (AUC-ROC of 0.91)
3. It's less prone to overfitting compared to more complex models
4. It offers probability outputs that can be used to rank potential customers

### Methodology

1. **Data Preprocessing**
   - Handled missing values
   - Encoded categorical variables
   - Feature scaling
   - Removed high-correlation features (e.g., "duration" was excluded as it's only known after a call)
2. **Exploratory Data Analysis**
   - Identified key demographic patterns
   - Analyzed campaign timing effects
   - Examined economic indicators' impact
   - Used correlation analysis to identify relationships between features
3. **Model Development and Training**
   - Split data (70% training, 30% testing)
   - Hyperparameter tuning with cross-validation
   - Addressed class imbalance with appropriate techniques including class weights and stratified sampling
4. **Model Evaluation**
   - Used confusion matrix and classification metrics
   - Performed ROC and precision-recall analysis
   - Evaluated model robustness with cross-validation
5. **Feature Importance Analysis**
   - Identified most influential factors for subscription decisions
   - Used Decision Tree visualization to explain feature relationships

## Conclusions and Recommendations

### Key Influencing Factors

1. **Contact method**: Clients contacted via cellular phone are more likely to subscribe
2. **Campaign timing**: Contacts made in March, September, October, and December yield higher success rates
3. **Previous outcome**: Previous success strongly predicts future subscription
4. **Economic indicators**: Employment variation rate and consumer confidence index significantly impact decisions
5. **Age and education**: Higher education level correlates with subscription likelihood

### Recommended Actions

1. **Target optimization**: Focus campaigns on clients with higher subscription probability
2. **Timing strategy**: Schedule campaigns during months with historically higher success rates
3. **Contact method**: Prioritize cellular phone contact when possible
4. **Follow-up strategy**: Implement specialized approach for previously successful clients
5. **Economic monitoring**: Adjust campaign intensity based on current economic indicators

### Business Impact

Implementing these recommendations can potentially:

- Increase campaign success rate by up to 4x compared to random selection
- Reduce marketing costs by focusing efforts on high-probability clients
- Improve customer experience by reducing unwanted solicitations
- Provide a data-driven framework for future campaign optimization

## Dataset Description

The dataset contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit.

### Features Include:

- **Client demographics**: age, job, marital status, education
- **Campaign information**: contact type, month, day of week, duration
- **Economic indicators**: employment variation rate, consumer price index
- **Previous campaign data**: outcome, number of contacts

### Data Challenges

- **Class Imbalance**: Only 11.7% of clients subscribed to term deposits
- **Temporal Dependencies**: Economic indicators vary over time and impact subscription rates
- **Feature Correlation**: Some features like 'duration' are highly predictive but not available before making calls
- **Multicollinearity**: Several economic indicators showed high correlation

## Technologies Used

- **Python**: Primary programming language
- **Jupyter Notebook**: Development environment
- **Key Libraries**:
  - pandas & numpy: Data manipulation and analysis
  - scikit-learn: Machine learning algorithms and evaluation metrics
  - matplotlib & seaborn: Data visualization
  - imbalanced-learn: Handling class imbalance
  - plotly: Interactive visualizations for exploring feature relationships

## Project Structure

```
├── bank-marketing-campaign-opening-a-term-deposit.ipynb
└── README.md
```

## Setup and Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook bank-marketing-campaign-opening-a-term-deposit.ipynb
   ```

## Future Improvements

- Implement more advanced models (XGBoost, Neural Networks) for comparison
- Develop a cost-benefit analysis framework to optimize marketing ROI
- Explore additional feature engineering opportunities
- Create a deployable prediction API for real-time decision making
- Incorporate customer lifetime value in targeting strategy
- Develop a time-series analysis to better predict optimal campaign timing
