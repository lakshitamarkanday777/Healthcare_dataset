# Diabetes Prediction Project

## Overview
This project analyzes a diabetes dataset to identify patterns and builds machine learning models to predict diabetes status (diabetic, non-diabetic, or prediabetic) based on patient demographics and laboratory measurements.

## Dataset
The dataset contains medical and demographic information including:
- Patient identifiers (ID, No_Pation)
- Demographics (Age, Gender, Age Range)
- Laboratory measurements (HbA1c, Urea, Creatinine, Cholesterol, Triglycerides, HDL, LDL, VLDL)
- Body metrics (BMI)
- Diabetes status (CLASS: Y-diabetic, N-non-diabetic, P-prediabetic)

## Exploratory Data Analysis (EDA)
The EDA reveals several key insights:

1. **Diabetes Distribution**: The dataset has a high representation of diabetic patients (>800 cases).

2. **Age-Related Patterns**: Diabetes prevalence peaks in the 60-70 age range, followed by the 70-80 age range. Younger age groups show more balanced distributions.

3. **Gender Analysis**:
   - HbA1c levels are similar between genders
   - Males show slightly higher urea levels and more extreme outliers
   - Cholesterol distributions are similar across genders
   - Males have slightly higher median BMI values

4. **Correlations**:
   - Strong positive correlations between:
     - Urea and Creatinine (0.62)
     - HbA1c and BMI (0.41)
     - Cholesterol and LDL (0.42)
     - HbA1c and Age (0.38)
   - HDL shows negative correlations with LDL (-0.14) and VLDL (-0.059)

5. **Outlier Analysis**: Extreme outliers were observed in the Creatinine measurements, with some values reaching ~800 (normal range is 30-80).

6. **Feature Distributions**:
   - BMI shows a bimodal distribution with peaks around 30 and 35 kg/m²
   - Most metrics show right-skewed distributions
   - HbA1c values range from 2.5% to 16%, centered around 8-9%

## Machine Learning Modeling

### Preprocessing
1. **Data Splitting**: Split data into training (80%) and testing (20%) sets
2. **Categorical Encoding**: Used Label Encoding for categorical variables (Gender, Age Range)
3. **Feature Scaling**: Applied MinMaxScaler to normalize numerical features

### Models Evaluated
- Logistic Regression (LR)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (NB)
- Support Vector Classifier (SVC)
- Random Forest Classifier (RFC)
- Decision Tree Regressor (DTR)
- XGBoost Classifier (XGB)

### Evaluation Methods
- Initial accuracy assessment on test set
- 5-fold cross-validation
- Classification reports (precision, recall, F1-score)
- Confusion matrix visualization

### Results
XGBoost (XGB) was identified as the best-performing model based on cross-validation results.

## Technologies Used
- Python
- pandas, numpy
- matplotlib, seaborn (for visualization)
- scikit-learn (for ML models and preprocessing)
- XGBoost

## Future Work
Potential improvements include:
- Hyperparameter tuning
- Feature engineering
- Addressing class imbalance
- Exploring more advanced models
- Medical interpretation of the model's decisions

## Usage
1. Clone the repository
2. Install required dependencies
3. Run the Jupyter notebook to reproduce the analysis and models

