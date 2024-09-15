# Credit Card Approval Prediction

## Authors

- [@kevoslp](https://www.github.com/kevoslp)

## Data source

- [Kaggle credit card approval prediction](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)

## Overview
This project is focused on building a machine learning model to predict whether a person will be classified as having "Good Debt" or "Bad Debt" based on their application and credit history. The project uses datasets including `application_record.csv` and `credit_record.csv` and explores the relationships between demographic, financial, and credit variables.

The main models implemented are:
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**

The Random Forest Classifier and XGBoost models performed best, yielding high accuracy and F1 scores. This project also includes various data cleaning techniques, such as handling missing values and dealing with imbalanced data using SMOTE.

## Results Summary
- **Random Forest Classifier:**
  - Training Accuracy: 99.41%
  - Test Accuracy: 99.33%
  - F1 Score: 0.99
  - High performance with minimal overfitting.

- **XGBoost Classifier:**
  - Training Accuracy: 98.45%
  - Test Accuracy: 98.37%
  - F1 Score: 0.98
  - A well-balanced model that handles imbalanced data effectively.

- **Logistic Regression:**
  - Training Accuracy: 60.77%
  - Test Accuracy: 60.91%
  - F1 Score: 0.61
  - This model struggled with accuracy and balancing the classes.

## Data Preprocessing
- Merging of application and credit datasets on the `ID` field.
- Handling missing values by dropping rows with nulls.
- Label encoding of categorical variables.
- Feature scaling and variance inflation factor (VIF) analysis to detect multicollinearity.
- Addressing imbalanced classes using SMOTE.

## Features
- **Demographic and Financial Information:**
  - Gender, Income, Family Status, Education, etc.
- **Credit Status:**
  - A transformed "STATUS" field mapping different debt conditions into `Good_Debt` and `Bad_Debt`.

## Model Evaluation
Models are evaluated using several metrics:
- Accuracy
- F1 Score
- ROC AUC

The project also provides visualizations like correlation matrices and ROC curves to better understand the performance of the models.

## Installation
To run the code in this project, you need the following Python libraries:
```bash
pip install numpy pandas statsmodels scikit-learn imbalanced-learn xgboost
