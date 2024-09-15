# Credit Card Approval Prediction

## Authors

- [@kevint4890](https://www.github.com/kevint4890)

## Data source

- [Kaggle credit card approval prediction](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)

## Overview
This project is focused on building a machine-learning model to predict whether a person will be approved for a credit card. Using demographic, financial, and credit history information, the model classifies applicants as either approved or rejected. The data used in this project includes two datasets: 'application_record.csv' and 'credit_record.csv'.

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
  - Random Forest was selected as the final model due to its consistently high accuracy and ability to handle class imbalance effectively.

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
- Since the dataset was imbalanced (more "Good Debt" than "Bad Debt"), SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the classes.

## Features
- **Demographic Information:**
  - Gender, Income, Family Status, Education, etc.
- **Financial Information:**
  - Annual Income, Real estate ownership, Car ownership, etc.
- **Credit Status:**
  - A transformed "STATUS" field mapping different debt conditions into `Good_Debt` and `Bad_Debt`.

## Model Evaluation
Models are evaluated using several metrics:
- Accuracy: Measures how often the model is correct.
- F1 Score: Balances precision and recall, useful for handling imbalanced data.
- ROC AUC: Assesses the model's ability to distinguish between classes.

The project also provides visualizations like correlation matrices and ROC curves to better understand the performance of the models.

## Installation
To run the code in this project, you need the following Python libraries:
```bash
pip install numpy pandas statsmodels scikit-learn imbalanced-learn xgboost streamlit
```
## Usage
To run the Streamlit app for credit approval prediction, follow these steps:
  1. Ensure all required libraries are installed.
  2. Run the Streamlit app from your terminal:
```bash
streamlit run .predicton.py
```
  3. Input information through the Streamlit interface, and the model will predict whether your credit card application would be approved or rejected.

## Conclusion
This project demonstrates a successful implementation of machine learning models to predict credit card approval. The Random Forest model, in particular, showed excellent performance, achieving over 99% accuracy on both the training and test datasets. Future improvements could include fine-tuning the models further and integrating additional financial or behavioral features to enhance prediction accuracy.
