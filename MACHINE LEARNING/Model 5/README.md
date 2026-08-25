# Diabetes Outcome Classification

## Project Overview

This educational machine-learning project analyzes a diabetes-labelled dataset with two numeric features and a binary `Outcome` column. It compares linear regression with logistic regression and visualizes the data and model results.

## Problem Statement

The project uses `Feature1` and `Feature2` to predict the binary `Outcome` value in the dataset.

The notebook is an educational modeling exercise. It does not document clinical variables, medical interpretation, or use as a healthcare diagnostic tool.

## Dataset

The dataset contains 120 rows and three columns:

- `Feature1`: Numeric input feature
- `Feature2`: Numeric input feature
- `Outcome`: Binary target value

All 120 rows are reported as non-null. The notebook reports an equal average of `0.5` for the binary outcome, but a full class-count table is not documented.

## Technologies Used

- Python
- Jupyter Notebook
- pandas for loading and manipulating the dataset
- NumPy for numerical calculations
- Matplotlib for scatter plots
- Seaborn for the confusion-matrix heatmap
- scikit-learn for data splitting, scaling, regression, classification, and evaluation

## Features

### Data Analysis

- Load and inspect the diabetes dataset.
- Display data types and descriptive statistics.
- Check for categorical columns.
- Explore feature and outcome relationships with scatter plots and box plots.
- Display a pair plot.

### Machine Learning

- Separate `Feature1` and `Feature2` from the `Outcome` target.
- Split the data into training and testing sets.
- Train a linear-regression model as a numeric baseline.
- Train a standardized logistic-regression classifier.
- Evaluate the linear model with MAE, MSE, RMSE, and R² score.
- Evaluate the logistic model with accuracy, a classification report, and a confusion matrix.

## Model Configuration and Results

Both experiments use an 80/20 train-test split with `random_state=42`, producing 96 training rows and 24 testing rows.

### Linear Regression Baseline

- MAE: `0.052959901543395714`
- MSE: `0.0036533658649111358`
- RMSE: `0.060443079545231114`
- R² Score: `0.9852843444881901`

### Logistic Regression Classifier

The features are standardized with `StandardScaler` before training.

- Accuracy: `1.0`
- Test confusion matrix: `[[13, 0], [0, 11]]`
- Precision, recall, and F1 score: `1.00` for both recorded classes

These results come from one small test split and should not be treated as evidence of real-world or clinical performance.

## Workflow

1. Load `diabetes.csv` with pandas.
2. Inspect the dataset and calculate descriptive statistics.
3. Check data types and categorical columns.
4. Create exploratory plots for the features and outcome.
5. Set `Feature1` and `Feature2` as inputs and `Outcome` as the target.
6. Split the data into training and testing sets with `random_state=42`.
7. Train and evaluate the linear-regression baseline.
8. Standardize the features using training-set statistics.
9. Train and evaluate the logistic-regression classifier.
10. Display the classifier confusion matrix and classification report.

## Installation and Setup

### Requirements

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

### Steps

1. Install Python and the required packages.
2. Open `model4.ipynb` in Jupyter Notebook or Visual Studio Code.
3. Run the notebook cells from top to bottom.

Exact installation commands and package versions: `TODO: Information required`

Environment variables: `TODO: Information required`

Configuration files: `TODO: Information required`

## Project Structure

```text
Model 5/
├── model4.ipynb
├── diabetes.csv
├── model4.md
├── README.md
└── model4_files/
```

- `model4.ipynb`: Main diabetes outcome analysis and modeling notebook.
- `diabetes.csv`: Source dataset.
- `model4.md`: Markdown export of the notebook.
- `README.md`: Project documentation.
- `model4_files/`: Notebook-related assets; detailed contents are not documented.

## Limitations and Missing Information

- The dataset source and attribution are not documented.
- The meanings and units of `Feature1` and `Feature2` are not documented.
- Package versions and exact installation commands are not documented.
- No license information is provided.
- Results are based on one small train-test split without cross-validation.
- The perfect logistic-regression test score may reflect the small dataset and should be independently validated.
- No clinical interpretation or diagnostic validation is provided.
