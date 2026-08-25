# Employee Attrition Analysis

## Project Overview

This educational machine-learning project analyzes employee information and experiments with predicting employee attrition. It also includes a linear-regression baseline that predicts the binary `OverTime` field.

## Problem Statement

The project explores employee data to understand workforce characteristics and classify whether an employee may leave the company (`Attrition`).

## Dataset

`Employee.csv` contains 1,470 rows and 35 columns. The dataset includes employee demographics, work conditions, satisfaction measures, compensation information, travel details, and company tenure.

Important fields include:

- `Age`
- `Attrition`
- `BusinessTravel`
- `Department`
- `DistanceFromHome`
- `Education`
- `EducationField`
- `JobRole`
- `JobSatisfaction`
- `MonthlyIncome`
- `OverTime`
- `PerformanceRating`
- `TotalWorkingYears`
- `YearsAtCompany`
- `YearsInCurrentRole`
- `YearsSinceLastPromotion`
- `YearsWithCurrManager`

The notebook reports no missing values in the dataset.

## Technologies Used

- Python
- Jupyter Notebook
- pandas for loading and manipulating employee data
- NumPy for numerical calculations
- Matplotlib for plots
- Seaborn for statistical visualizations and the confusion matrix heatmap
- scikit-learn for data splitting, encoding, scaling, regression, classification, and evaluation

## Features

### Data Analysis

- Inspect rows, shape, columns, data types, and descriptive statistics.
- Identify numeric and categorical columns.
- Explore employee data with visualizations.
- Analyze relationships and distributions.

### Linear Regression Experiment

- Convert `OverTime` from `Yes`/`No` to `1`/`0`.
- One-hot encode categorical columns.
- Predict `OverTime` with `LinearRegression`.
- Evaluate predictions with MAE, MSE, RMSE, and R² score.
- Plot actual versus predicted values.

### Logistic Regression Experiment

- Convert `Attrition` and `OverTime` from text labels to binary values.
- Remove `EmployeeNumber`, `EmployeeCount`, and `StandardHours`.
- One-hot encode the remaining categorical columns.
- Standardize the features with `StandardScaler`.
- Predict `Attrition` with `LogisticRegression`.
- Report accuracy, precision, recall, and F1 score.
- Display a confusion matrix.

## Model Configuration and Results

Both experiments use an 80/20 train-test split with `random_state=42`.

### Linear Regression for OverTime

- MAE: `0.37580048635661617`
- MSE: `0.2037515976257438`
- RMSE: `0.45138852181434985`
- R² Score: `-0.05401119710208824`

This is a baseline experiment for a binary target. Its negative R² score indicates weak performance on the recorded test split.

### Logistic Regression for Attrition

- Accuracy: `0.8775510204081632`
- Class `0` F1 score: `0.93`
- Class `1` F1 score: `0.49`
- Class `1` recall: `0.44`

The accuracy is 87.76%, but the lower recall and F1 score for the attrition class show that the model identifies employees who leave less effectively than employees who do not leave.

## Workflow

1. Load `Employee.csv` with pandas.
2. Inspect the dataset and calculate descriptive statistics.
3. Analyze employee columns and create visualizations.
4. Convert categorical target values to binary values where required.
5. One-hot encode categorical features.
6. Remove selected identifier or constant columns for the attrition experiment.
7. Split the data into training and testing sets.
8. Train the linear-regression baseline for `OverTime`.
9. Train the scaled logistic-regression classifier for `Attrition`.
10. Generate predictions and calculate evaluation metrics.
11. Display comparison plots and a confusion matrix.

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
2. Open `Employe.ipynb` in Jupyter Notebook or Visual Studio Code.
3. Run the notebook cells from top to bottom.

Exact installation commands and package versions: `TODO: Information required`

Environment variables: `TODO: Information required`

Configuration files: `TODO: Information required`

## Project Structure

```text
Model 4/
├── Employe.ipynb
├── Employee.csv
├── Employe.md
├── README.md
└── Employe_files/
```

- `Employe.ipynb`: Main employee-analysis and machine-learning notebook.
- `Employee.csv`: Source employee dataset.
- `Employe.md`: Markdown export of the notebook.
- `README.md`: Project documentation.
- `Employe_files/`: Notebook-related assets; detailed contents are not documented.

## Limitations and Missing Information

- The dataset source and attribution are not documented.
- Package versions and exact installation commands are not documented.
- No license information is provided.
- The linear regression experiment uses a binary target and is not the preferred classification approach.
- The attrition classes are imbalanced in the recorded test results.
- No cross-validation, hyperparameter tuning, or alternative classifier comparison is documented.
- The notebook does not document how the model should be used with new employee records.
