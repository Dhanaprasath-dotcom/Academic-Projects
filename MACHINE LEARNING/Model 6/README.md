# Medical Result Classification

## Project Overview

This educational machine-learning project explores a medical dataset containing patient measurements and a binary `Result` label. It performs exploratory data analysis and experiments with linear regression, logistic regression, and K-nearest neighbors (KNN).

This documentation describes the notebook workflow only. It does not establish clinical validity or provide medical advice.

## Problem Statement

The main classification task uses patient measurements to predict whether the recorded `Result` is `positive` or `negative`.

The notebook also includes regression and KNN experiments for learning purposes.

## Dataset

`Medicaldataset.csv` contains 1,319 rows and nine columns:

- `Age`
- `Gender`
- `Heart rate`
- `Systolic blood pressure`
- `Diastolic blood pressure`
- `Blood sugar`
- `CK-MB`
- `Troponin`
- `Result`

The notebook reports all 1,319 values as non-null. `Result` is a text column with `positive` and `negative` labels. The meaning, units, and encoding of some measurements are not documented.

## Technologies Used

- Python
- Jupyter Notebook
- pandas for loading and inspecting the dataset
- NumPy for numerical calculations
- Matplotlib for plots
- Seaborn for statistical visualizations and confusion matrices
- scikit-learn for splitting, scaling, regression, classification, and evaluation

## Features

### Data Analysis

- Load and inspect the medical dataset.
- Display data types and descriptive statistics.
- Check for missing values.
- Explore distributions and relationships with plots.
- Create scatter plots, box plots, pair plots, and heatmaps.

### Linear Regression Experiment

- Convert categorical data with one-hot encoding.
- Predict `Age` from the remaining columns.
- Evaluate predictions with MAE, MSE, RMSE, and R² score.
- Plot actual versus predicted age.

### Logistic Regression Experiment

- Use all columns except `Result` as features.
- Standardize features using `StandardScaler`.
- Predict positive or negative `Result` values.
- Evaluate with accuracy, a classification report, and a confusion matrix.

### KNN Experiment

- Use `KNeighborsClassifier` with `n_neighbors=5`.
- Standardize the features.
- Generate classification predictions and evaluation outputs.

## Model Configuration and Results

The documented experiments use an 80/20 train-test split with `random_state=42`, producing 1,055 training rows and 264 test rows.

### Linear Regression for Age

- MAE: `10.657162660472808`
- MSE: `178.00472608350964`
- RMSE: `13.341841180418452`
- R² Score: `0.05819549979926175`

### Logistic Regression for Result

- Accuracy: `80.68181818181817%`
- Negative class: precision `0.77`, recall `0.71`, F1 `0.74`
- Positive class: precision `0.83`, recall `0.87`, F1 `0.85`
- Confusion matrix: `[[72, 29], [22, 141]]`

### KNN Experiment

- Number of neighbors: `5`
- Recorded accuracy: `0.04924242424242424`

The KNN section uses `Age` as the target while applying a classifier to many distinct age values. The notebook also reports undefined-metric warnings, so this result should not be treated as a valid medical classification result.

## Workflow

1. Load `Medicaldataset.csv` with pandas.
2. Inspect the dataset, data types, and descriptive statistics.
3. Check for missing values.
4. Explore the measurements and labels with visualizations.
5. Encode categorical data where required.
6. Split the data into training and testing sets.
7. Train the linear-regression experiment for `Age`.
8. Standardize features and train logistic regression for `Result`.
9. Train the KNN experiment with five neighbors.
10. Calculate evaluation metrics and display plots.

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
2. Open `Medical.ipynb` in Jupyter Notebook or Visual Studio Code.
3. Run the notebook cells from top to bottom.

Exact installation commands and package versions: `TODO: Information required`

Environment variables: `TODO: Information required`

Configuration files: `TODO: Information required`

## Project Structure

```text
Model 6/
├── Medical.ipynb
├── Medicaldataset.csv
├── Medical.md
├── README.md
└── Medical_files/
```

- `Medical.ipynb`: Main medical-data analysis and modeling notebook.
- `Medicaldataset.csv`: Source dataset.
- `Medical.md`: Markdown export of the notebook.
- `README.md`: Project documentation.
- `Medical_files/`: Notebook-related assets; detailed contents are not documented.

## Limitations and Missing Information

- The dataset source and attribution are not documented.
- Measurement units and the meaning of encoded values are not documented.
- Package versions and exact installation commands are not documented.
- No license information is provided.
- Results are based on one train-test split without cross-validation.
- The notebook does not establish clinical or diagnostic performance.
- The KNN experiment has a target-type mismatch and should be corrected before interpretation.
- The dataset's collection method and population are not documented.
