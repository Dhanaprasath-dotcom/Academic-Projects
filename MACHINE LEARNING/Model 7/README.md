# Titanic Data Analysis

## Project Overview

This educational data-analysis and machine-learning project explores the Titanic passenger dataset. It covers data inspection, missing-value handling, exploratory visualizations, categorical encoding, and a model-building experiment.

## Problem Statement

The project analyzes passenger information such as class, sex, age, fare, family relationships, ticket, cabin, and embarkation port. The notebook includes a model-building section using the prepared passenger data.

## Dataset

`Titanic-Dataset.csv` contains 891 passenger records and 12 columns:

- `PassengerId`
- `Survived`
- `Pclass`
- `Name`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Ticket`
- `Fare`
- `Cabin`
- `Embarked`

The initial missing-value check reports 177 missing `Age` values, 687 missing `Cabin` values, and 2 missing `Embarked` values.

## Technologies Used

- Python
- Jupyter Notebook
- pandas for loading, inspecting, cleaning, and transforming data
- NumPy for numerical calculations
- Matplotlib for visualizations
- Seaborn for statistical plots
- scikit-learn for preprocessing, data splitting, regression, and evaluation metrics

## Features

### Data Preprocessing

- Load the Titanic CSV dataset.
- Inspect rows, columns, shape, data types, and descriptive statistics.
- Check missing values.
- Fill missing `Age` values during exploratory preprocessing examples.
- Fill missing `Embarked` values with the mode in a preprocessing example.
- Demonstrate dropping rows with missing values.
- Remove rows with missing `Age` in the model-building section.
- Convert categorical columns to numeric dummy variables with `drop_first=True`.

### Exploratory Analysis

- Analyze passenger and survival-related fields.
- Generate plots for distributions and relationships in the dataset.

### Model-Building Experiment

- Split the prepared data into training and testing sets.
- Train a `LinearRegression` model using the model-building code shown in the notebook.
- Generate predictions.
- Calculate MAE, MSE, RMSE, R² score, accuracy, a confusion matrix, and a classification report.

## Recorded Model Results

The notebook records these results from its model-building section:

- MAE: `0.21678321678321677`
- MSE: `0.21678321678321677`
- RMSE: `0.46559984620188266`
- R² Score: `0.09010673234811162`
- Accuracy: `0.7832167832167832`
- Confusion matrix: `[[73, 14], [17, 39]]`

The classification report records:

- Class `0`: precision `0.81`, recall `0.84`, F1 `0.82`
- Class `1`: precision `0.74`, recall `0.70`, F1 `0.72`

### Important Model Note

The exported notebook shows `Age` as the target for the `LinearRegression` code, while the recorded accuracy, confusion matrix, and classification report appear to evaluate binary values such as survival. This target-and-metric mismatch should be corrected and re-run before interpreting the model results.

## Workflow

1. Load `Titanic-Dataset.csv` with pandas.
2. Inspect the dataset and its descriptive statistics.
3. Identify missing values in `Age`, `Cabin`, and `Embarked`.
4. Demonstrate missing-value filling and row removal.
5. Remove rows with missing `Age` for the model-building section.
6. One-hot encode categorical columns.
7. Define features and target according to the selected modeling task.
8. Split the data into training and testing sets.
9. Train the linear-regression experiment.
10. Generate predictions and calculate evaluation outputs.
11. Review the visualizations and verify that the target matches the metrics being used.

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
2. Open `Titanic.ipynb` in Jupyter Notebook or Visual Studio Code.
3. Run the notebook cells from top to bottom.

Exact installation commands and package versions: `TODO: Information required`

Environment variables: `TODO: Information required`

Configuration files: `TODO: Information required`

## Project Structure

```text
Model 7/
├── Titanic.ipynb
├── Titanic-Dataset.csv
├── Titanic.md
├── README.md
└── Titanic_files/
```

- `Titanic.ipynb`: Main Titanic analysis and modeling notebook.
- `Titanic-Dataset.csv`: Source dataset.
- `Titanic.md`: Markdown export of the notebook.
- `README.md`: Project documentation.
- `Titanic_files/`: Notebook-related assets; detailed contents are not documented.

## Limitations and Missing Information

- The dataset source and attribution are not documented.
- The notebook does not document package versions or exact installation commands.
- No license information is provided.
- The treatment of `Cabin` missing values is not clearly defined for the final model.
- The model target and reported evaluation metrics are inconsistent in the exported notebook.
- A classification model such as logistic regression is not documented for the binary `Survived` target.
- The results are based on one train-test split without documented cross-validation.
