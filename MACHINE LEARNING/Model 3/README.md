# House Price Prediction

## Project Overview

This educational machine-learning project analyzes a house-price dataset and uses linear regression to predict house prices from property characteristics.

## Problem Statement

The project explores whether information such as area, bedrooms, bathrooms, floors, year built, location, condition, and garage availability can be used to estimate house prices.

## Dataset

The dataset contains 2,000 rows and these columns:

- `Id`
- `Area`
- `Bedrooms`
- `Bathrooms`
- `Floors`
- `YearBuilt`
- `Location`
- `Condition`
- `Garage`
- `Price`

The notebook reports no missing values in the dataset.

## Technologies Used

- Python
- Jupyter Notebook
- pandas for loading and inspecting the dataset
- NumPy for numerical calculations
- Matplotlib for visualizations
- Seaborn for statistical plots
- scikit-learn for data splitting, linear regression, and evaluation metrics

## Features

### Data Analysis

- Display the first rows, shape, columns, and data types.
- Generate descriptive statistics.
- Explore price, area, bathroom, bedroom, and other property variables.
- Create histograms, scatter plots, box plots, bar plots, pair plots, and a correlation heatmap.

### Machine Learning

- Convert categorical columns to numeric dummy variables using `pd.get_dummies(..., drop_first=True)`.
- Use all remaining columns except `Price` as features.
- Split data into training and testing sets.
- Train a `LinearRegression` model.
- Predict house prices.
- Evaluate predictions with MAE, MSE, RMSE, and R² score.

## Model Configuration

- Model: Linear Regression
- Target: `Price`
- Test size: 20%
- Training size: 80%
- Random state: `42`
- Categorical encoding: One-hot encoding with the first category removed

## Recorded Results

The notebook reports these test-set results:

- MAE: `242867.44926338625`
- MSE: `78279764120.86241`
- RMSE: `279785.2106900263`
- R² Score: `-0.006181784611834162`

The negative R² score indicates that this linear regression setup performs poorly on the recorded test split and does not explain the target variation effectively.

## Workflow

1. Load `House Price Prediction Dataset.csv` with pandas.
2. Inspect the dataset and calculate descriptive statistics.
3. Explore the distributions and relationships between variables.
4. Convert categorical columns into numeric dummy variables.
5. Separate the features from the `Price` target column.
6. Split the data into training and testing sets using `random_state=42`.
7. Train the linear regression model.
8. Generate price predictions for the test data.
9. Calculate MAE, MSE, RMSE, and R² score.
10. Display visualizations for further analysis.

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
2. Open `house..ipynb` in Jupyter Notebook or Visual Studio Code.
3. Run the notebook cells from top to bottom.

Exact installation commands and package versions: `TODO: Information required`

Environment variables: `TODO: Information required`

Configuration files: `TODO: Information required`

## Project Structure

```text
Model 3/
├── house..ipynb
├── House Price Prediction Dataset.csv
├── house..md
├── READmd.txt
├── README.md
└── house._files/
```

- `house..ipynb`: Main house-price analysis and machine-learning notebook.
- `House Price Prediction Dataset.csv`: Source dataset.
- `house..md`: Markdown export of the notebook.
- `READmd.txt`: Existing documentation file.
- `README.md`: Project documentation.
- `house._files/`: Notebook-related assets; detailed contents are not documented.

## Limitations and Missing Information

- The dataset source and attribution are not documented.
- Package versions and exact installation commands are not documented.
- No license information is provided.
- The recorded linear regression performance is weak.
- The notebook does not document feature scaling, cross-validation, or comparison with other models.
- The reason for including or excluding `Id` is not explained.
