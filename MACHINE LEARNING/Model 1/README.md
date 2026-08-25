# GTA Vehicle Top-Speed Regression

## Project Overview

This educational machine-learning project uses data from 50 GTA 5 vehicles to study the relationship between vehicle price and top speed. It compares a scikit-learn linear regression model with a manual ordinary least-squares implementation.

## Dataset

The source dataset contains:

- Vehicle rank
- Vehicle name
- Manufacturer
- Vehicle class
- Price in GTA dollars
- Top speed in mph
- Lap time in seconds
- DLC or release update

The notebook cleans the price values by removing the dollar sign and commas. Rows missing price or top-speed values are removed. Lap time and categorical columns are retained for exploration but are not used in the regression model.

## Model

- Feature: `Price (GTA$)`
- Target: `Top Speed (mph)`
- Algorithm: Simple linear regression
- Training/testing split: 80% / 20%
- Random state: `42`
- Training rows: 40
- Testing rows: 10

The model is implemented in two ways:

1. With `sklearn.linear_model.LinearRegression`.
2. Manually, by calculating the slope and intercept with ordinary least-squares equations.

Both implementations produce approximately the following results:

- Slope: `0.000015`
- Intercept: `93.1591`
- Test MSE: `1841.3470`

## Features

### Core Features

- Explore the GTA vehicle dataset.
- Clean price data for numerical analysis.
- Train a simple linear regression model.
- Calculate predictions with scikit-learn.
- Calculate predictions manually.
- Compare the two implementations using mean squared error.
- Visualize vehicle prices, top speeds, and the regression line.

### Additional Outputs

- Export actual and predicted test-set speeds to `gta_vehicle_regression_results.csv`.
- Export the notebook analysis to `GTA.md`.

## Project Workflow

1. Load `GTA 5 Vehicle Data Export.csv` with pandas.
2. Inspect the dataset and its descriptive statistics.
3. Convert the formatted price column to numeric values.
4. Remove rows missing price or top-speed values.
5. Set vehicle price as the input feature and top speed as the target.
6. Split the data into training and testing sets with `random_state=42`.
7. Train the scikit-learn linear regression model.
8. Calculate a second model manually using ordinary least-squares equations.
9. Predict top speeds for the test set with both implementations.
10. Compare their mean squared error values.
11. Save the predictions and display a regression plot.

## Installation and Setup

### Requirements

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Visual Studio Code is mentioned as an option for opening the notebook.

### Steps

1. Install Python with Jupyter, pandas, NumPy, Matplotlib, and scikit-learn.
2. Open `GTA.ipynb` in Jupyter Notebook or Visual Studio Code.
3. Run the notebook cells from top to bottom.

### Configuration

Environment variables: `TODO: Information required`

Configuration files: `TODO: Information required`

Exact installation commands: `TODO: Information required`

## Project Structure

```text
Model 1/
├── GTA.ipynb
├── GTA 5 Vehicle Data Export.csv
├── gta_vehicle_regression_results.csv
├── GTA.md
├── READmd.txt
├── README.md
└── GTA_files/
```

- `GTA.ipynb`: Main Jupyter Notebook containing exploration, preprocessing, regression, evaluation, and visualization.
- `GTA 5 Vehicle Data Export.csv`: Source vehicle dataset.
- `gta_vehicle_regression_results.csv`: Actual and predicted top speeds from the test set.
- `GTA.md`: Markdown export of the notebook.
- `READmd.txt`: Existing project documentation.
- `README.md`: This project README.
- `GTA_files/`: Notebook-related folder; its detailed contents are not documented.

## Limitations

This is a small educational dataset. Price alone may not accurately predict top speed, especially because the data includes different vehicle types such as cars, motorcycles, aircraft, and watercraft.

## Missing Information

- Python and package version requirements.
- Exact installation commands.
- Environment variables and configuration details.
- Dataset source or attribution.
- License information.
- Detailed contents of `GTA_files/`.
- A more complete evaluation such as additional metrics or validation methods.
