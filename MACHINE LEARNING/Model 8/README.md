# Traffic Data Analysis

## Project Overview

This educational data-analysis and machine-learning project explores traffic counts recorded at road junctions. It includes exploratory analysis, regression, binary traffic-level classification, and K-Means clustering.

## Problem Statement

The project analyzes traffic volume by date and junction, then experiments with predicting vehicle counts and classifying traffic levels based on whether the vehicle count exceeds 20.

## Dataset

`traffic.csv` contains 48,120 records and four columns:

- `DateTime`: Recorded date and time
- `Junction`: Junction identifier
- `Vehicles`: Vehicle count
- `ID`: Record identifier

The notebook reads the `DateTime` column as an object and the other three columns as integers. The documented dataset summary reports 48,120 non-null values in every column.

## Technologies Used

- Python
- Jupyter Notebook
- pandas for loading, inspecting, and transforming traffic data
- NumPy for numerical calculations
- Matplotlib for plots
- Seaborn for statistical visualizations and confusion matrices
- scikit-learn for splitting, scaling, regression, classification, and clustering

## Features

### Data Analysis

- Load and inspect the traffic dataset.
- Review shape, columns, data types, and descriptive statistics.
- Calculate value counts for junctions and vehicle counts.
- Explore traffic patterns with plots and categorical comparisons.

### Linear Regression

- Use `Junction` as the feature and `Vehicles` as the target.
- Standardize the feature with `StandardScaler`.
- Train a `LinearRegression` model.
- Evaluate with mean squared error and R² score.
- Display actual data and the regression line.

### Logistic Regression

- Create a binary `Result` target using `Vehicles > 20`.
- Use `Junction` and `Vehicles` as features.
- Standardize the features.
- Train a `LogisticRegression` classifier.
- Evaluate with accuracy, a classification report, and a confusion matrix.

### KNN Classification

- Create the same binary traffic-level target.
- Standardize `Junction` and `Vehicles`.
- Train a `KNeighborsClassifier` with `n_neighbors=5`.
- Evaluate with accuracy, a classification report, and a confusion matrix.

### K-Means Clustering

- Standardize `Junction` and `Vehicles`.
- Create three clusters with `KMeans(n_clusters=3, random_state=42)`.
- Add cluster labels to the DataFrame.
- Attempt to visualize the clusters.

## Model Configuration and Results

The documented supervised-learning experiments use an 80/20 train-test split with `random_state=42`, producing 38,496 training rows and 9,624 test rows.

### Linear Regression

- Feature: `Junction`
- Target: `Vehicles`
- Mean squared error: `254.93967005535174`
- R² score: `0.3744529336905429`

### Logistic Regression

- Target: `Result = (Vehicles > 20).astype(int)`
- Recorded accuracy: `100.0%`
- Confusion matrix: `[[6171, 0], [0, 3453]]`

### KNN Classification

- Neighbors: `5`
- Recorded accuracy: `100.0%`
- Confusion matrix: `[[6171, 0], [0, 3453]]`

The perfect classification results are not independent evidence of prediction quality because `Vehicles`, the value used to create `Result`, is also included as an input feature.

## Workflow

1. Load `traffic.csv` with pandas.
2. Inspect the dataset and calculate summaries.
3. Explore junctions, vehicle counts, and traffic relationships.
4. Standardize selected numeric features where required.
5. Train linear regression to estimate vehicle counts from junction identifiers.
6. Evaluate the regression output with MSE and R².
7. Create a binary traffic-level target from vehicle counts.
8. Train and evaluate logistic regression and KNN classifiers.
9. Scale junction and vehicle features for K-Means clustering.
10. Add cluster labels and attempt a cluster visualization.

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
2. Open `Traffic.ipynb` in Jupyter Notebook or Visual Studio Code.
3. Run the notebook cells from top to bottom.

Exact installation commands and package versions: `TODO: Information required`

Environment variables: `TODO: Information required`

Configuration files: `TODO: Information required`

## Project Structure

```text
Model 8/
├── Traffic.ipynb
├── traffic.csv
├── Traffic.md
├── README.md
└── Traffic_files/
```

- `Traffic.ipynb`: Main traffic analysis and modeling notebook.
- `traffic.csv`: Source traffic dataset.
- `Traffic.md`: Markdown export of the notebook.
- `README.md`: Project documentation.
- `Traffic_files/`: Notebook-related assets; detailed contents are not documented.

## Limitations and Missing Information

- The dataset source and attribution are not documented.
- Package versions and exact installation commands are not documented.
- No license information is provided.
- `DateTime` is not documented as being converted into date or time features.
- The classification target is derived from `Vehicles`, which is also used as an input feature, causing target leakage.
- The reported perfect classification scores should not be interpreted as generalization performance.
- The K-Means plotting expression appears to use invalid multi-column indexing and may fail when executed.
- The clustering results do not include documented cluster-quality metrics.
