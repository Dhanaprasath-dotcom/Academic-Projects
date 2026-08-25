# Salary Data Analysis

## Project Overview

This educational data-analysis project explores a salary dataset containing information about age, gender, education level, job title, years of experience, and salary. The notebook performs data inspection, missing-value analysis, descriptive analysis, correlation analysis, and visual exploration.

## Dataset

The dataset contains 375 rows and these columns:

- `Age`
- `Gender`
- `Education Level`
- `Job Title`
- `Years of Experience`
- `Salary`

The notebook reports two missing values in each column. Numeric summaries include age, years of experience, and salary.

## Features

- Load the salary CSV file with pandas.
- Inspect rows, columns, shape, data types, and descriptive statistics.
- Count missing values and display a missing-values heatmap.
- Explore value counts for categorical and discrete columns.
- Analyze numeric correlations.
- Identify highly correlated columns using a threshold greater than `0.90`.
- Create scatter plots, box plots, count plots, and a correlation heatmap.
- Prepare feature and target variables in one notebook section using `Education Level` as the target.

The notebook does not document a completed machine-learning training or evaluation step.

## Technologies Used

- Python
- Jupyter Notebook
- pandas for loading, inspecting, grouping, and transforming data
- NumPy for numerical and matrix operations
- Matplotlib for plotting
- Seaborn for statistical visualizations

## Workflow

1. Load `salary data.csv` into a pandas DataFrame.
2. Inspect the dataset structure and statistical summary.
3. Check for missing values and visualize them.
4. Explore distributions and value counts.
5. Examine relationships between salary, experience, age, gender, and education level.
6. Calculate the numeric correlation matrix.
7. Identify highly correlated columns using a `0.90` threshold.
8. Create feature and target variables in the notebook's preparation section.

## Installation and Setup

### Requirements

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn

### Steps

1. Install Python and the required packages.
2. Open `salarydata.ipynb` in Jupyter Notebook or Visual Studio Code.
3. Run the notebook cells from top to bottom.

Exact installation commands and package versions: `TODO: Information required`

Environment variables: `TODO: Information required`

Configuration files: `TODO: Information required`

## Project Structure

```text
Model 2/
├── salarydata.ipynb
├── salary data.csv
├── salarydata.md
├── READmd.txt
├── README.md
└── salarydata_files/
```

- `salarydata.ipynb`: Main notebook for salary data exploration.
- `salary data.csv`: Source dataset.
- `salarydata.md`: Markdown export of the notebook.
- `READmd.txt`: Existing project documentation file.
- `README.md`: Project documentation.
- `salarydata_files/`: Notebook-related assets; detailed contents are not documented.

## Limitations and Missing Information

- The dataset source and attribution are not documented.
- Python and dependency versions are not documented.
- Exact installation commands are not documented.
- No license information is provided.
- The notebook does not show a completed model-training or model-evaluation workflow.
- The treatment of missing values is not documented as a completed preprocessing step.
- The rationale for selecting `Education Level` as the target is not explained.
