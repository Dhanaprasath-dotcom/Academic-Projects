# IBM Employee Dataset - Exploratory Data Analysis

### 1. Project Summary

This project performs Exploratory Data Analysis (EDA) on the IBM Employee dataset using R in a Jupyter Notebook. It studies employee characteristics, salary patterns, satisfaction levels, work-life balance, and employee attrition.

### 2. Problem Statement

Organizations need to understand employee attrition and identify patterns associated with age, department, salary, job satisfaction, education, and work-life balance.

### 3. Target Users

- Students learning R and data analysis
- Data analysts
- Human resource professionals
- Researchers studying employee attrition

### 4. Technologies Used

- R
- Jupyter Notebook
- IRkernel
- `ggplot2`
- `dplyr`
- CSV dataset

### 5. Features

- Load and inspect the employee dataset
- Check missing values and duplicate records
- Convert categorical columns to factors
- Calculate descriptive statistics
- Create histograms and bar plots
- Detect outliers with box plots
- Compare attrition across categorical variables
- Analyze numerical relationships with scatter plots
- Calculate a correlation matrix
- Summarize salary, tenure, department, and attrition patterns

### 6. System Workflow

1. Load `IBM.csv` into R.
2. Inspect the dataset structure, dimensions, and column names.
3. Check for missing values and duplicate records.
4. Convert character columns to factors.
5. Calculate mean, median, standard deviation, range, and quartiles.
6. Generate univariate charts and outlier plots.
7. Compare attrition with department, job satisfaction, marital status, and other variables.
8. Examine relationships between age, income, and years at the company.
9. Calculate correlations and summary tables.

### 7. Installation & Setup

Install R, Jupyter Notebook, the R kernel, and the required packages:

```r
install.packages("IRkernel")
IRkernel::installspec()
install.packages(c("ggplot2", "dplyr"))
```

Open `IBM_EDA.ipynb` in Jupyter Notebook or Visual Studio Code, select an R kernel, and keep `IBM.csv` in the same directory as the notebook. Run the cells from top to bottom.

The dataset is loaded with:

```r
data <- read.csv("IBM.csv")
```

### 8. Project Structure

```text
Model 2/
|-- IBM_EDA.ipynb    # Main R EDA notebook
|-- IBM.csv          # IBM employee dataset
|-- README.md        # Project documentation
|-- READmd.txt       # Older README file name
```

### 9. Missing Information

- Dataset source, size, and license are not documented.
- Final analytical findings and conclusions are not included.
- Variable coding, such as education and satisfaction scores, is not explained.
- R and package versions are not specified.
- No screenshots or sample outputs are provided.
- Limitations and future improvements are not documented.

### 10. Problems in Current README

- The existing `READmd.txt` file was empty.
- The filename is unconventional; `README.md` is the standard name.
- Installation commands and version requirements were missing.
- The project objectives and research questions were not clearly stated.
- Results, conclusions, and dataset metadata were missing.

### 11. Improvement Suggestions

- Add the dataset source, dimensions, license, and variable descriptions.
- Document key findings with supporting charts or tables.
- Explain the meaning of coded education and satisfaction values.
- Correct misleading labels in notebook comments and chart descriptions.
- Add limitations and possible future analysis.
- Specify supported R and package versions.
- Add screenshots of the most important visualizations.

### 12. README Quality Score

**8/10**

This README now documents the project purpose, users, technologies, workflow, setup, structure, limitations, and improvement areas. The score can be improved after the notebook's final findings, dataset source, variable definitions, and version information are added.
