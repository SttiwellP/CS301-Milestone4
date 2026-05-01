# Urban Data Science Project

## Project Goal

This project analyzes the relationship between urban air quality, environmental health risks, and natural disaster declarations across the United States.

The objective is to analyze the data and determine whether there is correlation between them, and if there is to describe how we know and what this means.

---

## Datasets Used

### 1. Urban Air Quality and Health Impact Dataset

Contains:
- temperature
- humidity
- precipitation
- heat index
- severity score
- health risk score

### 2. US Natural Disaster Declarations Dataset

Contains:
- declaration dates
- disaster types
- state
- region
- incident types

---

## Project Steps

### Step 1: Exploratory Data Analysis

- Feature distributions
- Correlation matrix
- Bivariate analysis

### Step 2: Hypothesis Testing

- Statistical significance testing
- p-value reporting

### Step 3: Model Building

- Regression / Classification / Clustering
- Model evaluation using RMSE, R², or Confusion Matrix

### Step 4: Knowledge Discovery

- Actionable insights
- Feature importance
- Urban policy recommendations

---

## Docker Bonus

This project is containerized using Docker for reproducibility.

### Build Docker Image

docker build -t urban-project .

### Run Container

docker run -v ${PWD}/output:/app/output urban-project

Generated visualizations will be saved in the /output folder.

---

## Team Members

- Sttiwell
- Pratik
- Nate
- Colin
