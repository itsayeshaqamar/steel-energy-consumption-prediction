<img width="212" height="212" alt="logo" src="https://github.com/user-attachments/assets/e9b2db24-18ac-4eaf-83ee-58253fec7f5c" />
# Steel Industry Energy Consumption Prediction 

## Week 2 Task Overview

This task focuses on analyzing and predicting energy consumption in a steel manufacturing industry using machine learning. The objective is to understand the factors affecting electricity usage, engineer meaningful features, perform exploratory data analysis (EDA), and build baseline regression models capable of accurately predicting energy consumption.

The task is divided into two main stages:

- **Part 1:** Exploratory Data Analysis (EDA) and Feature Engineering
- **Part 2:** Baseline Machine Learning Models

---

# Dataset Information

The dataset contains electricity consumption records collected every 15 minutes throughout the year 2018.

### Dataset Summary

- **Total Records:** 35,040
- **Features (Original):** 11
- **Features After Engineering:** 17

### Target Variable

- **Usage_kWh**
  - Energy consumption (kilowatt-hours)
  - This is the value predicted by the machine learning models.

### Original Features

- Date and time
- Reactive power measurements
- CO₂ emissions
- Power factors
- Time information (NSM)
- Weekday/Weekend
- Day of week
- Load type

---

# Environment Setup

## Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

## Create a virtual environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Feature Engineering Steps

Several new features were created to improve model performance.

### 1. Date Conversion

Converted the `date` column into datetime format.

### 2. Time-Based Features

Extracted:

- Hour
- Day of Week Name
- Month
- Day Type (Weekday/Weekend)

### 3. Power Factor Ratio

Created a new feature:

```
Power_Factor_Ratio =
Leading_Current_Power_Factor
/
Lagging_Current_Power_Factor
```

This feature captures the relationship between leading and lagging power factors.

### 4. High Load Indicator

Created a binary feature using the 75th percentile of energy consumption.

- 1 → High Load
- 0 → Normal Load

---

# Exploratory Data Analysis (EDA)

Several analyses were performed to better understand the dataset.

## Dataset Inspection

- Checked dataset shape
- Verified data types
- Reviewed descriptive statistics

## Missing Values

- No missing values in original dataset
- One engineered value handled after ratio calculation

## Outlier Detection

Used the IQR method.

Results:

- Outliers detected: **328**
- Percentage: **0.94%**

Since these values represent genuine high energy consumption periods, they were retained.

## Correlation Analysis

Strong positive correlations with energy consumption:

| Feature | Correlation |
|----------|------------:|
| CO2(tCO2) | 0.988 |
| Lagging Reactive Power | 0.896 |
| Lagging Power Factor | 0.386 |

## Load Type Analysis

Average energy consumption:

| Load Type | Average Usage (kWh) |
|-----------|--------------------:|
| Light Load | 8.63 |
| Medium Load | 38.45 |
| Maximum Load | 59.27 |

## Hourly Energy Consumption

Peak usage occurred around **9:00 AM**, while the lowest average consumption occurred during the early morning hours.

---

# Model Training Process

## Data Preparation

Before training:

- Removed unnecessary columns
  - date
  - High_Load

- Removed duplicate categorical features
  - WeekStatus
  - Day_of_week

- Applied One-Hot Encoding to categorical variables.

## Train-Test Split

- Training Data: 80%
- Testing Data: 20%

## Models Trained

The following regression algorithms were implemented:

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

## Model Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-Fold Cross Validation

---

# Results and Conclusions

## Model Performance

| Model | RMSE | R² Score |
|--------|------:|---------:|
| Linear Regression | 4.118 | 0.985 |
| Ridge Regression | 6.177 | 0.966 |
| Decision Tree | 1.539 | 0.998 |
| Random Forest | **1.052** | **0.999** |

## Cross Validation

| Model | Mean CV RMSE |
|--------|-------------:|
| Linear Regression | 4.567 |
| Ridge Regression | 6.931 |
| Decision Tree | 2.499 |
| Random Forest | **2.103** |

## Final Model

Random Forest achieved the best overall performance by providing:

- Lowest prediction error
- Highest R² Score
- Best cross-validation performance
- Strong agreement between predicted and actual values

It was selected as the final baseline model for future improvement and optimization.

---

# Screenshots

Include screenshots of the following outputs in the repository (recommended):

- Dataset overview
- Feature engineering output
- Correlation heatmap
- Load type analysis
- Hourly energy consumption plot
- RMSE comparison chart
- Predicted vs Actual Values (Random Forest)

---

# Project Structure

```
├── week2_eda.ipynb
├── week2_baseline_models.ipynb
├── data/steel_energy_engineered.csv
 ├── data/steel_industry_data.csv
├── requirements.txt
└── README.md
```

---

# Author

**Ayesha Qamar**

AI/Ml Intern at IT Simplera Institute
