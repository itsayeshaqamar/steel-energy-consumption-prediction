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


# Exploratory Data Analysis (EDA) Findings

The Exploratory Data Analysis (EDA) phase was conducted to understand the dataset, evaluate data quality, discover important patterns, and identify the factors influencing energy consumption before building machine learning models.

### Dataset Understanding

- Analyzed **35,040 records** collected at **15-minute intervals** throughout 2018.
- Examined dataset structure, data types, dimensions, and summary statistics.
- Verified that the dataset contained **11 original features** related to electrical measurements, operating conditions, and time information.

### Data Quality Assessment

- Confirmed that the original dataset contained **no missing values**.
- One missing value appeared in the engineered **Power_Factor_Ratio** feature due to division by zero (lagging power factor = 0).
- The missing value affected only one record and had a negligible impact on the analysis.

### Feature Engineering

The following features were created to provide additional information for analysis and model training:

- **Hour** – Captures hourly energy consumption patterns.
- **Day_of_Week_Name** – Identifies the weekday of each observation.
- **Month** – Enables monthly trend analysis.
- **Day_Type** – Classifies records as Weekday or Weekend.
- **Power_Factor_Ratio** – Represents the relationship between leading and lagging power factors.
- **High_Load** – Binary indicator identifying observations above the 75th percentile of energy consumption.

### Outlier Analysis

- Applied the **Interquartile Range (IQR)** method on **Usage_kWh**.
- Detected **328 outliers (0.94%)**.
- These observations were retained because they represent genuine high-energy operating conditions rather than data errors.
  
<img width="1075" height="580" alt="Screenshot (4161)" src="https://github.com/user-attachments/assets/ace0c999-2309-4e2c-beef-6033de499721" />

### Key Findings
A correlation analysis was then conducted to measure the strength of the relationship between numerical features and the target variable. The resulting correlation heatmap revealed that **CO₂ emissions** have the strongest positive correlation with energy consumption (**0.988**), followed by **Lagging Current Reactive Power** (**0.896**) and **Lagging Current Power Factor** (**0.386**). These results indicate that as the plant consumes more electricity, CO₂ emissions and reactive power also increase significantly, making these variables valuable predictors for the regression models.
<img width="1392" height="772" alt="Screenshot (4162)" src="https://github.com/user-attachments/assets/22d7fde6-8ddc-41b5-8210-2e0c4ee33b76" />

### Visual Insights

The relationship between plant operating conditions and electricity consumption was further explored by analyzing the average energy usage for each **Load Type**. The grouped bar chart showed a clear increase in electricity consumption from **Light Load** to **Medium Load**, with the highest average energy usage observed during **Maximum Load** operations. This confirms that the operational load of the plant has a direct impact on electricity demand.
<img width="1444" height="587" alt="Screenshot (4157)" src="https://github.com/user-attachments/assets/088f1393-de89-4d39-8629-f74b8b911027" />


Finally, the hourly energy consumption trend was examined using a line chart. The analysis demonstrated that electricity usage follows a consistent daily pattern, with consumption gradually increasing during working hours, reaching its highest average level around **9:00 AM**, and then decreasing during the evening and overnight periods. This behavior aligns with the expected production schedule of an industrial manufacturing facility.
<img width="1327" height="599" alt="Screenshot (4158)" src="https://github.com/user-attachments/assets/a1f07e79-708b-47d0-be11-3e84d1c86d9a" />


Overall, the EDA phase provided a clear understanding of the dataset, identified the most influential features, and established a strong foundation for machine learning.

# Model Training Process

After completing feature engineering, the dataset was prepared for regression modeling through a series of preprocessing steps to ensure fair and reliable model evaluation.

### Data Preprocessing

The following preprocessing steps were performed:

- Removed the **date** column after extracting all useful time-based features.
- Removed **High_Load** to prevent target leakage since it was derived from the target variable.
- Removed duplicate categorical features (**WeekStatus** and **Day_of_week**) because equivalent engineered features already existed.
- Applied **One-Hot Encoding** to convert categorical variables into numerical format.

### Model Preparation

- **Target Variable:** `Usage_kWh`
- **Input Features:** All remaining engineered features
- **Train-Test Split:** 80% Training, 20% Testing
- **Random State:** 42 (for reproducibility)

### Regression Models

Four baseline regression models were trained:

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

### Evaluation Metrics

Each model was evaluated using:

- **MAE (Mean Absolute Error)** – Measures average prediction error.
- **RMSE (Root Mean Squared Error)** – Penalizes larger prediction errors.
- **R² Score** – Measures how well the model explains the variation in energy consumption.
- **5-Fold Cross Validation** – Evaluates model consistency and generalization on unseen data.

This process ensured a fair comparison between all regression models using the same training and testing data.

---

# Results and Conclusions

## Model Performance

| Model | RMSE | R² Score |
|--------|------:|---------:|
| Linear Regression | 4.118 | 0.985 |
| Ridge Regression | 6.177 | 0.966 |
| Decision Tree | 1.539 | 0.998 |
| Random Forest | **1.052** | **0.999** |

The four regression models produced noticeably different levels of predictive performance when evaluated on the testing dataset. Among the linear models, **Linear Regression** provided a strong baseline with relatively low prediction error, while **Ridge Regression** achieved slightly lower performance due to the effect of regularization on this particular dataset.

The tree-based models significantly outperformed the linear approaches. The **Decision Tree Regressor** achieved excellent prediction accuracy with a very low RMSE and a high R² score, demonstrating its ability to capture complex nonlinear relationships within the data. However, the **Random Forest Regressor** produced the best overall results across all evaluation metrics.

## Cross Validation
The 5-Fold Cross Validation results further supported the superiority of the Random Forest model. It achieved the **lowest average cross-validation RMSE (2.103)** among all evaluated models, indicating that its performance remains consistent across different subsets of the dataset. Although the cross-validation error is slightly higher than the testing RMSE, the difference is expected because each validation fold contains unseen data. The relatively small gap between these values suggests that the model generalizes well and does not exhibit significant overfitting.
| Model | Mean CV RMSE |
|--------|-------------:|
| Linear Regression | 4.567 |
| Ridge Regression | 6.931 |
| Decision Tree | 2.499 |
| Random Forest | **2.103** |

<img width="1165" height="606" alt="Screenshot (4159)" src="https://github.com/user-attachments/assets/9770a325-1842-4b1d-af4f-5748ebe1ad0e" />

To visually validate the model's predictions, a scatter plot comparing the **actual** and **predicted** energy consumption values was generated for the Random Forest model. Most data points were concentrated close to the ideal diagonal reference line, indicating strong agreement between the predicted and observed values. This confirms that the model accurately captures both low and high energy consumption patterns.

<img width="1009" height="779" alt="Screenshot (4163)" src="https://github.com/user-attachments/assets/424343fa-7c55-404d-9b1f-251ec320d5cd" />

Overall, this project successfully completed the end-to-end workflow of a supervised machine learning regression problem, beginning with data exploration and feature engineering, followed by preprocessing, baseline model development, evaluation, and model comparison. Based on its superior predictive accuracy, strong generalization ability, and consistent cross-validation performance, the **Random Forest Regressor** was selected as the final baseline model and provides a reliable foundation for future model optimization and deployment.

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
