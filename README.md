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

The Exploratory Data Analysis (EDA) phase was carried out to gain a comprehensive understanding of the Steel Industry Energy Consumption dataset before developing any machine learning models. The primary objective of this stage was to examine the dataset's structure, evaluate data quality, discover hidden patterns, understand relationships among variables, and identify the features that have the greatest influence on electricity consumption.

The analysis began by loading the dataset and reviewing its overall structure using the dataset dimensions, data types, summary statistics, and sample records. The dataset consists of **35,040 observations** collected at **15-minute intervals** throughout the year **2018**, providing detailed operational information about energy consumption in a steel manufacturing plant. The inspection confirmed that the dataset contains **11 original features**, representing electrical measurements, time-related information, and plant operating conditions.

Data quality checks were then performed to ensure that the dataset was reliable for further analysis. The original dataset contained **no missing values**, indicating that the collected measurements were complete. During feature engineering, a single missing value appeared in the newly created **Power_Factor_Ratio** feature because one record contained a lagging power factor of zero, making division mathematically undefined. This missing value represented only one observation out of more than thirty-five thousand records and therefore had a negligible impact on the overall analysis.

To provide additional information for analysis and model training, several new features were engineered from the original dataset. The **date** column was converted into a datetime format, allowing the extraction of meaningful temporal features such as **Hour**, **Day of Week**, **Month**, and **Day Type (Weekday/Weekend)**. These features were created to capture daily and seasonal patterns in electricity consumption. Another engineered feature, **Power_Factor_Ratio**, was introduced to represent the relationship between the leading and lagging power factors, providing a more informative measure than the individual variables alone. In addition, a binary **High_Load** feature was created using the 75th percentile of energy consumption to distinguish periods of unusually high electricity demand.

Outlier analysis was performed using the **Interquartile Range (IQR)** method on the target variable (**Usage_kWh**). The analysis identified **328 outliers**, representing approximately **0.94%** of the entire dataset. These observations correspond to periods of exceptionally high energy consumption rather than data entry errors or measurement issues. Since they reflect genuine plant operating conditions, the outliers were intentionally retained to ensure that the machine learning models learn to predict both normal and peak energy usage.

<img width="1075" height="580" alt="Screenshot (4161)" src="https://github.com/user-attachments/assets/ace0c999-2309-4e2c-beef-6033de499721" />


A correlation analysis was then conducted to measure the strength of the relationship between numerical features and the target variable. The resulting correlation heatmap revealed that **CO₂ emissions** have the strongest positive correlation with energy consumption (**0.988**), followed by **Lagging Current Reactive Power** (**0.896**) and **Lagging Current Power Factor** (**0.386**). These results indicate that as the plant consumes more electricity, CO₂ emissions and reactive power also increase significantly, making these variables valuable predictors for the regression models.
<img width="1392" height="772" alt="Screenshot (4162)" src="https://github.com/user-attachments/assets/22d7fde6-8ddc-41b5-8210-2e0c4ee33b76" />


The relationship between plant operating conditions and electricity consumption was further explored by analyzing the average energy usage for each **Load Type**. The grouped bar chart showed a clear increase in electricity consumption from **Light Load** to **Medium Load**, with the highest average energy usage observed during **Maximum Load** operations. This confirms that the operational load of the plant has a direct impact on electricity demand.
<img width="1444" height="587" alt="Screenshot (4157)" src="https://github.com/user-attachments/assets/088f1393-de89-4d39-8629-f74b8b911027" />


Finally, the hourly energy consumption trend was examined using a line chart. The analysis demonstrated that electricity usage follows a consistent daily pattern, with consumption gradually increasing during working hours, reaching its highest average level around **9:00 AM**, and then decreasing during the evening and overnight periods. This behavior aligns with the expected production schedule of an industrial manufacturing facility.
<img width="1327" height="599" alt="Screenshot (4158)" src="https://github.com/user-attachments/assets/a1f07e79-708b-47d0-be11-3e84d1c86d9a" />

Overall, the EDA phase provided valuable insights into the dataset, confirmed that the data was of high quality, highlighted the variables most strongly associated with electricity consumption, and established a solid foundation for feature selection and predictive modeling.

# Model Training Process

After completing the exploratory analysis and feature engineering, the dataset was prepared for machine learning by applying a series of preprocessing steps. The objective of this stage was to transform the engineered dataset into a format suitable for regression algorithms while ensuring that the evaluation remained fair and free from data leakage.

The first preprocessing step involved removing unnecessary and redundant features. The original **date** column was excluded because its information had already been represented through the engineered temporal features such as Hour, Month, and Day Type. The engineered **High_Load** feature was also removed because it was directly derived from the target variable (**Usage_kWh**) and would introduce target leakage if included during training. Additionally, duplicate categorical columns representing the same information were removed to avoid redundancy.

Machine learning algorithms require numerical input features; therefore, categorical variables such as **Load Type**, **Day of Week**, **Month**, and **Day Type** were transformed using **One-Hot Encoding**. This preprocessing technique converts each category into separate binary variables while preserving all available information without introducing artificial numerical relationships between categories.

The target variable for prediction was defined as **Usage_kWh**, while all remaining engineered features served as input variables. The dataset was then divided into **80% training data** and **20% testing data** using a fixed **random_state = 42** to ensure that the results could be reproduced consistently.

Four commonly used regression algorithms were selected to establish baseline performance:

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

Each model was trained using the same training dataset to provide a fair comparison. The models represent both traditional linear approaches and more advanced tree-based ensemble methods, allowing the strengths and limitations of different regression techniques to be evaluated.

Model performance was assessed using three complementary evaluation metrics. **Mean Absolute Error (MAE)** measured the average prediction error, **Root Mean Squared Error (RMSE)** measured the magnitude of prediction errors while placing greater emphasis on larger mistakes, and the **R² Score** measured how well each model explained the variation in energy consumption.

To further evaluate the robustness of each model, **5-Fold Cross Validation** was performed. This technique repeatedly divides the dataset into different training and validation subsets, producing a more reliable estimate of how well each model is expected to perform on unseen data. Using cross-validation reduces the likelihood of selecting a model that performs well only because of a favorable train-test split.

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
