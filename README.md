# 📶 Predicting Problematic Internet Use Severity in Children

This project was developed as part of a machine learning competition using data from the **Healthy Brain Network (HBN)**. It predicts **SII** (Severity of Internet Impairment), an ordinal variable indicating the level of problematic internet usage among children and adolescents.

---

## 📌 Problem Statement

Predict the severity of problematic internet use (SII: 0 to 3) using both static and time-series data collected from over 5,000 participants aged 5–22. The goal is to aid early intervention and promote healthier digital habits.

---

## 📂 Data Overview

### 📁 Datasets
- **Static Data**: Demographics, clinical scores, and fitness metrics (`train.csv`, `test.csv`)
- **Time-Series Data**: Sensor-based temporal activity data (`series_train.parquet`, `series_test.parquet`)
- **Target Variable**: `sii` – Ordinal scale [0, 1, 2, 3]

---

## 🧹 Data Preprocessing

### 🔸 Static Data
- Removed IDs and irrelevant columns
- Median imputation for missing continuous values
- Label encoding for categorical variables

### 🔸 Time-Series Data
- Extracted statistical features (mean, std, trend) using **Polars**
- Merged time-series features with static data based on ID
- Final dataset combined both data types with aligned features

### 📊 Visualizations
- Age distribution
- SII class distribution
- Correlation heatmaps

---

## 🧠 Model Architecture

An ensemble of the following regressors was used:
- **LightGBM**
- **XGBoost**
- **CatBoost**
- **Random Forest**
- **Gradient Boosting Regressor**

### 🤖 Ensemble Strategy
- Combined using **Voting Regressor**
- Equal weights assigned to all models

---

## 🏋️‍♀️ Training and Optimization

- Used **Stratified 5-Fold Cross-Validation** to maintain label balance
- Evaluated performance using **Quadratic Weighted Kappa (QWK)**
- Applied **Nelder-Mead optimization** to fine-tune thresholds for converting continuous outputs to ordinal predictions

---

## 📈 Results & Insights

- **Final QWK score**: **0.450**  
- Hybrid feature set (static + time-series) significantly improved model accuracy  
- Threshold tuning reduced misclassification and aligned predictions with the ordinal nature of SII  
- Demonstrated strong potential for early identification of problematic internet use

---

## 🛠 Tech Stack

- Python (Pandas, NumPy, Polars)
- Scikit-learn
- LightGBM, XGBoost, CatBoost
- Matplotlib, Seaborn
- Optimization: Nelder-Mead

---

