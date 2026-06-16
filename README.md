# 🚀 PRO-LIFT: NASA Aircraft Engine Predictive Maintenance

An intelligent system and Streamlit dashboard built to predict the Remaining Useful Life (RUL) and monitor the health status of aircraft engines utilizing the NASA CMAPS dataset.

## 🚀 Features & Workflow
* **Data Preprocessing:** Cleaned and normalized sensor data from multiple engine fleets, dropping constant sensors with zero variance.
* **Exploratory Data Analysis (EDA):** Analyzed multi-sensor telemetry trends, correlations, and outliers.
* **Machine Learning Modeling:** * **Regression Engine:** Implemented models to predict the exact remaining useful life (RUL) cycles.
  * **Classification System:** Categorized engine health states into three tactical tiers (Healthy, Warning, Critical) utilizing SMOTE for data balancing.
* **Interactive Dashboard:** Built a futuristic user interface using **Streamlit** to upload telemetry files, view dynamic digital twin gauges, and track engine degradation curves.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework & UI:** Streamlit, Custom CSS
* **Libraries:** Pandas, NumPy, Scikit-Learn, Plotly, Joblib, IMBLearn (SMOTE)
