# 🚀 PRO-LIFT Analytics: Mission Control
### Predictive Maintenance System for Rocket Boosters Engine Telemetry
**Developed by:** PLAN B Team

---

## 📌 Project Overview
**PRO-LIFT Analytics** is an advanced end-to-end Predictive Maintenance (PdM) and Explainable AI (XAI) dashboard powered by **Streamlit**. Utilizing the renowned **NASA CMAPS (Turbofan Engine Degradation Simulation) dataset**, the system ingests multi-sensor telemetry streams from rocket boosters to monitor structural health in real-time. 

The core objective is to mitigate catastrophic mission failures by dual-modeling the telemetry data:
1. **Regression System:** To accurately predict the **Remaining Useful Life (RUL)** of active boosters (T-minus cycles).
2. **Classification System:** To categorize booster health states into localized tiers (**Healthy**, **Warning**, **Critical**).

---

## 🚀 Key Features
*   **📡 Live Telemetry Stream Sync:** Interactive CSV uploading mechanism simulating a live satellite link to orbital boosters.
*   **🔭 Digital Twin Framework:** Deep-dive analysis for individual boosters tracking their custom propulsion degradation curves using dynamic Gauge meters and Plotly line paths.
*   **🔬 System Diagnostics:** Fully integrated Feature Importance breakdown directly evaluating which modules or sensors are the main drivers behind thermal and mechanical degradation.
*   **🎨 Futuristic UI/UX Design:** Cyberpunk/Space-themed design built entirely using CSS hacks over Streamlit components (`blur filters`, custom metrics, and neon gradients).

---

## 📊 Pipeline & Data Science Methodology

### 1. Data Understanding & Preprocessing
*   **Feature Structural Definition:** Mapping operational settings (`op_setting_1`, `op_setting_2`, `op_setting_3`) alongside 21 continuous telemetry sensor readings.
*   **Data Cleaning:** Identification and dropping of redundant sensors displaying **zero variance** (constant readings across cycles) to optimize computational efficiency and avoid multi-collinearity.
*   **RUL Optimization:** Computed the absolute Remaining Useful Life (RUL) per unit by capturing `max_cycles` and clipping the upper bound target at **125 cycles** to robustly handle initial steady-state operational phases.

### 2. Exploratory Data Analysis (EDA)
*   **Univariate Analysis:** Generated Histograms with KDE lines and Boxplots across all sensor metrics to detect extreme telemetry skewness and outliers.
*   **Bivariate Analysis:** Plotted massive multi-variable correlation heatmaps to assess structural dependencies and negative linear traits between telemetry trends and structural degradation (RUL).
*   **Target Segmentation:** Multi-tier target engineering mapping the exact boundaries for machine state transitions:
    *   🟢 **Healthy:** $RUL > 100\text{ cycles}$
    *   🟡 **Warning:** $50 < RUL \le 100\text{ cycles}$
    *   🔴 **Critical:** $RUL \le 50\text{ cycles}$

### 3. Model Architecture & Deployment
The operational backend relies on pre-trained serialized model assets (`.pkl`):
*   **Regression Pipeline:** Scaled input features through a fitted `StandardScaler` to output exact numerical RUL steps.
*   **Classification Pipeline:** Powered by a balanced multi-class classifier using specialized target labels encoded via `LabelEncoder` to predict tactical alerts (`Critical`, `Warning`, `Healthy`).

---

## 🛠️ Tech Stack & Architecture
*   **Core Logic:** Python 3.x, Pandas, NumPy, Scipy, Scikit-Learn
*   **Data Visualization:** Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
*   **Imbalanced Learning Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
*   **UI/UX Deployment:** Streamlit, Custom CSS HTML Components
*   **Model Serialization:** Joblib

---

## 📁 File Structure & Assets
```text
├── .streamlit/
│   └── config.toml           # Streamlit environment configs
├── reg_model.pkl             # Serialized Regression Model (RUL Engine)
├── cls_model.pkl             # Serialized Multi-class Classifier (Status)
├── scaler.pkl                # Fitted StandardScaler object
├── label_encoder.pkl         # Fitted LabelEncoder for state labels
├── features_list.pkl         # Saved Python list of optimal features
├── app.py                    # Main Streamlit "Mission Control" script
└── README.md                 # System Documentation
