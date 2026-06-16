# PRO-LIFT Analytics: Mission Control

## Predictive Maintenance and Explainable AI Platform for Rocket Booster Telemetry Monitoring

### Project Overview

**PRO-LIFT Analytics** is an end-to-end Predictive Maintenance (PdM) and Explainable AI (XAI) platform developed with Streamlit for monitoring the operational health of rocket booster engines. Leveraging the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset, the system analyzes multi-sensor telemetry data to detect degradation patterns, estimate equipment lifespan, and support proactive maintenance decisions.

The platform is designed to reduce the risk of mission-critical failures by combining predictive analytics with interpretable machine learning models that provide actionable insights into engine health and performance.

### Objectives

The system incorporates two complementary machine learning frameworks:

#### Remaining Useful Life (RUL) Prediction

A regression-based model estimates the remaining operational cycles of each booster, enabling maintenance planning before critical failures occur.

#### Health State Classification

A multi-class classification model categorizes booster health into operational risk levels:

* **Healthy**
* **Warning**
* **Critical**

This dual-model approach provides both quantitative lifespan estimates and qualitative health assessments.

---

## Key Features

### Real-Time Telemetry Integration

Interactive CSV-based data ingestion simulates live telemetry streams from operational booster systems, enabling real-time monitoring and analysis.

### Digital Twin Monitoring

Individual booster units can be analyzed through a digital twin interface that visualizes degradation trajectories, health indicators, and operational trends using dynamic gauges and interactive charts.

### Explainable AI Diagnostics

Integrated feature importance analysis identifies the sensors and operational parameters that contribute most significantly to performance degradation and maintenance predictions.

### Advanced Visualization Dashboard

A modern mission-control-inspired interface built with Streamlit and custom CSS delivers an intuitive monitoring experience through interactive dashboards, real-time metrics, and rich data visualizations.

---

## Data Science Pipeline

### 1. Data Understanding and Preprocessing

#### Feature Engineering

The dataset consists of:

* Three operational setting variables

  * `op_setting_1`
  * `op_setting_2`
  * `op_setting_3`
* Twenty-one continuous sensor measurements

#### Data Cleaning

Sensors exhibiting zero variance across operational cycles were removed to reduce redundancy, improve computational efficiency, and mitigate multicollinearity.

#### Remaining Useful Life Calculation

Remaining Useful Life (RUL) was calculated for each engine unit by subtracting the current cycle count from the maximum observed cycle count. Target values were capped at 125 cycles to reduce bias during early-stage degradation periods and improve model stability.

---

### 2. Exploratory Data Analysis (EDA)

#### Univariate Analysis

Statistical distributions of sensor readings were examined using:

* Histograms
* Kernel Density Estimation (KDE) plots
* Box plots

This analysis facilitated the identification of skewed distributions, anomalies, and outliers.

#### Correlation Analysis

Correlation heatmaps were generated to evaluate relationships among sensor measurements and identify variables strongly associated with engine degradation and RUL.

#### Health State Definition

Operational states were engineered based on RUL thresholds:

| Health State | Remaining Useful Life |
| ------------ | --------------------- |
| Healthy      | RUL > 100 cycles      |
| Warning      | 50 < RUL ≤ 100 cycles |
| Critical     | RUL ≤ 50 cycles       |

---

### 3. Model Development and Deployment

#### Regression Pipeline

A trained regression model predicts the exact Remaining Useful Life of an engine using standardized telemetry features processed through a fitted `StandardScaler`.

#### Classification Pipeline

A balanced multi-class classification model predicts operational health states and generates maintenance alerts. Target labels are encoded using a fitted `LabelEncoder`.

#### Model Deployment

All trained models and preprocessing artifacts are serialized and deployed within the Streamlit application to enable fast and consistent inference.

---

## Technology Stack

### Programming & Machine Learning

* Python 3.x
* Pandas
* NumPy
* SciPy
* Scikit-learn

### Data Visualization

* Plotly Express
* Plotly Graph Objects
* Matplotlib
* Seaborn

### Imbalanced Data Handling

* SMOTE (Synthetic Minority Over-sampling Technique)

### Deployment & User Interface

* Streamlit
* Custom CSS Components

### Model Persistence

* Joblib

---

## Project Structure

```text
├── .streamlit/
│   └── config.toml
├── reg_model.pkl
├── cls_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── features_list.pkl
├── app.py
└── README.md
```

### Asset Description

| File                | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `reg_model.pkl`     | Trained regression model for RUL prediction              |
| `cls_model.pkl`     | Trained classification model for health-state prediction |
| `scaler.pkl`        | Fitted feature scaling object                            |
| `label_encoder.pkl` | Label encoder for health-state classes                   |
| `features_list.pkl` | Selected feature set used during training                |
| `app.py`            | Main Streamlit application                               |
| `README.md`         | Project documentation                                    |

---

## Impact

PRO-LIFT Analytics demonstrates how predictive maintenance, machine learning, and explainable AI can be integrated into a unified decision-support system for aerospace applications. By providing accurate Remaining Useful Life estimates, health-state classification, and transparent diagnostic insights, the platform enables more reliable maintenance planning, improved operational safety, and reduced risk of mission failure.
