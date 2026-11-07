# Heart Attack Risk Prediction in India: Machine Learning Classification & Deployment

## a. Problem Statement
Cardiovascular disease (CVD) is the leading cause of mortality in India, accounting for nearly one-fourth of all national deaths and presenting substantial healthcare challenges across diverse urban and rural demographics. Early identification of individuals at high risk of a myocardial infarction (heart attack) can significantly mitigate adverse outcomes through preventive clinical interventions and targeted lifestyle modifications.

The objective of this project is to build, evaluate, and deploy an end-to-end Machine Learning pipeline that predicts binary heart attack risk (Heart_Attack_Risk: 0 for Low Risk, 1 for High Risk) using patient demographic, lifestyle, clinical, and socioeconomic parameters. Five distinct classification algorithms are comparatively evaluated across six performance metrics, and the best-performing model is deployed as an interactive web application using Streamlit Community Cloud.

---

## b. Dataset Description
- **Source**: Kaggle ([Heart Attack Risk & Prediction Dataset In India](https://www.kaggle.com/datasets/ankushpanday2/heart-attack-risk-and-prediction-dataset-in-india))
- **Total Records (Instances)**: 10,000 patient records (surpassing the mandatory requirement of >= 500)
- **Total Features**: 24 input features + 1 target column (surpassing the mandatory requirement of >= 12)
- **Target Variable**: Heart_Attack_Risk
  - **Class 0 (Low Risk)**: 6,993 instances (~69.93%)
  - **Class 1 (High Risk)**: 3,007 instances (~30.07%)
- **Data Quality**: 0 missing values across all records.

### Feature Dictionary

| Category | Feature Name | Data Type | Description / Units |
| :--- | :--- | :---: | :--- |
| **Demographics** | State_Name | Categorical | 28 Indian states |
| | Age | Numeric (Integer) | Patient age in years (20 to 79) |
| | Gender | Categorical | Male / Female |
| | Annual_Income | Numeric (Float) | Annual household income (in INR) |
| **Clinical & Lab Markers** | Cholesterol_Level | Numeric (Float) | Total serum cholesterol in mg/dL |
| | Triglyceride_Level | Numeric (Float) | Serum triglycerides in mg/dL |
| | LDL_Level | Numeric (Float) | Low-density lipoprotein in mg/dL |
| | HDL_Level | Numeric (Float) | High-density lipoprotein in mg/dL |
| | Systolic_BP | Numeric (Float) | Systolic blood pressure (mmHg) |
| | Diastolic_BP | Numeric (Float) | Diastolic blood pressure (mmHg) |
| **Medical Conditions** | Diabetes | Binary (0/1) | Clinical diagnosis of diabetes |
| | Hypertension | Binary (0/1) | Clinical diagnosis of hypertension |
| | Obesity | Binary (0/1) | Obesity status |
| | Family_History | Binary (0/1) | Family history of cardiovascular disease |
| | Heart_Attack_History| Binary (0/1) | Prior incidence of cardiac events |
| **Lifestyle & Habits** | Smoking | Binary (0/1) | Active tobacco smoker |
| | Alcohol_Consumption| Binary (0/1) | Regular alcohol intake |
| | Physical_Activity | Binary (0/1) | Regular physical exercise |
| | Diet_Score | Numeric (0 to 10)| Dietary health rating (higher is healthier) |
| | Stress_Level | Numeric (1 to 10)| Self-reported psychological stress |
| **Environmental & Access**| Air_Pollution_Exposure| Binary (0/1) | High environmental pollution exposure |
| | Healthcare_Access | Binary (0/1) | Accessibility to immediate medical facilities |
| | Emergency_Response_Time| Numeric (Float)| Average ambulance arrival time (in minutes) |
| | Health_Insurance | Binary (0/1) | Coverage under health insurance |
| **Target Variable** | Heart_Attack_Risk | Binary (0/1) | **0 = Low Risk, 1 = High Risk** |

*(Note: The arbitrary Patient_ID column was removed during initial cleaning as it carries no predictive signal).*

---

## c. GitHub Repository Link
- **Repository URL**: [https://github.com/bhogalprabjot/heart-attack-risk-and-prediction](https://github.com/bhogalprabjot/heart-attack-risk-and-prediction)

---
