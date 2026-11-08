# Heart Attack Risk Prediction in India: Machine Learning Classification & Deployment

## a. Problem Statement
Cardiovascular disease (CVD) is the leading cause of mortality in India, accounting for nearly one-fourth of all national deaths and presenting substantial healthcare challenges across diverse urban and rural demographics. Early identification of individuals at high risk of a myocardial infarction (heart attack) can significantly mitigate adverse outcomes through preventive clinical interventions, targeted medication, and lifestyle modifications.

The objective of this project is to build, evaluate, and deploy an end-to-end Machine Learning pipeline that predicts binary heart attack risk (`Heart_Attack_Risk`: 0 for Low Risk, 1 for High Risk) using patient demographic, lifestyle, clinical, and socioeconomic parameters. Five distinct classification algorithms are comparatively evaluated across six performance metrics, and the trained pipeline is deployed as an interactive web application using Streamlit Community Cloud.

---

## b. Dataset Description
- **Source**: Kaggle ([Heart Attack Risk & Prediction Dataset In India](https://www.kaggle.com/datasets/ankushpanday2/heart-attack-risk-and-prediction-dataset-in-india))
- **Total Records (Instances)**: 10,000 patient records (surpassing the mandatory requirement of $\ge 500$)
- **Total Features**: 24 input features + 1 target column (surpassing the mandatory requirement of $\ge 12$)
- **Target Variable**: `Heart_Attack_Risk`
  - **Class 0 (Low Risk)**: 6,993 instances (~69.93%)
  - **Class 1 (High Risk)**: 3,007 instances (~30.07%)
- **Data Quality**: 0 missing values across all records.
- **Holdout Evaluation Split**: 8,000 samples for training (80%), 2,000 samples for independent holdout evaluation (20%) using stratified sampling (`random_state=42`).

### Feature Dictionary

| Category | Feature Name | Data Type | Description / Units |
| :--- | :--- | :---: | :--- |
| **Demographics** | `State_Name` | Categorical | 28 Indian states (One-Hot Encoded) |
| | `Age` | Numeric (Integer) | Patient age in years (20 to 79) |
| | `Gender` | Categorical | Male / Female (One-Hot Encoded) |
| | `Annual_Income` | Numeric (Float) | Annual household income (in INR) |
| **Clinical & Lab Markers** | `Cholesterol_Level` | Numeric (Float) | Total serum cholesterol in mg/dL |
| | `Triglyceride_Level` | Numeric (Float) | Serum triglycerides in mg/dL |
| | `LDL_Level` | Numeric (Float) | Low-density lipoprotein in mg/dL |
| | `HDL_Level` | Numeric (Float) | High-density lipoprotein in mg/dL |
| | `Systolic_BP` | Numeric (Float) | Systolic blood pressure (mmHg) |
| | `Diastolic_BP` | Numeric (Float) | Diastolic blood pressure (mmHg) |
| **Medical Conditions** | `Diabetes` | Binary (0/1) | Clinical diagnosis of diabetes |
| | `Hypertension` | Binary (0/1) | Clinical diagnosis of hypertension |
| | `Obesity` | Binary (0/1) | Obesity status |
| | `Family_History` | Binary (0/1) | Family history of cardiovascular disease |
| | `Heart_Attack_History`| Binary (0/1) | Prior incidence of cardiac events |
| **Lifestyle & Habits** | `Smoking` | Binary (0/1) | Active tobacco smoker |
| | `Alcohol_Consumption`| Binary (0/1) | Regular alcohol intake |
| | `Physical_Activity` | Binary (0/1) | Regular physical exercise |
| | `Diet_Score` | Numeric (0 to 10)| Dietary health rating (higher is healthier) |
| | `Stress_Level` | Numeric (1 to 10)| Self-reported psychological stress |
| **Environmental & Access**| `Air_Pollution_Exposure`| Binary (0/1) | High environmental pollution exposure |
| | `Healthcare_Access` | Binary (0/1) | Accessibility to immediate medical facilities |
| | `Emergency_Response_Time`| Numeric (Float)| Average ambulance arrival time (in minutes) |
| | `Health_Insurance` | Binary (0/1) | Coverage under health insurance |
| **Target Variable** | `Heart_Attack_Risk` | Binary (0/1) | **0 = Low Risk, 1 = High Risk** |

*(Note: The arbitrary `Patient_ID` column was removed during preprocessing as it provides no predictive value).*

---

## c. GitHub Repository Link & App Deployment
- **GitHub Repository**: [https://github.com/bhogalprabjot/heart-attack-risk-and-prediction](https://github.com/bhogalprabjot/heart-attack-risk-and-prediction)
- **Streamlit Web App**: *[Streamlit Community Cloud Link will be updated upon deployment]*

---

## d. Models Used & Comparison Table

All five models were trained on the 80% training set (8,000 samples) and evaluated on the identical 20% holdout test set (2,000 samples). Continuous features were normalized via `StandardScaler`, categorical features were one-hot encoded, and `class_weight='balanced'` was applied to counter the 70:30 class imbalance.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5170 | 0.5026 | 0.3056 | 0.4775 | 0.3727 | +0.0106 |
| **Decision Tree** | 0.4610 | 0.4895 | 0.3050 | **0.6206** | **0.4090** | **+0.0123** |
| **K-Nearest Neighbors (KNN)** | **0.6390** | 0.4932 | 0.2734 | 0.1215 | 0.1682 | -0.0232 |
| **Naive Bayes (Gaussian)** | 0.6040 | **0.5244** | **0.3094** | 0.2579 | 0.2813 | +0.0112 |
| **Support Vector Machine (SVM)**| 0.5175 | 0.4840 | 0.2898 | 0.4176 | 0.3422 | -0.0203 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | When I initially trained this without class weighting, the model took the easy way out and predicted "0" for everybody to get an artificial 70% accuracy. Adding `class_weight='balanced'` forced it to actually search for risk indicators, giving a decent recall of 47.75% and an F1 of 0.3727. It serves as a dependable linear baseline that catches roughly half of the high-risk cases without completely blowing up false alarms. |
| **Decision Tree** | I kept the tree pruned to a max depth of 5 to avoid memorizing noise, and it ended up being the most sensitive model of the lot with a recall of 62.06% and F1 of 0.4090. In a hospital screening setup, this is exactly what you want—I would rather have doctors run a secondary blood test on a healthy person (false positive) than send home an acute patient with a heart attack (false negative). |
| **K-Nearest Neighbors (KNN)** | On paper, KNN looks like the best model if you only glance at accuracy (63.90%). But when you dig deeper, it actually performed the worst in real terms. Because one-hot encoding the Indian states pushed the dataset to 50 dimensions, Euclidean distance breaks down (curse of dimensionality). Almost all neighbors ended up looking equally distant, so the algorithm simply leaned toward the 70% majority class. It missed nearly 88% of actual patients (recall of just 12.15%) and ended up with a negative MCC (-0.0232), meaning its predictions were actually slightly worse than a coin flip. |
| **Naive Bayes (Gaussian)** | Naive Bayes surprised me by turning out to be the most reliable model overall. Because it looks at the probabilities of each feature independently instead of calculating geometrical distances in 50-D space, it didn't suffer from the curse of dimensionality like KNN did. It achieved the highest AUC (0.5244), the best precision (30.94%), and a positive correlation with ground truth (MCC: +0.0112). |
| **Support Vector Machine (SVM)** | SVM with the RBF kernel took the longest to train on the BITS Virtual Lab VM. While it managed a 41.76% recall, its MCC was negative (-0.0203) and its AUC sat below 0.50. The complex non-linear kernel didn't find any clear geometric separation in this feature space, making it heavy on computation without any real diagnostic benefit over simpler models. |
| **Overall, Winner for your dataset?** | **Gaussian Naive Bayes (Overall Winner) & Decision Tree (Clinical Runner-Up)**.<br><br>If the goal is balanced risk scoring with the most reliable probability outputs, **Gaussian Naive Bayes** is the clear winner for its best-in-class AUC (0.5244) and precision (30.94%). If the clinical priority is aggressive triage where missing a single sick patient is unacceptable, **Decision Tree** is the practical choice due to its high recall of 62.06%. |
