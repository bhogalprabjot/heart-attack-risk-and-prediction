import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Predictor | BITS Pilani ML",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #d90429;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #d90429;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to cache model & preprocessor loading
@st.cache_resource
def load_model_and_preprocessor(model_key: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    model_path = os.path.join(model_dir, f"{model_key}.joblib")
    
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor artifact not found at: {preprocessor_path}")
        return None, None
    if not os.path.exists(model_path):
        st.error(f"Model artifact not found at: {model_path}")
        return None, None
        
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    # Scikit-learn backwards compatibility patch for cross-version unpickling
    if hasattr(model, "__class__") and "LogisticRegression" in model.__class__.__name__:
        if not hasattr(model, "multi_class"):
            model.multi_class = "auto"

    return preprocessor, model

# Model Mapping Dictionary
MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors (KNN)": "knn",
    "Naive Bayes (Gaussian)": "naive_bayes",
    "Support Vector Machine (SVM)": "svm"
}

# --- HEADER SECTION ---
st.markdown('<div class="main-header">🫀 Heart Attack Risk Prediction in India</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">BITS Pilani M.Tech (AIML) — Machine Learning Assignment 2 | '
    'End-to-End Classification & Model Benchmark System</div>',
    unsafe_allow_html=True
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Configuration & Inputs")

# 1. Model Selection Dropdown (Mandatory 1 mark in rubric)
selected_model_name = st.sidebar.selectbox(
    "Select ML Classification Model:",
    options=list(MODEL_OPTIONS.keys()),
    index=3  # Default to Naive Bayes (Winner)
)
selected_model_key = MODEL_OPTIONS[selected_model_name]

st.sidebar.markdown("---")

# 2. Dataset Upload / Selection (Mandatory 1 mark in rubric)
st.sidebar.subheader("📁 Test Dataset Evaluation")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File (with features + target):",
    type=["csv"]
)

use_default = st.sidebar.checkbox(
    "Use Bundled Holdout Test Data (`test_data.csv`)",
    value=uploaded_file is None
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dataset Details:**
- **Domain:** Indian Cardiovascular Health
- **Total Features:** 24 input features
- **Target:** `Heart_Attack_Risk` (0: Low Risk, 1: High Risk)
- **Class Ratio:** ~70% Low Risk / ~30% High Risk
""")

# --- LOAD EVALUATION DATA ---
df_test = None
data_source_label = ""

if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        data_source_label = f"📁 Custom Uploaded File (`{uploaded_file.name}` — {df_test.shape[0]} rows, {df_test.shape[1]} columns)"
        st.sidebar.success(f"File uploaded successfully: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
elif use_default:
    default_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.csv")
    if os.path.exists(default_test_path):
        df_test = pd.read_csv(default_test_path)
        data_source_label = f"📦 Bundled Holdout Test Set (`test_data.csv` — {df_test.shape[0]} records)"
        st.sidebar.info(f"Loaded bundled test data ({df_test.shape[0]} records).")
    else:
        st.error("Default `test_data.csv` not found in repository root. Please upload a CSV file.")

# --- MAIN EVALUATION LOGIC ---
if df_test is not None:
    # Display Active Data Source Banner
    st.info(f"**Active Evaluation Dataset:** {data_source_label}")
    # Clean identifier if present
    if "Patient_ID" in df_test.columns:
        df_test = df_test.drop(columns=["Patient_ID"])
        
    target_col = "Heart_Attack_Risk"
    
    if target_col not in df_test.columns:
        st.warning(f"⚠️ Target column '{target_col}' not found in the uploaded CSV. Evaluation metrics require ground-truth labels.")
        st.write("First 5 rows of uploaded data:", df_test.head())
    else:
        X_eval = df_test.drop(columns=[target_col])
        y_eval = df_test[target_col]
        
        # Load Model and Preprocessor
        preprocessor, model = load_model_and_preprocessor(selected_model_key)
        
        if preprocessor is not None and model is not None:
            # Transform Features using fitted preprocessor
            try:
                X_eval_transformed = preprocessor.transform(X_eval)
            except Exception as e:
                st.error(f"Error during feature preprocessing: {e}")
                st.stop()
                
            # Predictions and Probabilities
            y_pred = model.predict(X_eval_transformed)
            
            # Predict probabilities for AUC calculation
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_eval_transformed)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_eval_transformed)
            else:
                y_proba = y_pred
                
            # Compute All 6 Required Metrics
            acc = accuracy_score(y_eval, y_pred)
            auc = roc_auc_score(y_eval, y_proba)
            prec = precision_score(y_eval, y_pred, zero_division=0)
            rec = recall_score(y_eval, y_pred, zero_division=0)
            f1 = f1_score(y_eval, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_eval, y_pred)
            
            # --- SECTION 1: METRICS DISPLAY (Mandatory 1 mark in rubric) ---
            st.subheader(f"📊 Model Performance Metrics: {selected_model_name}")
            
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
            
            col1.metric("Accuracy", f"{acc:.4f}", help="Overall proportion of correct classifications")
            col2.metric("ROC-AUC", f"{auc:.4f}", help="Area under the ROC Curve (Discriminatory power)")
            col3.metric("Precision", f"{prec:.4f}", help="Accuracy of positive risk predictions")
            col4.metric("Recall (Sensitivity)", f"{rec:.4f}", help="Proportion of actual cardiac risk cases identified")
            col5.metric("F1-Score", f"{f1:.4f}", help="Harmonic mean of Precision and Recall")
            col6.metric("MCC", f"{mcc:+.4f}", help="Matthews Correlation Coefficient (-1.0 to +1.0)")
            
            st.markdown("---")
            
            # --- SECTION 2: VISUALIZATIONS (Mandatory 1 mark in rubric) ---
            col_viz1, col_viz2 = st.columns([1, 1])
            
            with col_viz1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_eval, y_pred)
                
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["No Risk (0)", "High Risk (1)"],
                    yticklabels=["No Risk (0)", "High Risk (1)"],
                    ax=ax
                )
                plt.xlabel("Predicted Label")
                plt.ylabel("Actual Label")
                plt.title(f"Confusion Matrix ({selected_model_name})")
                st.pyplot(fig)
                plt.close()
                
            with col_viz2:
                st.subheader("Classification Report")
                report_dict = classification_report(
                    y_eval, y_pred,
                    target_names=["Low Risk (0)", "High Risk (1)"],
                    output_dict=True,
                    zero_division=0
                )
                df_report = pd.DataFrame(report_dict).T.round(4)
                st.dataframe(df_report.style.highlight_max(axis=0, color="#d8f3dc"), use_container_width=True)
                
                st.info(
                    f"**Analysis Note:** Evaluated on **{len(y_eval)}** test records. "
                    f"Actual high-risk cases: **{int(y_eval.sum())}**, "
                    f"Correctly caught (True Positives): **{cm[1, 1]}**."
                )
                
            # --- SECTION 3: DATA PREVIEW & PREDICTIONS ---
            with st.expander("🔍 Inspect Uploaded Test Data with Model Predictions"):
                df_display = X_eval.copy()
                df_display["Actual_Risk"] = y_eval.values
                df_display["Predicted_Risk"] = y_pred
                if hasattr(model, "predict_proba"):
                    df_display["Risk_Probability (%)"] = (y_proba * 100).round(2)
                st.dataframe(df_display.head(50), use_container_width=True)
                
else:
    st.info("👆 Please upload a test CSV file or check 'Use Bundled Holdout Test Data' in the sidebar to begin.")
