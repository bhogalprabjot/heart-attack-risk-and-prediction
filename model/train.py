import os
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
    print("=" * 70)
    print("  HEART ATTACK RISK PREDICTION IN INDIA: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Path Resolutions
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(base_dir, ".."))

    # Check for train and test datasets in repo root first, with fallback to data/
    train_path = os.path.join(repo_dir, "train_data.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join(repo_dir, "..", "data", "train_data.csv")

    test_path = os.path.join(repo_dir, "test_data.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join(repo_dir, "..", "data", "test_data.csv")

    print("\n[1/5] Loading datasets...")
    print(f"  Train dataset: {train_path}")
    print(f"  Test dataset:  {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Clean Patient_ID if present
    for df in [train_df, test_df]:
        if "Patient_ID" in df.columns:
            df.drop(columns=["Patient_ID"], inplace=True)

    target_col = "Heart_Attack_Risk"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"  X_train shape: {X_train.shape}, y_train class ratio: {y_train.mean():.4f}")
    print(f"  X_test shape:  {X_test.shape}, y_test class ratio:  {y_test.mean():.4f}")

    # 2. Define Features Grouping
    categorical_cols = ["State_Name", "Gender"]
    binary_cols = [
        "Diabetes", "Hypertension", "Obesity", "Smoking", "Alcohol_Consumption",
        "Physical_Activity", "Air_Pollution_Exposure", "Family_History",
        "Healthcare_Access", "Heart_Attack_History", "Health_Insurance"
    ]
    continuous_cols = [
        "Age", "Diet_Score", "Cholesterol_Level", "Triglyceride_Level", "LDL_Level",
        "HDL_Level", "Systolic_BP", "Diastolic_BP", "Stress_Level",
        "Emergency_Response_Time", "Annual_Income"
    ]

    print("\n[2/5] Building Preprocessing Pipeline (ColumnTransformer)...")
    print(f"  - Continuous features ({len(continuous_cols)}): StandardScaler")
    print(f"  - Categorical features ({len(categorical_cols)}): OneHotEncoder(drop='first')")
    print(f"  - Binary features ({len(binary_cols)}): passthrough")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
            ("bin", "passthrough", binary_cols)
        ]
    )

    # Fit strictly on train data to prevent any data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"  Transformed feature count: {X_train_processed.shape[1]}")

    # Save fitted preprocessor artifact
    preprocessor_path = os.path.join(base_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  -> Saved fitted preprocessor to: {preprocessor_path}")

    # 3. Model Definitions
    # Using class_weight='balanced' where supported to handle the 70:30 class imbalance
    # and prevent models from defaulting to the majority class (Recall=0).
    models = {
        "logistic_regression": (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        ),
        "decision_tree": (
            "Decision Tree",
            DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42)
        ),
        "knn": (
            "K-Nearest Neighbors",
            KNeighborsClassifier(n_neighbors=7, weights="distance")
        ),
        "naive_bayes": (
            "Naive Bayes",
            GaussianNB()
        ),
        "svm": (
            "Support Vector Machine",
            SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42)
        )
    }

    # 4. Training and Evaluation Loop
    print("\n[3/5] Training & Evaluating 5 Classification Models...")
    results = []

    for idx, (key, (display_name, model)) in enumerate(models.items(), 1):
        print(f"\n  [{idx}/5] Training {display_name}...")
        start_time = time.time()
        model.fit(X_train_processed, y_train)
        elapsed = time.time() - start_time

        # Predict classes and class probabilities
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1]

        # Calculate all 6 mandatory metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Save model artifact
        model_file = os.path.join(base_dir, f"{key}.joblib")
        joblib.dump(model, model_file)

        print(f"    Completed in {elapsed:.2f}s | Saved to {key}.joblib")
        print(f"    Scores -> Accuracy: {acc:.4f} | AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")

        results.append({
            "Model": display_name,
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "MCC": mcc
        })

    # 5. Format & Print Comparison Table
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("  [4/5] FINAL MODEL COMPARISON TABLE (Holdout Test Set)")
    print("=" * 70)
    header = f"{'Model':<25} | {'Accuracy':<8} | {'AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'MCC':<8}"
    separator = "-" * len(header)
    print(header)
    print(separator)
    for _, row in df_results.iterrows():
        print(f"{row['Model']:<25} | {row['Accuracy']:<8.4f} | {row['AUC']:<8.4f} | {row['Precision']:<9.4f} | {row['Recall']:<8.4f} | {row['F1']:<8.4f} | {row['MCC']:<8.4f}")

    # Save summary table as CSV for quick reference
    metrics_summary_path = os.path.join(base_dir, "metrics_summary.csv")
    df_results.to_csv(metrics_summary_path, index=False)
    print(f"\n  Summary table saved to: {metrics_summary_path}")

    # Identify best model by AUC and F1
    winner = df_results.sort_values(by=["AUC", "F1", "MCC"], ascending=False).iloc[0]
    print("\n" + "=" * 70)
    print(f"  [5/5] OVERALL WINNER / BEST MODEL: {winner['Model']}")
    print(f"    AUC: {winner['AUC']:.4f} | F1: {winner['F1']:.4f} | Accuracy: {winner['Accuracy']:.4f} | MCC: {winner['MCC']:.4f}")
    print("=" * 70)
    print("\nAll artifacts generated successfully! Ready for verification.")

if __name__ == "__main__":
    main()
