import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    # 1. Load dataset
    print("Loading phishing_features.csv...")
    df = pd.read_csv("phishing_features.csv")
    
    X = df.drop(columns=["label"])
    y = df["label"]
    
    feature_names = list(X.columns)
    print(f"Dataset shape: {df.shape}")
    print(f"Features count: {len(feature_names)}")

    # 2. Train/Test split
    # Stratification ensures that the proportion of target labels (0=Legitimate, 1=Phishing)
    # remains identical in both the training set and testing set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("\n--- Split Sizes ---")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # 3. Train Logistic Regression Baseline (with StandardScaler in Pipeline)
    print("\nTraining Logistic Regression (Baseline)...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds)
    lr_rec = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    lr_cm = confusion_matrix(y_test, lr_preds)

    # 4. Train Random Forest (Main Model)
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds)
    rf_rec = recall_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_cm = confusion_matrix(y_test, rf_preds)

    # 5. Display Confusion Matrices
    print("\n==================================================")
    print("           EVALUATION METRICS & CONFUSION MATRICES")
    print("==================================================")

    print("\n--- Logistic Regression (Baseline) ---")
    print(f"Accuracy:  {lr_acc:.4f}")
    print(f"Precision: {lr_prec:.4f}")
    print(f"Recall:    {lr_rec:.4f}")
    print(f"F1-Score:  {lr_f1:.4f}")
    print("\nConfusion Matrix (Logistic Regression):")
    print("                Predicted 0 (Legit)   Predicted 1 (Phishing)")
    print(f"Actual 0 (Legit)    {lr_cm[0,0]:<20} {lr_cm[0,1]}")
    print(f"Actual 1 (Phish)    {lr_cm[1,0]:<20} {lr_cm[1,1]}")

    print("\n--- Random Forest (Main Model) ---")
    print(f"Accuracy:  {rf_acc:.4f}")
    print(f"Precision: {rf_prec:.4f}")
    print(f"Recall:    {rf_rec:.4f}")
    print(f"F1-Score:  {rf_f1:.4f}")
    print("\nConfusion Matrix (Random Forest):")
    print("                Predicted 0 (Legit)   Predicted 1 (Phishing)")
    print(f"Actual 0 (Legit)    {rf_cm[0,0]:<20} {rf_cm[0,1]}")
    print(f"Actual 1 (Phish)    {rf_cm[1,0]:<20} {rf_cm[1,1]}")

    # 6. Create results table
    results_df = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": round(lr_acc, 4),
            "Precision": round(lr_prec, 4),
            "Recall": round(lr_rec, 4),
            "F1-score": round(lr_f1, 4)
        },
        {
            "Model": "Random Forest",
            "Accuracy": round(rf_acc, 4),
            "Precision": round(rf_prec, 4),
            "Recall": round(rf_rec, 4),
            "F1-score": round(rf_f1, 4)
        }
    ])
    results_df.to_csv("model_results.csv", index=False)
    print("\nSaved comparison table to model_results.csv")

    # 7. Select Final Model
    # Choose based on F1-score & Recall as required
    if rf_f1 > lr_f1:
        winner_name = "Random Forest"
        winner_model = rf_model
        winner_preds = rf_preds
        winner_probs = rf_probs
        reason = f"Random Forest achieved higher F1-score ({rf_f1:.4f} vs {lr_f1:.4f}) and Recall ({rf_rec:.4f} vs {lr_rec:.4f})."
    else:
        winner_name = "Logistic Regression"
        winner_model = lr_pipeline
        winner_preds = lr_preds
        winner_probs = lr_probs
        reason = f"Logistic Regression achieved higher F1-score ({lr_f1:.4f} vs {rf_f1:.4f}) and Recall ({lr_rec:.4f} vs {rf_rec:.4f})."

    print("\n==================================================")
    print(f"FINAL MODEL SELECTION: {winner_name}")
    print(f"Reason: {reason}")
    print("==================================================")

    # 8. Save final model
    joblib.dump(winner_model, "phishing_model.pkl")
    print("\nSuccessfully saved final model to phishing_model.pkl")

    # 9. Save test data/predictions for later analysis
    test_results_df = pd.DataFrame({
        "actual_label": y_test.values,
        "predicted_label": winner_preds,
        "phishing_probability": winner_probs
    })
    test_results_df.to_csv("test_predictions.csv", index=False)
    print("Successfully saved test predictions to test_predictions.csv")

if __name__ == "__main__":
    main()
