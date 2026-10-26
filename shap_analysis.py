"""
shap_analysis.py - Phase 3: SHAP Explainability for Phishing Detection Model

This script explains WHY the Random Forest model makes its predictions
using SHAP (SHapley Additive exPlanations).

Key distinctions:
  GLOBAL explanation: Which features generally influence the model across many URLs?
  LOCAL explanation : Which features influenced the prediction for ONE specific URL?

Note: SHAP explains the model's learned behavior, NOT real-world causation.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")        # non-interactive backend - must be set before pyplot import
import matplotlib.pyplot as plt
import joblib
import shap

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 0. Configuration
# --------------------------------------------------
FEATURE_COLUMNS = [
    'url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'num_at',
    'num_percent', 'num_question', 'num_equals', 'num_ampersand', 'uses_https',
    'has_ip', 'num_subdomains', 'domain_length', 'has_login', 'has_verify',
    'has_secure', 'has_account', 'has_update'
]

SAMPLE_SIZE  = 5000   # representative sample for SHAP value calculation
RANDOM_STATE = 42
OUTPUT_DIR   = "shap_outputs"

# --------------------------------------------------
# 1. Setup output directory
# --------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("\n" + "="*60)
print("  PHASE 3 - SHAP EXPLAINABILITY ANALYSIS")
print("="*60)
print(f"SHAP version : {shap.__version__}")
print(f"Output folder: {OUTPUT_DIR}/")

# --------------------------------------------------
# 2. Load model
# --------------------------------------------------
print("\n[1/6] Loading model from phishing_model.pkl ...")
model = joblib.load("phishing_model.pkl")
print(f"      Model type: {type(model).__name__}")

# --------------------------------------------------
# 3. Load feature data and validate
# --------------------------------------------------
print("\n[2/6] Loading phishing_features.csv ...")
df_all = pd.read_csv("phishing_features.csv")
X_all  = df_all[FEATURE_COLUMNS]
y_all  = df_all["label"]

assert list(X_all.columns) == FEATURE_COLUMNS, \
    f"Feature mismatch!\nExpected: {FEATURE_COLUMNS}\nGot: {list(X_all.columns)}"
assert X_all.shape[1] == 18, f"Expected 18 features, got {X_all.shape[1]}"

print(f"      Dataset shape  : {df_all.shape}")
print(f"      Feature columns: {X_all.shape[1]} verified")
print(f"      Label dist     : {dict(y_all.value_counts().sort_index())}  (0=Legit, 1=Phishing)")

# --------------------------------------------------
# 4. Reconstruct the exact test split (same seed as training)
# --------------------------------------------------
from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
print(f"\n      Test set size : {X_test.shape[0]} samples")

# --------------------------------------------------
# 5. Load test_predictions.csv
# --------------------------------------------------
test_preds_df = pd.read_csv("test_predictions.csv")
test_preds_df = test_preds_df.reset_index(drop=True)

# --------------------------------------------------
# 6. Attempt to load original URLs from PhiUSIIL.csv
# --------------------------------------------------
url_col = None
if os.path.exists("data/PhiUSIIL.csv"):
    raw_df = pd.read_csv("data/PhiUSIIL.csv")
    if "URL" in raw_df.columns:
        _, url_test, _, _ = train_test_split(
            raw_df["URL"], y_all, test_size=0.20, random_state=42, stratify=y_all
        )
        url_col = url_test.reset_index(drop=True)
        print("      Loaded original URLs from data/PhiUSIIL.csv")
    else:
        print("      'URL' column not found in PhiUSIIL.csv - URLs will be shown as N/A")
else:
    print("      data/PhiUSIIL.csv not found - URLs will be shown as N/A")

def get_url(idx):
    if url_col is not None and idx < len(url_col):
        return url_col.iloc[idx]
    return "N/A"

# --------------------------------------------------
# 7. Sample for SHAP computation
# --------------------------------------------------
print(f"\n[3/6] Sampling {SAMPLE_SIZE} test rows for SHAP (out of {len(X_test)}) ...")
rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
sample_idx = np.sort(sample_idx)

X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

# --------------------------------------------------
# 8. Compute SHAP values
# --------------------------------------------------
print(f"      Creating SHAP TreeExplainer for {type(model).__name__} ...")
explainer = shap.TreeExplainer(model)

print(f"      Computing SHAP values for {len(X_sample)} samples ...")
shap_output = explainer.shap_values(X_sample)

# Handle both list-of-arrays (older SHAP) and 3-D array (newer SHAP)
if isinstance(shap_output, list):
    shap_vals_phishing = shap_output[1]
    expected_value     = explainer.expected_value[1]
else:
    if shap_output.ndim == 3:
        shap_vals_phishing = shap_output[:, :, 1]
        expected_value     = explainer.expected_value[1]
    else:
        shap_vals_phishing = shap_output
        ev = explainer.expected_value
        expected_value = ev[0] if hasattr(ev, '__len__') else float(ev)

print(f"      SHAP values shape: {shap_vals_phishing.shape}")
print(f"      Base value (expected_value for phishing class): {expected_value:.4f}")

# --------------------------------------------------
# 9. Global feature importance table
# --------------------------------------------------
print("\n[4/6] Computing global feature importance (mean |SHAP|) ...")
mean_abs_shap = np.abs(shap_vals_phishing).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature":                  FEATURE_COLUMNS,
    "Mean Absolute SHAP Value": mean_abs_shap
}).sort_values("Mean Absolute SHAP Value", ascending=False).reset_index(drop=True)

importance_df.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"), index=False)
print(f"\n      Top 10 features by mean |SHAP| (class = Phishing):")
print(f"      {'Rank':<6} {'Feature':<22} {'Mean |SHAP|'}")
print(f"      {'----':<6} {'-------':<22} {'-----------'}")
for rank, row in importance_df.head(10).iterrows():
    print(f"      {rank+1:<6} {row['Feature']:<22} {row['Mean Absolute SHAP Value']:.6f}")

# --------------------------------------------------
# 10. SHAP Plots - Global
# --------------------------------------------------
print("\n[5/6] Generating SHAP plots ...")

# 10a. Beeswarm / Summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_vals_phishing,
    X_sample,
    feature_names=FEATURE_COLUMNS,
    show=False,
    max_display=18,
    plot_type="dot"
)
plt.title(
    f"SHAP Beeswarm Summary Plot - Phishing Detection\n"
    f"(Random Forest, {len(X_sample):,} test samples, class = Phishing)",
    fontsize=13, pad=12
)
plt.xlabel("SHAP Value  (positive = pushes toward Phishing)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: shap_outputs/shap_summary.png")

# 10b. Bar importance plot
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_vals_phishing,
    X_sample,
    feature_names=FEATURE_COLUMNS,
    show=False,
    max_display=18,
    plot_type="bar"
)
plt.title(
    f"SHAP Global Feature Importance - Phishing Detection\n"
    f"(Mean |SHAP Value|, {len(X_sample):,} samples, class = Phishing)",
    fontsize=13, pad=12
)
plt.xlabel("Mean |SHAP Value|  (higher = more influential)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: shap_outputs/shap_feature_importance.png")

# --------------------------------------------------
# 11. Individual / Local explanations
# --------------------------------------------------
print("\n[6/6] Generating individual (local) explanations ...")

sample_pred_df = pd.DataFrame({
    "sample_idx":    range(len(X_sample)),
    "actual":        y_sample.values,
    "predicted":     model.predict(X_sample),
    "phishing_prob": model.predict_proba(X_sample)[:, 1]
})

def find_case(actual, predicted, sample_df):
    mask = (sample_df["actual"] == actual) & (sample_df["predicted"] == predicted)
    hits = sample_df[mask]
    if len(hits) == 0:
        return None
    if actual == 1:
        return hits["phishing_prob"].idxmax()
    else:
        return hits["phishing_prob"].idxmin()

cases = {
    "legitimate_correct": find_case(actual=0, predicted=0, sample_df=sample_pred_df),
    "phishing_correct":   find_case(actual=1, predicted=1, sample_df=sample_pred_df),
    "false_positive":     find_case(actual=0, predicted=1, sample_df=sample_pred_df),
    "false_negative":     find_case(actual=1, predicted=0, sample_df=sample_pred_df),
}

label_names = {0: "Legitimate", 1: "Phishing"}

def print_local_explanation(case_name, test_orig_idx, row_features, shap_row, phish_prob, actual, predicted):
    url = get_url(test_orig_idx)
    print(f"\n  -- {case_name.upper().replace('_', ' ')} --")
    print(f"  URL              : {url}")
    print(f"  Actual           : {label_names[actual]}")
    print(f"  Predicted        : {label_names[predicted]}")
    print(f"  Phishing prob    : {phish_prob*100:.2f}%")

    feat_shap = sorted(zip(FEATURE_COLUMNS, shap_row), key=lambda x: x[1], reverse=True)
    pushing_phishing = [(f, v) for f, v in feat_shap if v > 1e-6]
    pushing_legit    = [(f, v) for f, v in reversed(feat_shap) if v < -1e-6]

    print(f"\n  Features INCREASING phishing prediction (positive SHAP):")
    if pushing_phishing:
        for feat, val in pushing_phishing[:5]:
            print(f"    + {feat:<22} SHAP={val:+.4f}  (value={row_features[feat]})")
    else:
        print("    (none)")

    print(f"\n  Features DECREASING phishing prediction (negative SHAP):")
    if pushing_legit:
        for feat, val in pushing_legit[:5]:
            print(f"    - {feat:<22} SHAP={val:+.4f}  (value={row_features[feat]})")
    else:
        print("    (none)")

def save_waterfall_plot(case_name, shap_row, row_features, phish_prob, actual, predicted):
    sorted_pairs = sorted(zip(FEATURE_COLUMNS, shap_row), key=lambda x: abs(x[1]), reverse=True)
    feats  = [f for f, _ in sorted_pairs[:10]]
    vals   = [v for _, v in sorted_pairs[:10]]
    labels = [f"{f} = {row_features[f]}" for f in feats]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value  (positive = toward Phishing, negative = toward Legitimate)", fontsize=10)
    title = (
        f"Local SHAP Explanation - {case_name.replace('_', ' ').title()}\n"
        f"Actual: {label_names[actual]} | Predicted: {label_names[predicted]} | "
        f"Phishing Prob: {phish_prob*100:.2f}%"
    )
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{case_name}_explanation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: shap_outputs/{case_name}_explanation.png")

print("\n" + "="*60)
print("  LOCAL (INDIVIDUAL) SHAP EXPLANATIONS")
print("="*60)

generated_cases = []
for case_name, df_idx in cases.items():
    if df_idx is None:
        print(f"\n  [SKIPPED] {case_name.upper()} - no matching sample found in the {SAMPLE_SIZE}-sample set.")
        continue

    row          = sample_pred_df.loc[df_idx]
    sample_i     = int(row["sample_idx"])
    actual       = int(row["actual"])
    predicted    = int(row["predicted"])
    phish_prob   = float(row["phishing_prob"])
    shap_row     = shap_vals_phishing[sample_i]
    row_features = X_sample.iloc[sample_i].to_dict()
    test_orig_idx = int(sample_idx[sample_i])

    print_local_explanation(
        case_name, test_orig_idx, row_features, shap_row, phish_prob, actual, predicted
    )
    save_waterfall_plot(case_name, shap_row, row_features, phish_prob, actual, predicted)
    generated_cases.append(case_name)

# --------------------------------------------------
# 12. Summary report
# --------------------------------------------------
print("\n" + "="*60)
print("  PHASE 3 SUMMARY")
print("="*60)
print(f"\n  SHAP ran successfully")
print(f"  Samples used for SHAP : {len(X_sample):,} (of {len(X_test):,} test samples)")
print(f"\n  Top 10 Features by Mean |SHAP| (Global, class = Phishing):")
for rank, row in importance_df.head(10).iterrows():
    print(f"    {rank+1:>2}. {row['Feature']:<22} {row['Mean Absolute SHAP Value']:.6f}")

print(f"\n  Generated files:")
generated_files = [
    os.path.join(OUTPUT_DIR, "shap_summary.png"),
    os.path.join(OUTPUT_DIR, "shap_feature_importance.png"),
    os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"),
] + [os.path.join(OUTPUT_DIR, f"{c}_explanation.png") for c in generated_cases]

for f in generated_files:
    exists = "OK" if os.path.exists(f) else "MISSING"
    print(f"    [{exists}] {f}")

print(f"\n  Individual explanations generated: {len(generated_cases)}")
for c in generated_cases:
    print(f"    - {c.replace('_', ' ')}")

print(f"\n  NOTES / LIMITATIONS:")
print(f"    * SHAP values computed for class 1 (Phishing) only.")
print(f"    * Positive SHAP value  -> feature pushes prediction TOWARD Phishing.")
print(f"    * Negative SHAP value  -> feature pushes prediction TOWARD Legitimate.")
print(f"    * SHAP explains the model's behavior, NOT real-world causation.")
print(f"    * A sample of {SAMPLE_SIZE:,} rows was used (not all {len(X_test):,} test rows)")
print(f"      to keep computation and plot sizes manageable.")
print(f"\n  Phase 3 complete.\n")
