import sys
from urllib.parse import urlparse
import joblib
import pandas as pd
import shap
from feature_engineering import extract_features

FEATURE_COLUMNS = [
    'url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'num_at',
    'num_percent', 'num_question', 'num_equals', 'num_ampersand', 'uses_https',
    'has_ip', 'num_subdomains', 'domain_length', 'has_login', 'has_verify',
    'has_secure', 'has_account', 'has_update'
]

def normalize_url(url: str) -> str:
    """
    Standardize real-world user URLs into the canonical format expected by the model:
    1. Prepend 'https://' if scheme is omitted.
    2. Strip trailing root slash.
    3. Ensure standard apex domains (e.g. 'example.com') include 'www.' to match
       standard legitimate URL training distribution.
    """
    u = url.strip()
    if not u:
        return u

    if not u.startswith(('http://', 'https://')):
        u = 'https://' + u

    parsed = urlparse(u)
    scheme = parsed.scheme
    netloc = parsed.netloc or parsed.path.split('/')[0]
    path = parsed.path if parsed.netloc else ('/' + '/'.join(parsed.path.split('/')[1:]) if '/' in parsed.path else '')

    if path == '/':
        path = ''

    parts = [p for p in netloc.split('.') if p]
    if len(parts) == 2 and not any(part.isdigit() for part in parts):
        netloc = 'www.' + netloc

    normalized = f"{scheme}://{netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"

    return normalized

def check_url(raw_url: str, model, explainer):
    normalized = normalize_url(raw_url)
    feat_dict = extract_features(normalized)
    features_df = pd.DataFrame([feat_dict])[FEATURE_COLUMNS]
    
    # Model prediction and probability
    prob = model.predict_proba(features_df)[0][1]
    prediction = "PHISHING" if prob >= 0.5 else "LEGITIMATE"

    # Compute SHAP values for this specific single URL
    shap_vals = explainer.shap_values(features_df)
    
    # Handle SHAP output dimensions across different SHAP versions
    if isinstance(shap_vals, list):
        # Class 1 = Phishing
        shap_for_url = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        shap_for_url = shap_vals[0, :, 1]
    else:
        shap_for_url = shap_vals[0]

    # Pair features with their SHAP impact and raw values
    feat_impact = []
    for feat, s_val in zip(FEATURE_COLUMNS, shap_for_url):
        feat_impact.append((feat, feat_dict[feat], s_val))

    # Features increasing phishing prediction (positive SHAP)
    pushing_phishing = sorted([item for item in feat_impact if item[2] > 0.005], key=lambda x: x[2], reverse=True)
    # Features decreasing phishing prediction (negative SHAP)
    pushing_legit = sorted([item for item in feat_impact if item[2] < -0.005], key=lambda x: x[2])

    print("\n" + "=" * 60)
    print(f"Input URL   : {raw_url}")
    if normalized != raw_url:
        print(f"Evaluated As: {normalized}")
    print(f"Prediction  : {prediction}")
    print(f"Phishing Probability : {prob * 100:.2f}%")
    print("-" * 60)
    print("SHAP EXPLANATION (Why did the model make this decision?):")

    if pushing_phishing:
        print("\n  [+] Features pushing towards PHISHING:")
        for feat, val, s_val in pushing_phishing[:5]:
            print(f"      * {feat:<16} = {str(val):<5} (SHAP impact: +{s_val:.4f})")
    else:
        print("\n  [+] Features pushing towards PHISHING: None significant")

    if pushing_legit:
        print("\n  [-] Features pushing towards LEGITIMATE:")
        for feat, val, s_val in pushing_legit[:5]:
            print(f"      * {feat:<16} = {str(val):<5} (SHAP impact: {s_val:.4f})")
    else:
        print("\n  [-] Features pushing towards LEGITIMATE: None significant")

    print("=" * 60 + "\n")

def main():
    print("Loading model and initializing SHAP TreeExplainer...")
    model = joblib.load("phishing_model.pkl")
    explainer = shap.TreeExplainer(model)
    print("Ready!")

    if len(sys.argv) > 1:
        raw_url = sys.argv[1].strip()
        check_url(raw_url, model, explainer)
        return

    print("\n=== Phishing URL Detector with SHAP Explainability ===")
    print("Enter any URL to test. (Type 'q' or press Ctrl+C to quit)\n")

    while True:
        try:
            url = input("Enter URL: ").strip()
            if not url:
                continue
            if url.lower() in ('q', 'quit', 'exit'):
                print("Exiting.")
                break
            check_url(url, model, explainer)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
