import joblib
import pandas as pd
from feature_engineering import extract_features

FEATURE_COLUMNS = [
    'url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'num_at',
    'num_percent', 'num_question', 'num_equals', 'num_ampersand', 'uses_https',
    'has_ip', 'num_subdomains', 'domain_length', 'has_login', 'has_verify',
    'has_secure', 'has_account', 'has_update'
]

def main():
    # 1. Load trained model
    try:
        model = joblib.load("phishing_model.pkl")
    except Exception as e:
        print(f"Error loading model 'phishing_model.pkl': {e}")
        return

    # 3. Test URLs covering all required categories
    test_urls = [
        # Normal legitimate HTTPS URL
        ("Normal legitimate HTTPS URL", "https://www.google.com"),
        # Legitimate-looking URL with a login page
        ("Legitimate-looking URL with a login page", "https://accounts.google.com/login"),
        # Suspicious URL containing login/verify/account
        ("Suspicious URL containing login/verify/account", "http://secure-verify-account-update.login-bank-alert.com/login.php?user=verify&account=update"),
        # HTTP URL with multiple subdomains
        ("HTTP URL with multiple subdomains", "http://login.service.update.verify.bank.com/account/login.html"),
        # IP-address-based URL
        ("IP-address-based URL", "http://192.168.1.1/login/verify.html")
    ]

    print("==================================================")
    print("           MODEL PREDICTION TEST RESULTS          ")
    print("==================================================")

    for category, url in test_urls:
        try:
            # 4. Extract features and format DataFrame with exact column names and order
            features_dict = extract_features(url)
            df_features = pd.DataFrame([features_dict])[FEATURE_COLUMNS]

            # Predict class and probability
            prediction = model.predict(df_features)[0]
            
            if hasattr(model, "predict_proba"):
                # Class 1 is assumed to be Phishing (0 = Legitimate, 1 = Phishing)
                proba_phishing = model.predict_proba(df_features)[0][1]
                proba_str = f"{proba_phishing * 100:.2f}%"
            else:
                proba_str = "N/A (predict_proba not available)"

            pred_label = "PHISHING" if prediction == 1 else "LEGITIMATE"

            # 5. Print results in clear format
            print(f"\nCategory: {category}")
            print(f"URL: {url}")
            print(f"Prediction: {pred_label}")
            print(f"Phishing probability: {proba_str}")

        except Exception as e:
            print(f"\nCategory: {category}")
            print(f"URL: {url}")
            print(f"Error evaluating URL: {e}")

    print("\n==================================================")

if __name__ == "__main__":
    main()
