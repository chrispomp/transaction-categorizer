import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from google.cloud import bigquery
import joblib

# --- Configuration ---
# Ensure you have run 'gcloud auth application-default login'
# and have set your project ID locally: export PROJECT_ID="your-gcp-project-id"
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = "fsi-banking-agentspace.equifax_txns"
SOURCE_TABLE = f"{DATASET}.transactions"
SAMPLE_LIMIT = 20000 # Use a representative sample of your data
OUTPUT_DIR = "ml_assets" # Save assets to the ml_assets folder

print(f"Using Project ID: {PROJECT_ID}")
if not PROJECT_ID:
    raise ValueError("PROJECT_ID environment variable not set.")

# --- 1. Load Data ---
print("Loading data from BigQuery...")
bq_client = bigquery.Client(project=PROJECT_ID)
query = f"""
    SELECT
        description_cleaned, merchant_name, amount, channel, account_type
    FROM `{SOURCE_TABLE}`
    WHERE category_l1 IS NOT NULL AND category_l2 IS NOT NULL # Only use labeled data
    LIMIT {SAMPLE_LIMIT}
"""
df = bq_client.query(query).to_dataframe()
df['text_features'] = df['description_cleaned'].fillna('') + ' ' + df['merchant_name'].fillna('')
print(f"Loaded {len(df)} rows.")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Define, Fit, and Save Transformers ---
print("Fitting and saving transformers...")

# Fit and save the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000).fit(df['text_features'])
joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.pkl'))
print("✅ Saved tfidf_vectorizer.pkl")

# Fit and save the StandardScaler
scaler = StandardScaler().fit(df[['amount']])
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
print("✅ Saved scaler.pkl")

# Fit and save the OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore').fit(df[['channel', 'account_type']])
joblib.dump(encoder, os.path.join(OUTPUT_DIR, 'onehot_encoder.pkl'))
print("✅ Saved onehot_encoder.pkl")

print(f"\n🎉 Feature engineering setup complete! Assets saved in '{OUTPUT_DIR}'.")