import os
import pandas as pd
from google.cloud import bigquery
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# --- Configuration ---
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = "fsi-banking-agentspace.equifax_txns"
SOURCE_TABLE = f"{DATASET}.transactions"
SAMPLE_LIMIT = 20000
ASSETS_DIR = "ml_assets" # Load assets from and save to this folder

print(f"Using Project ID: {PROJECT_ID}")
if not PROJECT_ID:
    raise ValueError("PROJECT_ID environment variable not set.")

# --- 1. Load Data ---
print("Loading data...")
bq_client = bigquery.Client(project=PROJECT_ID)
query = f"""
    SELECT
        description_cleaned, merchant_name, amount, channel, account_type,
        CONCAT(category_l1, '||', category_l2) as target_category
    FROM `{SOURCE_TABLE}`
    WHERE category_l1 IS NOT NULL AND category_l2 IS NOT NULL
    LIMIT {SAMPLE_LIMIT}
"""
df = bq_client.query(query).to_dataframe()
df['text_features'] = df['description_cleaned'].fillna('') + ' ' + df['merchant_name'].fillna('')
print(f"Loaded {len(df)} rows.")

# --- Filter out categories with only one example ---
print(f"Original dataset size: {len(df)}")
value_counts = df['target_category'].value_counts()
to_remove = value_counts[value_counts < 2].index
df = df[~df['target_category'].isin(to_remove)]
print(f"Filtered dataset size (removed categories with < 2 examples): {len(df)}")

# --- 2. Load Transformers and Engineer Features ---
print("Loading transformers and applying feature transformations...")
vectorizer = joblib.load(os.path.join(ASSETS_DIR, 'tfidf_vectorizer.pkl'))
scaler = joblib.load(os.path.join(ASSETS_DIR, 'scaler.pkl'))
encoder = joblib.load(os.path.join(ASSETS_DIR, 'onehot_encoder.pkl'))

X_text = vectorizer.transform(df['text_features'])
X_numeric = scaler.transform(df[['amount']])
X_categorical = encoder.transform(df[['channel', 'account_type']])

X = hstack([X_text, X_numeric, X_categorical]).tocsr()
y, labels = pd.factorize(df['target_category'])

joblib.dump(labels, os.path.join(ASSETS_DIR, 'labels.pkl'))
print("✅ Saved labels.pkl")

# --- 3. Train Model ---
print("Training XGBoost model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(labels),
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1
)
model.fit(X_train, y_train)

# --- 4. Evaluate and Save ---
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# CHANGED: Use joblib.dump instead of model.save_model
joblib.dump(model, os.path.join(ASSETS_DIR, 'model.joblib'))
print(f"✅ Saved trained model to {ASSETS_DIR}/model.joblib") # Note the new .joblib extension
print("\n🎉 Model training complete!")