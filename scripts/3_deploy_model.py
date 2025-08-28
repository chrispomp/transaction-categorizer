import os
from google.cloud import aiplatform

# --- Configuration ---
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = "us-central1"
BUCKET_URI = f"gs://{PROJECT_ID}-ml-artifacts"
MODEL_DISPLAY_NAME = "transaction-categorizer-skl"
LOCAL_MODEL_FILE_PATH = "ml_assets/model.joblib" # Corrected path for final script

print(f"Using Project ID: {PROJECT_ID}")
if not PROJECT_ID:
    raise ValueError("PROJECT_ID environment variable not set.")

print("Initializing Vertex AI client...")
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

# --- 1. Upload the model to Vertex AI Model Registry ---
print(f"Uploading scikit-learn model '{MODEL_DISPLAY_NAME}'...")
model = aiplatform.Model.upload_scikit_learn_model_file(
    model_file_path=LOCAL_MODEL_FILE_PATH,
    display_name=MODEL_DISPLAY_NAME,
    sync=True,
)
print(f"Model registered. Resource name: {model.resource_name}")

# --- 2. Create a Vertex AI Endpoint ---
print(f"Creating endpoint '{MODEL_DISPLAY_NAME}-endpoint'...")
# Check if an endpoint with the same name already exists to avoid errors on re-runs
endpoints = aiplatform.Endpoint.list(
    filter=f'display_name="{MODEL_DISPLAY_NAME}-endpoint"',
    order_by="create_time desc",
    project=PROJECT_ID,
    location=REGION,
)
if endpoints:
    endpoint = endpoints[0]
    print("Endpoint already exists.")
else:
    endpoint = aiplatform.Endpoint.create(
        display_name=f"{MODEL_DISPLAY_NAME}-endpoint",
        project=PROJECT_ID,
        location=REGION,
        sync=True,
    )
print(f"Endpoint created/found. Resource name: {endpoint.resource_name}")

# --- 3. Deploy the Model to the Endpoint ---
print(f"Deploying model to endpoint...")
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=MODEL_DISPLAY_NAME,
    machine_type="n1-standard-2",
    sync=True
)

print("\n🎉 Model deployment complete!")
print("\nIMPORTANT: Your Vertex AI Endpoint ID for the Cloud Function is:")
print(endpoint.resource_name)