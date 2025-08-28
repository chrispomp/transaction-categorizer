# scripts/4_deploy_to_existing_endpoint.py
import os
from google.cloud import aiplatform

# --- Configuration ---
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = "us-central1"
MODEL_DISPLAY_NAME = "transaction-categorizer-skl"

# --- PASTE YOUR IDs FROM THE PREVIOUS SCRIPT'S OUTPUT HERE ---
MODEL_RESOURCE_NAME = "projects/925334697476/locations/us-central1/models/8891314027708284928"
ENDPOINT_RESOURCE_NAME = "projects/925334697476/locations/us-central1/endpoints/6699776247318183936"
# ----------------------------------------------------------------

print(f"Using Project ID: {PROJECT_ID}")
if not all([PROJECT_ID, MODEL_RESOURCE_NAME, ENDPOINT_RESOURCE_NAME]):
    raise ValueError("Please fill in the PROJECT_ID, MODEL_RESOURCE_NAME, and ENDPOINT_RESOURCE_NAME variables.")

print("Initializing Vertex AI client...")
aiplatform.init(project=PROJECT_ID, location=REGION)

# Get a reference to the existing Model and Endpoint
model = aiplatform.Model(model_name=MODEL_RESOURCE_NAME)
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_RESOURCE_NAME)

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