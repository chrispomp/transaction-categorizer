bq mk --table \
    --schema schema.json \
    fsi-banking-agentspace.equifax_txns.golden_dataset_corrections


gcloud auth application-default login

python3 -m venv .venv
source .venv/bin/activate

pip install google-cloud-bigquery pandas scikit-learn xgboost joblib google-cloud-aiplatform gcsfs

pip install -r requirements.txt

export PROJECT_ID="fsi-banking-agentspace"
gcloud config set project $PROJECT_ID

export BUCKET_NAME="gs://${PROJECT_ID}-ml-artifacts"
gsutil mb -p $PROJECT_ID -l us-central1 $BUCKET_NAME


python scripts/1_create_features.py

python scripts/2_train_model.py

python scripts/3_deploy_model.py


gcloud functions deploy transaction-processor \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=./cloud_function \
  --entry-point=transaction_processor \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars="GCP_PROJECT=fsi-banking-agentspace,BQ_DATASET=fsi-banking-agentspace.equifax_txns,VERTEX_ML_ENDPOINT_ID=projects/fsi-banking-agentspace/locations/us-central1/endpoints/1234567890123456789"
