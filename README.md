### Stage 1: Environment Setup & Authentication

First, let's get your local environment ready and connect it to Google Cloud.

1.  **Authenticate with Google Cloud:** This command will open a browser window for you to log in and grant the necessary permissions to your Google Cloud account.

    ```bash
    gcloud auth application-default login
    ```

2.  **Create a Virtual Environment:** This isolates your project's Python dependencies.

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:** Install all the required Python libraries from your `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

-----

### Stage 2: Configure Google Cloud Project

Now, let's configure your project ID and create a storage bucket for your machine learning artifacts.

1.  **Set Your Project ID:** Replace `"fsi-banking-agentspace"` with your actual Google Cloud Project ID. This command sets it as an environment variable and configures the `gcloud` CLI to use it by default.

    ```bash
    export PROJECT_ID="fsi-banking-agentspace"
    gcloud config set project $PROJECT_ID
    ```

2.  **Create a Google Cloud Storage Bucket:** This bucket will store your trained model files. The command uses the project ID you just set to create a uniquely named bucket.

    ```bash
    export BUCKET_NAME="gs://${PROJECT_ID}-ml-artifacts"
    gsutil mb -p $PROJECT_ID -l us-central1 $BUCKET_NAME
    ```

-----

### Stage 3: Prepare BigQuery

Create the BigQuery table that will store the corrected transaction classifications.

  * **Create the Corrections Table:** This command uses the `schema.json` file to define the structure of a new table in your BigQuery dataset.
    ```bash
    bq mk --table \
        --schema schema.json \
        fsi-banking-agentspace.equifax_txns.golden_dataset_corrections
    ```

-----

### Stage 4: Run the ML Pipeline Scripts

Execute the Python scripts in order to create features, train the model, and deploy it.

1.  **Create Features:** This script will likely process your raw data and prepare it for training.

    ```bash
    python scripts/1_create_features.py
    ```

2.  **Train the Model:** This script trains the XGBoost classifier on the features you just created and saves the final model as `model.joblib` in the `ml_assets` folder.

    ```bash
    python scripts/2_train_model.py
    ```

-----

### Stage 5: Deploy the Model to Vertex AI

This is a critical step. You'll run the deployment script and need to **save the output** for the next stage.

1.  **Run the Deployment Script:** This script uploads your `model.joblib` file to Vertex AI, creates a new endpoint, and deploys the model to that endpoint.

    ```bash
    python scripts/3_deploy_model.py
    ```

2.  **‼️ Important: Copy the Endpoint ID:** After the script finishes, it will print the resource name of your new endpoint. It will look something like this:
    `projects/fsi-banking-agentspace/locations/us-central1/endpoints/1234567890123456789`
    **Copy this entire string.** You will need it in the final step.

-----

### Stage 6: Deploy the Cloud Function

Before running the final command, you must make the code corrections discussed previously.

1.  **Prepare the Cloud Function Code:**

      * **Copy ML Assets:** Copy the entire `ml_assets` folder into the `cloud_function` directory. Your Cloud Function needs these `.pkl` files to process the prediction data.
      * **Correct `main.py`:** Make the two required code changes to your `cloud_function/main.py` file:
          * Fix the environment variable lookup.
          * Add the feature engineering logic to the `_execute_phase_2_ml` function.

2.  **Deploy the Function:** Use the command below, but **replace the placeholder** with the actual endpoint ID you copied in Stage 5.

    ```bash
    gcloud functions deploy transaction-processor \
      --gen2 \
      --runtime=python311 \
      --region=us-central1 \
      --source=./cloud_function \
      --entry-point=transaction_processor \
      --trigger-http \
      --allow-unauthenticated \
      --set-env-vars="GCP_PROJECT=fsi-banking-agentspace,BQ_DATASET=fsi-banking-agentspace.equifax_txns,VERTEX_ML_ENDPOINT_ID=PASTE_YOUR_ENDPOINT_ID_HERE"
    ```

After completing these steps, your entire end-to-end transaction categorization service will be deployed and ready to use.