import functions_framework
from flask import jsonify
import os
import logging
import json
import uuid
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
import asyncio

# Google Cloud & ML Libraries
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import pandas as pd
import joblib # For loading scikit-learn transformers

# --- Configuration & Initialization ---

# Set up structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetch environment variables
PROJECT_ID = os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_REGION", "us-central1")
DATASET_ID = os.environ.get("BQ_DATASET", "fsi-banking-agentspace.equifax_txns")
ML_ENDPOINT_ID = os.environ.get("projects/925334697476/locations/us-central1/endpoints/6699776247318183936") # e.g., "projects/123/locations/us-central1/endpoints/456"

# Ensure required environment variables are set
if not all([PROJECT_ID, ML_ENDPOINT_ID]):
    raise ValueError("Missing required environment variables: GCP_PROJECT, VERTEX_ML_ENDPOINT_ID")

# Initialize GCP clients
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    # The endpoint resource is retrieved using the ID
    ml_endpoint = vertexai.aiplatform.Endpoint(ML_ENDPOINT_ID)
    # Initialize the Generative Model
    llm_model = GenerativeModel("gemini-1.5-flash-001")
    logging.info("Successfully initialized BigQuery and Vertex AI clients.")
except Exception as e:
    logging.error(f"Failed to initialize GCP clients: {e}")
    raise

# Load pre-trained feature engineering transformers
# These .pkl files must be bundled with the function source code
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('onehot_encoder.pkl')
    logging.info("Successfully loaded ML feature transformers.")
except FileNotFoundError as e:
    logging.error(f"Could not find a required .pkl file: {e}. Ensure transformers are bundled with the function.")
    raise

# --- Constants ---
TRANSACTIONS_TABLE = f"{DATASET_ID}.transactions"
CORRECTIONS_TABLE = f"{DATASET_ID}.golden_dataset_corrections"


# --- Helper Functions ---

def _build_scope_filter(params: dict) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
    """Builds the WHERE clause and query parameters for BQ queries."""
    filters = []
    query_params = []

    consumer_name = params.get('consumer_name')
    if consumer_name and consumer_name.lower() != 'all consumers':
        filters.append("consumer_name = @consumer_name")
        query_params.append(bigquery.ScalarQueryParameter("consumer_name", "STRING", consumer_name))

    time_period = params.get('time_period')
    now = datetime.now(timezone.utc)
    if time_period == 'last_3_months':
        start_date = now - relativedelta(months=3)
        filters.append("transaction_date >= @start_date")
        query_params.append(bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date))
    elif time_period == 'last_6_months':
        start_date = now - relativedelta(months=6)
        filters.append("transaction_date >= @start_date")
        query_params.append(bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date))
    elif time_period == 'custom' and params.get('start_date') and params.get('end_date'):
        filters.append("transaction_date BETWEEN @start_date AND @end_date")
        query_params.append(bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", params['start_date']))
        query_params.append(bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", params['end_date']))
    
    return " AND ".join(filters), query_params

def _run_bq_query(query: str, query_params: list = None) -> bigquery.table.RowIterator:
    """Executes a BigQuery query with parameters and returns the results."""
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    try:
        query_job = bq_client.query(query, job_config=job_config)
        logging.info(f"Executing BQ query (first 100 chars): {query[:100]}")
        return query_job.result()
    except Exception as e:
        logging.error(f"BigQuery query failed: {e}")
        raise

# --- Phase 1: Rules Engine ---

def _execute_phase_1_rules(scope_filter: str, scope_params: list) -> int:
    """Executes the deterministic rule-based categorization using BQ MERGE."""
    rules = [
        # Rule 1: Income Categorization
        f"""
            MERGE `{TRANSACTIONS_TABLE}` T
            USING (SELECT transaction_id FROM `{TRANSACTIONS_TABLE}` WHERE transaction_type = 'Credit' AND amount > 0 AND category_l1 IS NULL AND {scope_filter}) S
            ON T.transaction_id = S.transaction_id
            WHEN MATCHED THEN UPDATE SET category_l1 = 'Income', category_l2 = 'Income'
        """,
        # Rule 2: Internal Transfers
        f"""
            MERGE `{TRANSACTIONS_TABLE}` T
            USING (SELECT transaction_id FROM `{TRANSACTIONS_TABLE}` WHERE (channel = 'Internal Transfer' OR description_cleaned LIKE '%transfer from%' OR description_cleaned LIKE '%transfer to%') AND category_l1 IS NULL AND {scope_filter}) S
            ON T.transaction_id = S.transaction_id
            WHEN MATCHED THEN UPDATE SET category_l1 = 'Transfer', category_l2 = 'Internal Transfer'
        """,
        # Rule 3: P2P Transfers
        f"""
            MERGE `{TRANSACTIONS_TABLE}` T
            USING (SELECT transaction_id FROM `{TRANSACTIONS_TABLE}` WHERE (channel = 'P2P' OR merchant_name IN ('Venmo', 'Cash App', 'Zelle', 'PayPal')) AND category_l1 IS NULL AND {scope_filter}) S
            ON T.transaction_id = S.transaction_id
            WHEN MATCHED THEN UPDATE SET category_l1 = 'Transfer', category_l2 = 'Peer to Peer Transfer'
        """,
    ]
    
    total_updated = 0
    for i, rule_sql in enumerate(rules):
        logging.info(f"Running categorization rule #{i+1}...")
        results = _run_bq_query(rule_sql, scope_params)
        logging.info(f"Rule #{i+1} executed.")
    return total_updated

# --- Phase 2: ML Classification ---

def _execute_phase_2_ml(scope_filter: str, scope_params: list) -> int:
    """Fetches uncategorized data, gets predictions, and updates BQ."""
    query = f"""
        SELECT transaction_id, description_cleaned, merchant_name, amount, channel, account_type
        FROM `{TRANSACTIONS_TABLE}`
        WHERE category_l1 IS NULL AND {scope_filter}
    """
    df = _run_bq_query(query, scope_params).to_dataframe()

    if df.empty:
        logging.info("Phase 2: No transactions to process with ML model.")
        return 0

    # This example assumes the endpoint expects raw JSON and handles featurization internally
    # or that the model was trained on these raw feature names.
    instances = df.to_dict('records')
    
    logging.info(f"Phase 2: Sending {len(instances)} instances to Vertex AI Endpoint.")
    predictions = ml_endpoint.predict(instances=instances)
    
    updates = []
    for pred_obj, original_row in zip(predictions.predictions, df.itertuples()):
        pred_dict = dict(pred_obj)
        if pred_dict.get('confidence', 0) > 0.90:
            updates.append({
                "transaction_id": original_row.transaction_id,
                "category_l1": pred_dict['category_l1'],
                "category_l2": pred_dict['category_l2']
            })
            
    if not updates:
        logging.info("Phase 2: No high-confidence predictions to update.")
        return 0

    updates_df = pd.DataFrame(updates)
    temp_table_id = f"{DATASET_ID}.temp_updates_{uuid.uuid4().hex}"
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    bq_client.load_table_from_dataframe(updates_df, temp_table_id, job_config=job_config).result()

    merge_sql = f"""
        MERGE `{TRANSACTIONS_TABLE}` T
        USING `{temp_table_id}` S
        ON T.transaction_id = S.transaction_id
        WHEN MATCHED THEN
            UPDATE SET T.category_l1 = S.category_l1, T.category_l2 = S.category_l2
    """
    _run_bq_query(merge_sql)
    logging.info(f"Phase 2: Updated {len(updates)} transactions from ML model.")
    
    bq_client.delete_table(temp_table_id, not_found_ok=True)
    return len(updates)

# --- Phase 3: LLM Reasoning ---

async def _get_llm_categorization(row: pd.Series) -> dict | None:
    """Creates a prompt and gets categorization from the Gemini model."""
    prompt = f"""
        You are a financial expert. Analyze the following transaction and determine the most accurate 'category_l1' and 'category_l2'.

        **Available Categories:**
        - category_l1: 'Income', 'Expense', 'Transfer'
        - category_l2: 'Groceries', 'Dining & Drinks', 'Shopping', 'Software & Tech', 'Income', 'Interest Income', 'Refund', 'Peer to Peer Transfer', 'Bills & Utilities', 'Entertainment & Rec.', 'Health & Wellness', 'Auto & Transport', 'Travel & Vacation', 'Medical', 'Insurance'

        **Consumer Context:**
        - Persona: {row.get('persona_type', 'Unknown')}

        **Transaction Data:**
        - Description: {row.get('description_raw', '')}
        - Merchant: {row.get('merchant_name', '')}
        - Amount: {row.get('amount', 0.0)}
        - Type: {row.get('transaction_type', '')}
        - Channel: {row.get('channel', '')}

        **Instructions:**
        1. Consider the consumer's persona. An 'Adobe' purchase for a 'Freelance Creative' is likely a business 'Expense', not personal.
        2. Analyze the merchant and description carefully.
        3. Return ONLY a single, valid JSON object with the keys "category_l1" and "category_l2".

        **Example Output:**
        {{"category_l1": "Expense", "category_l2": "Software & Tech"}}
    """
    try:
        response = await llm_model.generate_content_async(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        result = json.loads(response.text)
        result['transaction_id'] = row.transaction_id
        return result
    except Exception as e:
        logging.warning(f"LLM generation failed for transaction {row.transaction_id}: {e}")
        return None

async def _execute_phase_3_llm(scope_filter: str, scope_params: list) -> int:
    """Handles the final LLM-based categorization for remaining transactions."""
    query = f"""
        SELECT transaction_id, persona_type, description_raw, merchant_name, amount, transaction_type, channel
        FROM `{TRANSACTIONS_TABLE}`
        WHERE category_l1 IS NULL AND {scope_filter}
    """
    df = _run_bq_query(query, scope_params).to_dataframe()

    if df.empty:
        logging.info("Phase 3: No transactions remaining for LLM processing.")
        return 0

    logging.info(f"Phase 3: Processing {len(df)} transactions with Gemini.")
    
    tasks = [_get_llm_categorization(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    
    updates = [res for res in results if res and 'category_l1' in res and 'category_l2' in res]

    if not updates:
        logging.info("Phase 3: No valid categorizations received from LLM.")
        return 0

    updates_df = pd.DataFrame(updates)
    temp_table_id = f"{DATASET_ID}.temp_llm_updates_{uuid.uuid4().hex}"
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    bq_client.load_table_from_dataframe(updates_df, temp_table_id, job_config=job_config).result()

    merge_sql = f"""
        MERGE `{TRANSACTIONS_TABLE}` T
        USING `{temp_table_id}` S
        ON T.transaction_id = S.transaction_id
        WHEN MATCHED THEN
            UPDATE SET T.category_l1 = S.category_l1, T.category_l2 = S.category_l2
    """
    _run_bq_query(merge_sql)
    logging.info(f"Phase 3: Updated {len(updates)} transactions from LLM.")
    
    bq_client.delete_table(temp_table_id, not_found_ok=True)
    return len(updates)


# --- Main Handler Functions ---

def handle_categorization(params: dict):
    """Orchestrates the three-phase categorization process."""
    scope_filter, scope_params = _build_scope_filter(params)
    if not scope_filter:
        return {"error": "Invalid scope. Please provide a consumer or time period."}
        
    count_query = f"SELECT COUNT(*) as total FROM `{TRANSACTIONS_TABLE}` WHERE category_l1 IS NULL AND {scope_filter}"
    initial_uncategorized = next(_run_bq_query(count_query, scope_params))['total']
    
    _execute_phase_1_rules(scope_filter, scope_params)
    count_after_p1 = next(_run_bq_query(count_query, scope_params))['total']
    categorized_by_rules = initial_uncategorized - count_after_p1
    
    categorized_by_ml = _execute_phase_2_ml(scope_filter, scope_params)
    
    categorized_by_llm = asyncio.run(_execute_phase_3_llm(scope_filter, scope_params))
    
    final_uncategorized = next(_run_bq_query(count_query, scope_params))['total']
    
    summary = {
        "status": "success",
        "initial_uncategorized_count": int(initial_uncategorized),
        "categorized_by_rules": int(categorized_by_rules),
        "categorized_by_ml": int(categorized_by_ml),
        "categorized_by_llm": int(categorized_by_llm),
        "final_uncategorized_count": int(final_uncategorized)
    }
    logging.info(f"Categorization job complete. Summary: {summary}")
    return summary

def handle_audit(params: dict):
    """Runs a series of data integrity checks."""
    scope_filter, scope_params = _build_scope_filter(params)
    if not scope_filter:
        return {"error": "Invalid scope for audit."}

    audit_query = f"""
        SELECT transaction_id, 'Credit transaction with negative amount' AS reason FROM `{TRANSACTIONS_TABLE}` WHERE transaction_type = 'Credit' AND amount < 0 AND {scope_filter}
        UNION ALL
        SELECT transaction_id, 'Debit transaction with positive amount' AS reason FROM `{TRANSACTIONS_TABLE}` WHERE transaction_type = 'Debit' AND amount > 0 AND {scope_filter}
        UNION ALL
        SELECT transaction_id, 'Income category with Debit transaction type' AS reason FROM `{TRANSACTIONS_TABLE}` WHERE category_l1 = 'Income' AND transaction_type = 'Debit' AND {scope_filter}
        UNION ALL
        SELECT transaction_id, 'Merchant-Category Mismatch (Netflix not Entertainment)' AS reason FROM `{TRANSACTIONS_TABLE}` WHERE merchant_name = 'Netflix' AND category_l2 <> 'Entertainment & Rec.' AND {scope_filter}
    """
    results = _run_bq_query(audit_query, scope_params)
    issues = [dict(row) for row in results]
    
    summary = {
        "status": "success",
        "audit_issues_found": len(issues),
        "issues": issues
    }
    logging.info(f"Audit complete. Found {len(issues)} issues.")
    return summary

def handle_correction(params: dict):
    """Handles a user-submitted correction for a single transaction."""
    txn_id = params.get('transaction_id')
    new_l1 = params.get('new_category_l1')
    new_l2 = params.get('new_category_l2')

    if not all([txn_id, new_l1, new_l2]):
        return {"error": "Missing parameters for correction."}
        
    query = f"SELECT category_l1, category_l2 FROM `{TRANSACTIONS_TABLE}` WHERE transaction_id = @txn_id"
    query_params = [bigquery.ScalarQueryParameter("txn_id", "STRING", txn_id)]
    try:
        original = next(_run_bq_query(query, query_params))
    except StopIteration:
        return {"error": f"Transaction ID '{txn_id}' not found."}

    update_sql = f"""
        UPDATE `{TRANSACTIONS_TABLE}`
        SET category_l1 = @new_l1, category_l2 = @new_l2
        WHERE transaction_id = @txn_id
    """
    update_params = [
        bigquery.ScalarQueryParameter("new_l1", "STRING", new_l1),
        bigquery.ScalarQueryParameter("new_l2", "STRING", new_l2),
        bigquery.ScalarQueryParameter("txn_id", "STRING", txn_id),
    ]
    _run_bq_query(update_sql, update_params)
    
    correction_record = {
        "correction_id": str(uuid.uuid4()),
        "transaction_id": txn_id,
        "corrected_by": "user_placeholder",
        "correction_timestamp": datetime.now(timezone.utc).isoformat(),
        "previous_category_l1": original['category_l1'],
        "previous_category_l2": original['category_l2'],
        "corrected_category_l1": new_l1,
        "corrected_category_l2": new_l2
    }
    errors = bq_client.insert_rows_json(CORRECTIONS_TABLE, [correction_record])
    if errors:
        logging.error(f"Failed to insert correction record: {errors}")
        return {"error": "Failed to log correction."}

    logging.info(f"Successfully corrected transaction {txn_id}.")
    return {"status": "success", "message": f"Transaction {txn_id} corrected."}

# --- Cloud Function Entry Point ---

@functions_framework.http
def transaction_processor(request):
    """Main entry point for all agent tool calls."""
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return jsonify({'error': 'Invalid JSON payload'}), 400
            
        operation = request_json.get('operation')
        params = request_json.get('params')

        if operation == 'categorize':
            summary = handle_categorization(params)
        elif operation == 'audit':
            summary = handle_audit(params)
        elif operation == 'correct':
            summary = handle_correction(params)
        else:
            return jsonify({'error': 'Invalid operation specified'}), 400

        return jsonify(summary)

    except Exception as e:
        logging.error(f"An unhandled error occurred: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500