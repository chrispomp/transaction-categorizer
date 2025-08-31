# app.py

import json
import logging
from typing import Dict, Any

from google.cloud import bigquery
# Correct ADK imports based on official documentation
from google.adk.agents import Agent

# --- Configuration ---
# Configure logging to see the agent's operations.
logging.basicConfig(level=logging.INFO)

# Define the full BigQuery table ID for easy reference.
PROJECT_ID = "fsi-banking-agentspace" # Replace with your project ID
DATASET_ID = "equifax_txns"
TABLE_NAME = "transactions"
TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
TEMP_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.temp_categorization_updates"

# Initialize the BigQuery client.
# The client will use the credentials from your gcloud setup.
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    logging.info("Successfully initialized BigQuery client.")
except Exception as e:
    logging.error(f"Failed to initialize BigQuery client: {e}")
    bq_client = None

# --- Constants & Validation ---
VALID_CATEGORIES = {
  "Income": ["Gig Income", "Payroll", "Other Income"],
  "Expense": [
      "Groceries", "Dining & Drinks", "Shopping", "Entertainment",
      "Health & Wellness", "Auto & Transport", "Travel & Vacation",
      "Software & Tech", "Medical", "Insurance", "Bills & Utilities"
    ],
  "Transfer": ["Credit Card Payment", "Internal Transfer"]
}

def is_valid_category(category_l2: str) -> bool:
    """Checks if a category is in the official list."""
    for l1_categories in VALID_CATEGORIES.values():
        if category_l2 in l1_categories:
            return True
    return False

# --- Tool Definitions ---
# Note: The @tool decorator has been removed. Tools are now standard Python functions.
# They return a final result (string or dict) for the agent to process.

def audit_data_quality() -> str:
    """
    Runs a comprehensive audit of the financial data's current state.
    It checks for three types of issues:
    1. Uncategorized Transactions: Counts transactions missing L1 or L2 categories.
    2. Mismatched Transaction Types: Finds illogical pairings (e.g., 'Income' as 'Debit').
    3. Inconsistent Recurring Transactions: Identifies recurring payments with conflicting categories.
    The tool then presents the findings in three summary tables.
    """
    if not bq_client:
        return "Error: BigQuery client is not initialized. Please check credentials and configuration."

    queries = {
        "Uncategorized Transactions": f"""
            SELECT
              CASE
                WHEN category_l1 IS NULL AND category_l2 IS NULL THEN 'Missing L1 & L2'
                WHEN category_l1 IS NULL THEN 'Missing L1 Only'
                WHEN category_l2 IS NULL THEN 'Missing L2 Only'
              END AS issue_type,
              transaction_type, channel, is_recurring, COUNT(transaction_id) AS transaction_count
            FROM `{TABLE_ID}`
            WHERE category_l1 IS NULL OR category_l2 IS NULL
            GROUP BY 1, 2, 3, 4 ORDER BY transaction_count DESC;
        """,
        "Mismatched Transaction Types": f"""
            SELECT transaction_type, category_l1, COUNT(transaction_id) AS transaction_count
            FROM `{TABLE_ID}`
            WHERE (category_l1 = 'Income' AND (transaction_type = 'Debit' OR amount < 0))
               OR (category_l1 = 'Expense' AND (transaction_type = 'Credit' OR amount > 0))
            GROUP BY 1, 2 ORDER BY transaction_count DESC;
        """,
        "Inconsistent Recurring Transactions": f"""
            SELECT
              consumer_name, merchant_name_cleaned,
              COUNT(DISTINCT category_l2) AS distinct_category_count,
              ARRAY_AGG(DISTINCT category_l2 IGNORE NULLS) AS categories_assigned,
              COUNT(t.transaction_id) AS total_inconsistent_transactions
            FROM `{TABLE_ID}` AS t
            WHERE t.is_recurring = TRUE AND t.merchant_name_cleaned IS NOT NULL
            GROUP BY 1, 2
            HAVING COUNT(DISTINCT category_l2) > 1
            ORDER BY total_inconsistent_transactions DESC;
        """
    }

    results_markdown = "Here is the data quality audit summary:\n\n"
    for title, query in queries.items():
        try:
            logging.info(f"Running audit query: {title}")
            df = bq_client.query(query).to_dataframe()
            results_markdown += f"### {title}\n"
            if not df.empty:
                results_markdown += df.to_markdown(index=False) + "\n\n"
            else:
                results_markdown += "✅ No issues found.\n\n"
        except Exception as e:
            return f"An error occurred while running the '{title}' query: {e}"

    return results_markdown

def run_hybrid_categorization() -> Dict[str, Any]:
    """
    Executes a three-phase workflow to cleanse, classify, and categorize financial data.
    - Phase 1: Cleanses text fields and assigns foundational L1 categories.
    - Phase 2: Applies high-confidence rules for fast and accurate L2 categorization.
    - Phase 3: Uses the Gemini AI model for remaining, more ambiguous transactions.
    The tool returns a JSON summary of the operations performed in each phase.
    """
    if not bq_client:
        return {"error": "BigQuery client is not initialized."}

    summary = {}

    # --- Phase 1: Data Cleansing & L1 Categorization ---
    sql_phase1 = f"""
        UPDATE `{TABLE_ID}`
        SET
          transaction_type = CASE
            WHEN transaction_type = 'Credit' AND amount < 0 THEN 'Debit'
            WHEN transaction_type = 'Debit' AND amount > 0 THEN 'Credit'
            ELSE transaction_type
          END,
          description_cleaned = LOWER(REGEXP_REPLACE(description_raw, r'[^a-zA-Z0-9\\s]', ' ')),
          merchant_name_cleaned = LOWER(REGEXP_REPLACE(merchant_name_raw, r'[^a-zA-Z0-9\\s]', ' ')),
          category_l1 = CASE
            WHEN REGEXP_CONTAINS(LOWER(description_raw), r'\\b(credit card payment|internal transfer|transfer to)\\b') THEN 'Transfer'
            WHEN transaction_type = 'Credit' AND amount > 0 THEN 'Income'
            WHEN transaction_type = 'Debit' AND amount < 0 THEN 'Expense'
            ELSE category_l1
          END,
          categorization_update_timestamp = CURRENT_TIMESTAMP()
        WHERE
          description_cleaned IS NULL
          OR merchant_name_cleaned IS NULL
          OR category_l1 IS NULL
          OR (transaction_type = 'Credit' AND amount < 0)
          OR (transaction_type = 'Debit' AND amount > 0);
    """
    try:
        logging.info("Executing Phase 1 SQL...")
        query_job = bq_client.query(sql_phase1)
        query_job.result()
        summary["phase1_count"] = query_job.num_dml_affected_rows or 0
    except Exception as e:
        return {"error": f"An error occurred during Phase 1: {e}"}

    # --- Phase 2: Rules-Based L2 Categorization ---
    sql_phase2 = f"""
        MERGE `{TABLE_ID}` AS T
        USING (
            SELECT
              transaction_id,
              CASE
                WHEN S.category_l1 = 'Transfer' AND REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(payment|xfer|transfer)\\b') THEN 'Credit Card Payment'
                WHEN S.category_l1 = 'Income' THEN CASE WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(uber|lyft|doordash|instacart|upwork|fiverr|taskrabbit|gig)\\b') THEN 'Gig Income' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(payroll|paycheck|direct\\s*deposit|adp)\\b') THEN 'Payroll' ELSE 'Other Income' END
                WHEN S.category_l1 = 'Expense' THEN CASE WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(uber|lyft|chevron|shell|mobil|gas|metro|parking|toll|auto\\s*repair|jiffy\\s*lube)\\b') THEN 'Auto & Transport' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(tmobile|at&t|verizon|comcast|spectrum|pge|internet|phone|utility|sewer|water)\\b') THEN 'Bills & Utilities' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(mcdonalds|starbucks|restaurant|cafe|dominos|deli|dunkin|wendys|taco\\s*bell|coffee|pizza|chipotle|eats)\\b') THEN 'Dining & Drinks' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(amc|theatres|dicks\\s*sporting|netflix|spotify|hulu|ticketmaster|eventbrite|steam\\s*games|disney\\s*plus)\\b') THEN 'Entertainment' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(whole\\s*foods|trader\\s*joes|safeway|kroger|costco|groceries|wal-mart)\\b') THEN 'Groceries' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(fitness|equinox|24\\s*hour|la\\s*fitness|yoga|gym|vitamin|gnc|peloton)\\b') THEN 'Health & Wellness' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(geico|state\\s*farm|progressive|allstate|metlife|insurance)\\b') THEN 'Insurance' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(hospital|doctor|medical|dr\\s*|dentist|pharmacy|cvs|walgreens|clinic)\\b') THEN 'Medical' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(amazon|amzn|target|walmart|best\\s*buy|macys|nike|shopping|home\\s*depot|lowes)\\b') THEN 'Shopping' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(adobe|google|microsoft|zoom|slack|software|aws|tech|openai|godaddy)\\b') THEN 'Software & Tech' WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(merchant_name_cleaned, ''), ' ', IFNULL(description_cleaned, ''))), r'\\b(delta|united|jetblue|southwest|hotel|resort|airport|marriott|hilton|airbnb|expedia)\\b') THEN 'Travel & Vacation' ELSE NULL END
                ELSE NULL
              END AS new_category_l2
            FROM `{TABLE_ID}` S
            WHERE category_l2 IS NULL AND category_l1 IN ('Income', 'Expense', 'Transfer')
        ) AS U
        ON T.transaction_id = U.transaction_id
        WHEN MATCHED AND U.new_category_l2 IS NOT NULL THEN
          UPDATE SET T.category_l2 = U.new_category_l2, T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """
    try:
        logging.info("Executing Phase 2 SQL...")
        query_job = bq_client.query(sql_phase2)
        query_job.result()
        summary["phase2_count"] = query_job.num_dml_affected_rows or 0
    except Exception as e:
        return {"error": f"An error occurred during Phase 2: {e}"}

    # --- Phase 3: Gemini-Powered Categorization ---
    logging.info("Starting Phase 3: Gemini AI Categorization.")
    summary["phase3_count"] = 0
    
    batch_size = 500
    while True:
        fetch_sql = f"""
            SELECT transaction_id, description_cleaned, merchant_name_cleaned
            FROM `{TABLE_ID}` WHERE category_l2 IS NULL LIMIT {batch_size};
        """
        batch_df = bq_client.query(fetch_sql).to_dataframe()

        if batch_df.empty:
            logging.info("No more transactions to categorize with AI. Exiting loop.")
            break

        batch_data_json = batch_df.to_json(orient='records')
        prompt = f"""
            Analyze each transaction in the JSON array below.
            Assign the most appropriate `category_l2` from this list: {json.dumps(VALID_CATEGORIES)}.
            Return a valid JSON array where each object has only `transaction_id` and `category_l2`.
            DATA: {batch_data_json}
        """
        
        try:
            logging.info(f"Sending batch of {len(batch_df)} to Gemini...")
            # NOTE: The llm object is implicitly available to tools in ADK.
            response = llm.send_request(prompt)
            cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
            llm_results = json.loads(cleaned_response)
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Error processing AI response. Skipping batch. Error: {e}")
            continue

        validated_updates_df =_validate_llm_results(llm_results)
        
        if validated_updates_df.empty:
            logging.warning("No valid categorizations returned from AI for this batch.")
            continue
        
        try:
            _update_bq_with_dataframe(validated_updates_df)
            summary["phase3_count"] += len(validated_updates_df)
        except Exception as e:
            logging.error(f"Error updating database with AI results: {e}")
            continue

    # Final Summary Count
    final_count_df = bq_client.query(
        f"SELECT count(transaction_id) as total FROM `{TABLE_ID}` WHERE category_l2 IS NULL"
    ).to_dataframe()
    summary["final_uncategorized"] = int(final_count_df['total'][0])

    return summary

def _validate_llm_results(llm_results: list) -> 'pd.DataFrame':
    """Helper to validate the JSON output from the LLM."""
    import pandas as pd
    validated_updates = []
    for item in llm_results:
        if (isinstance(item, dict) and
            'transaction_id' in item and
            'category_l2' in item and
            is_valid_category(item['category_l2'])):
            validated_updates.append(item)
    return pd.DataFrame(validated_updates)

def _update_bq_with_dataframe(df: 'pd.DataFrame'):
    """Helper to upload a DataFrame to a temp table and run a MERGE."""
    import pandas as pd
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("transaction_id", "STRING"),
            bigquery.SchemaField("category_l2", "STRING"),
        ],
        write_disposition="WRITE_TRUNCATE",
    )
    bq_client.load_table_from_dataframe(df, TEMP_TABLE_ID, job_config=job_config).result()
    
    merge_sql = f"""
        MERGE `{TABLE_ID}` T
        USING `{TEMP_TABLE_ID}` U
        ON T.transaction_id = U.transaction_id
        WHEN MATCHED THEN
            UPDATE SET T.category_l2 = U.category_l2, T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """
    bq_client.query(merge_sql).result()
    logging.info(f"Successfully merged {len(df)} records from temporary table.")


def reset_all_categorizations(confirm: bool = False) -> Dict[str, Any]:
    """
    Resets and clears ALL categorization and data cleansing fields for all transactions.
    This includes category_l1, category_l2, description_cleaned, and merchant_name_cleaned.
    This is a destructive action. Requires user confirmation by passing confirm=True.

    Args:
        confirm: Must be set to True to execute the reset.
    """
    if not bq_client:
        return {"error": "BigQuery client is not initialized."}

    if not confirm:
        # The agent's instructions will guide it to re-prompt the user
        # when it receives this response.
        return {
            "confirmation_required": True,
            "message": "This will clear ALL categorization data. This action cannot be undone. Are you sure you want to proceed?"
        }

    reset_sql = f"""
        UPDATE `{TABLE_ID}`
        SET
          category_l1 = NULL, category_l2 = NULL,
          description_cleaned = NULL, merchant_name_cleaned = NULL,
          categorization_update_timestamp = NULL
        WHERE TRUE;
    """
    try:
        logging.info("Executing reset SQL...")
        query_job = bq_client.query(reset_sql)
        query_job.result()
        return {
            "reset_complete": True,
            "affected_rows": query_job.num_dml_affected_rows or 0
        }
    except Exception as e:
        return {"error": f"An error occurred during the reset operation: {e}"}

# --- Agent Definition ---
# This is the main entry point for your agent, defined AFTER the tools.
# The `instructions` are updated to handle the new return types from the tools.
agent = Agent(
    model="gemini-2.5-flash",
    # Pass the tool functions directly into the tools list.
    tools=[
        audit_data_quality,
        run_hybrid_categorization,
        reset_all_categorizations
    ],
    instructions="""You are a meticulous and proactive Financial Data Analyst agent.
    Your primary function is to analyze, cleanse, and categorize financial transactions using your available tools.
    
    **Your Behavior:**
    - When a tool returns a result, present it clearly to the user.
    - If a tool returns a JSON object with a summary, format it into a user-friendly, readable summary using markdown.
    - If a tool returns a response with `{"confirmation_required": true}`, you MUST ask the user the question from the 'message' field and wait for their confirmation before calling the tool again with `confirm=True`.
    - Always conclude your responses by suggesting the next logical step (e.g., after an audit, suggest running categorization).
    - Present any query results or audit tables in clear, easy-to-read markdown.
    - If a tool returns an error, apologize and present the error message clearly to the user.
    """,
)