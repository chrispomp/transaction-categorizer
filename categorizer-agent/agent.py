# agent.py

import json
import logging
from typing import Dict, Any

from google.cloud import bigquery
from google.adk.agents import Agent, LlmAgent
from google.adk.tools.agent_tool import AgentTool

# --- Configuration ---
logging.basicConfig(level=logging.INFO)

PROJECT_ID = "fsi-banking-agentspace"
DATASET_ID = "equifax_txns"
TABLE_NAME = "transactions"
TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
TEMP_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.temp_categorization_updates"

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

# --- Helper Functions (Internal) ---
def _validate_llm_results(llm_results: list) -> 'pd.DataFrame':
    """Helper to validate the JSON output from the LLM."""
    import pandas as pd
    validated_updates = []
    if not isinstance(llm_results, list):
        logging.warning("LLM results are not a list, cannot validate.")
        return pd.DataFrame()
    for item in llm_results:
        if (isinstance(item, dict) and
            'transaction_id' in item and
            'category_l2' in item and
            is_valid_category(item['category_l2'])):
            validated_updates.append(item)
    return pd.DataFrame(validated_updates)

def _update_bq_with_dataframe(df: 'pd.DataFrame'):
    """Helper to upload a DataFrame to a temp table and run a MERGE."""
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


# --- Tool Definitions ---

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

def run_phase1_cleansing() -> Dict[str, Any]:
    """Phase 1: Cleanses text fields and assigns foundational L1 categories."""
    if not bq_client:
        return {"error": "BigQuery client is not initialized."}
    
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
        return {"phase1_cleansed_rows": query_job.num_dml_affected_rows or 0}
    except Exception as e:
        return {"error": f"An error occurred during Phase 1: {e}"}

def run_phase2_rules() -> Dict[str, Any]:
    """Phase 2: Applies high-confidence rules for fast and accurate L2 categorization."""
    if not bq_client:
        return {"error": "BigQuery client is not initialized."}

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
        return {"phase2_rules_categorized_rows": query_job.num_dml_affected_rows or 0}
    except Exception as e:
        return {"error": f"An error occurred during Phase 2: {e}"}

def get_uncategorized_transactions(batch_size: int = 500) -> str:
    """Retrieves a batch of transactions that still need L2 categorization."""
    if not bq_client:
        return json.dumps({"error": "BigQuery client is not initialized."})
    
    fetch_sql = f"""
        SELECT transaction_id, description_cleaned, merchant_name_cleaned
        FROM `{TABLE_ID}` WHERE category_l2 IS NULL LIMIT {batch_size};
    """
    try:
        logging.info(f"Fetching up to {batch_size} uncategorized transactions.")
        batch_df = bq_client.query(fetch_sql).to_dataframe()
        
        if batch_df.empty:
            logging.info("No more transactions to categorize with AI.")
            return json.dumps({"status": "complete", "count": 0})
            
        return batch_df.to_json(orient='records')
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch transactions: {e}"})

def update_categorizations(categorizations_json: str) -> Dict[str, Any]:
    """Updates the BigQuery table with the L2 categories provided by the AI."""
    if not bq_client:
        return {"error": "BigQuery client is not initialized."}
    
    try:
        llm_results = json.loads(categorizations_json)
        validated_updates_df = _validate_llm_results(llm_results)
        
        if validated_updates_df.empty:
            logging.warning("No valid categorizations were provided to update.")
            return {"status": "no_valid_updates", "updated_count": 0}
        
        _update_bq_with_dataframe(validated_updates_df)
        return {"status": "success", "updated_count": len(validated_updates_df)}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for categorizations."}
    except Exception as e:
        return {"error": f"Failed to update database: {e}"}

def reset_all_categorizations(confirm: bool = False) -> Dict[str, Any]:
    """
    Resets and clears ALL categorization and data cleansing fields for all transactions.
    This includes category_l1, category_l2, description_cleaned, and merchant_name_cleaned.
    This is a destructive action. Requires user confirmation by passing confirm=True.
    """
    if not bq_client:
        return {"error": "BigQuery client is not initialized."}

    if not confirm:
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

# --- Sub-Agent Definitions ---

phase1_agent = LlmAgent(
    name="phase1_cleansing_agent",
    model="gemini-2.5-flash",
    tools=[run_phase1_cleansing],
    instruction="Your task is to run the data cleansing and L1 categorization phase by calling the `run_phase1_cleansing` tool."
)

phase2_agent = LlmAgent(
    name="phase2_rules_agent",
    model="gemini-2.5-flash",
    tools=[run_phase2_rules],
    instruction="Your task is to run the rules-based L2 categorization phase by calling the `run_phase2_rules` tool."
)

phase3_agent = LlmAgent(
    name="phase3_ai_categorization_agent",
    model="gemini-2.5-pro",
    tools=[get_uncategorized_transactions, update_categorizations],
    instruction=f"""
        You are an AI categorization specialist. Your process is as follows:
        1. Use the `get_uncategorized_transactions` tool to fetch a batch of data.
        2. If the tool returns a JSON object with a 'status' of 'complete', your job is done. Announce that there are no more transactions to process.
        3. If you receive a JSON array of transaction data, analyze each transaction within it.
        4. For each transaction, assign the most appropriate `category_l2` from this list: {json.dumps(VALID_CATEGORIES)}.
        5. Format your results as a JSON array string where each object has only `transaction_id` and `category_l2`.
        6. Use the `update_categorizations` tool, passing this JSON string as the `categorizations_json` argument to update the database.
        7. After updating, repeat the process by calling `get_uncategorized_transactions` again to see if more data is available. Continue until the tool returns a 'complete' status.
    """
)


# --- Main Orchestrator Agent ---
# This is the entry point the ADK looks for.
root_agent = Agent(
    name="transaction_categorizer_orchestrator",
    model="gemini-2.5-flash",
    tools=[
        audit_data_quality,
        reset_all_categorizations,
        AgentTool(agent=phase1_agent),
        AgentTool(agent=phase2_agent),
        AgentTool(agent=phase3_agent)
    ],
    instruction="""
    You are the orchestrator for a multi-phase transaction categorization process.
    Your available tools can run an audit, reset data, or execute the categorization phases using specialized agents.
    - Start by suggesting an `audit_data_quality` check to the user to assess the current state of the data.
    - Based on user requests, you can run the categorization phases sequentially by calling the `phase1_cleansing_agent`, `phase2_rules_agent`, and `phase3_ai_categorization_agent` tools.
    - If the user wants to start over, use the `reset_all_categorizations` tool, but always ask for confirmation first by checking the tool's response.
    - Clearly report the summary from each tool call back to the user and suggest the next logical step.
    """
)