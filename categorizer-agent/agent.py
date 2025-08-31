from __future__ import annotations
import os
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv

import pandas as pd
import vertexai
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError

# ADK Core Components
from google.adk.agents import Agent, LlmAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext, AgentTool
from google.genai import types

# Import Runner and SessionService for a complete runnable example
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import asyncio


# --- 1. Configuration & Initialization ---
# Ensure you have a .env file with your project details
# GCP_PROJECT_ID=your-gcp-project-id
# BIGQUERY_DATASET=your_dataset
# BIGQUERY_TABLE=transactions
# BIGQUERY_TEMP_TABLE=transactions_temp_update

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Use GOOGLE_CLOUD_PROJECT as it's a common standard
    PROJECT_ID = os.environ["GCP_PROJECT_ID"]
    DATASET_ID = os.environ["BIGQUERY_DATASET"]
    TABLE_NAME = os.environ["BIGQUERY_TABLE"]
    TEMP_TABLE_NAME = os.environ["BIGQUERY_TEMP_TABLE"]
    LOCATION = os.getenv("GCP_LOCATION", "us-central1")

    TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
    TEMP_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TEMP_TABLE_NAME}"

    # Set GOOGLE_GENAI_USE_VERTEXAI to ensure ADK uses Vertex AI
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    bq_client = bigquery.Client(project=PROJECT_ID)
    logger.info("✅ Successfully initialized Vertex AI and BigQuery clients for project %s.", PROJECT_ID)

except KeyError as e:
    logger.critical(f"❌ Critical Error: Missing required environment variable: {e}")
    raise
except (GoogleAPICallError, Exception) as e:
    logger.critical(f"❌ Critical Error: Failed to initialize Google Cloud services: {e}")
    raise

# --- 2. Constants & Validation ---
VALID_CATEGORIES = {
    "Income": ["Gig Income", "Payroll", "Other Income"],
    "Expense": [
        "Groceries", "Dining & Drinks", "Shopping", "Entertainment",
        "Health & Wellness", "Auto & Transport", "Travel & Vacation",
        "Software & Tech", "Medical", "Insurance", "Bills & Utilities"
    ],
    "Transfer": ["Credit Card Payment", "Internal Transfer"]
}
ALL_VALID_L2_CATEGORIES = {cat for sublist in VALID_CATEGORIES.values() for cat in sublist}
ALL_VALID_L2_CATEGORIES_STR = json.dumps(list(ALL_VALID_L2_CATEGORIES))


def is_valid_category(category_l2: str) -> bool:
    """Checks if a category is in the predefined valid set."""
    return category_l2 in ALL_VALID_L2_CATEGORIES

def _validate_llm_results(categorized_json_string: str) -> pd.DataFrame:
    """
    Parses and validates the JSON string from the LLM, ensuring it adheres
    to the required schema and contains valid categories.
    """
    validated_updates = []
    try:
        # The agent should return clean JSON, but this handles potential markdown wrappers.
        if "```json" in categorized_json_string:
            cleaned_string = categorized_json_string.split("```json")[1].split("```")[0].strip()
        else:
            cleaned_string = categorized_json_string.strip()

        llm_results = json.loads(cleaned_string)

        if not isinstance(llm_results, list):
            logger.warning("LLM output is not a list. Input: %s", categorized_json_string)
            return pd.DataFrame()

        for item in llm_results:
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict item in LLM results: %s", item)
                continue

            transaction_id = item.get('transaction_id')
            category_l2 = item.get('category_l2')

            if not transaction_id or not category_l2:
                logger.warning("Skipping item with missing 'transaction_id' or 'category_l2': %s", item)
                continue

            if is_valid_category(category_l2):
                validated_updates.append({'transaction_id': transaction_id, 'category_l2': category_l2})
            else:
                logger.warning("Skipping invalid category '%s' for transaction_id '%s'", category_l2, transaction_id)

    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to decode or parse LLM JSON: {e}\nInput string: {categorized_json_string}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM result validation: {e}")
        return pd.DataFrame()

    return pd.DataFrame(validated_updates)


# --- 3. Tool Definitions ---

def audit_data_quality() -> str:
    """
    Runs a comprehensive audit on the transaction table to identify common data quality issues
    like missing categories, mismatched transaction types, and inconsistent recurring payments.
    Returns a markdown-formatted report of the findings.
    """
    logger.info("Starting data quality audit...")
    queries = {
        "Uncategorized Transactions": {
            "query": f"SELECT CASE WHEN category_l1 IS NULL AND category_l2 IS NULL THEN 'Missing L1 & L2' WHEN category_l1 IS NULL THEN 'Missing L1 Only' WHEN category_l2 IS NULL THEN 'Missing L2 Only' END AS issue_type, transaction_type, channel, is_recurring, COUNT(transaction_id) AS transaction_count FROM `{TABLE_ID}` WHERE category_l1 IS NULL OR category_l2 IS NULL GROUP BY 1, 2, 3, 4 ORDER BY transaction_count DESC;",
            "summary": "Breakdown of transactions missing category labels."
        },
        "Mismatched Transaction Types": {
            "query": f"SELECT transaction_type, category_l1, COUNT(transaction_id) AS transaction_count FROM `{TABLE_ID}` WHERE (category_l1 = 'Income' AND (transaction_type = 'Debit' OR amount < 0)) OR (category_l1 = 'Expense' AND (transaction_type = 'Credit' OR amount > 0)) GROUP BY 1, 2 ORDER BY transaction_count DESC;",
            "summary": "Highlights conflicts between transaction type and L1 category."
        },
        "Inconsistent Recurring Transactions": {
            "query": f"SELECT consumer_name, merchant_name_cleaned, COUNT(DISTINCT category_l2) AS distinct_category_count, ARRAY_AGG(DISTINCT category_l2 IGNORE NULLS) AS categories_assigned, COUNT(t.transaction_id) AS total_inconsistent_transactions FROM `{TABLE_ID}` AS t WHERE t.is_recurring = TRUE AND t.merchant_name_cleaned IS NOT NULL GROUP BY 1, 2 HAVING COUNT(DISTINCT category_l2) > 1 ORDER BY total_inconsistent_transactions DESC;",
            "summary": "Lists recurring transactions assigned multiple conflicting L2 categories."
        }
    }
    results_markdown = "🔍 **Data Quality Audit Report**\n\n"
    for title, data in queries.items():
        results_markdown += f"---\n\n### {title}\n_{data['summary']}_\n\n"
        try:
            df = bq_client.query(data["query"]).to_dataframe()
            if not df.empty:
                results_markdown += df.to_markdown(index=False) + "\n\n"
            else:
                results_markdown += "✅ **No issues found in this category.**\n\n"
        except GoogleAPICallError as e:
            logger.error(f"BigQuery error during '{title}' audit: {e}")
            results_markdown += f"❌ **Error executing query for this category.** Please check logs.\n\n"
        except Exception as e:
            logger.error(f"Unexpected error during '{title}' audit: {e}")
            results_markdown += f"❌ **An unexpected error occurred.** Please check logs.\n\n"
    results_markdown += "---"
    logger.info("Data quality audit finished.")
    return results_markdown


def run_cleansing_and_rules_categorization() -> str:
    """
    Executes a comprehensive SQL MERGE statement to cleanse raw data and apply a deterministic,
    rules-based engine for initial transaction categorization. This is the first pass for categorization.
    """
    logger.info("Starting data cleansing and rules-based categorization...")
    # This SQL query has been updated to use cleansed fields for keyword matching.
    sql_combined_phase = f"""
    MERGE `{TABLE_ID}` AS T
    USING (
        -- Step 1: Use a CTE to cleanse the data first.
        WITH CleansedData AS (
            SELECT
                transaction_id,
                description_raw,
                amount,
                category_l1,
                category_l2,
                -- Correct transaction type based on amount
                CASE
                    WHEN transaction_type = 'Credit' AND amount < 0 THEN 'Debit'
                    WHEN transaction_type = 'Debit' AND amount > 0 THEN 'Credit'
                    ELSE transaction_type
                END AS cleansed_transaction_type,
                -- Create cleansed description
                LOWER(REGEXP_REPLACE(description_raw, r'[^a-zA-Z0-9\\s]', ' ')) AS cleansed_description,
                -- Create cleansed merchant name
                LOWER(REGEXP_REPLACE(merchant_name_raw, r'[^a-zA-Z0-9\\s]', ' ')) AS cleansed_merchant_name
            FROM `{TABLE_ID}`
            WHERE categorization_method IS NULL OR categorization_method != 'rules-based'
        )
        -- Step 2: Apply categorization rules using the cleansed data from the CTE.
        SELECT
            transaction_id,
            cleansed_transaction_type,
            cleansed_description,
            cleansed_merchant_name,
            -- L1 Category Logic (Improved for clarity)
            COALESCE(
                CASE WHEN REGEXP_CONTAINS(LOWER(description_raw), r'\\b(credit card payment|online payment thank you|mobile payment|card payment|internal transfer|transfer from|transfer to)\\b') THEN 'Transfer' END,
                CASE WHEN cleansed_transaction_type = 'Credit' THEN 'Income' END,
                CASE WHEN cleansed_transaction_type = 'Debit' THEN 'Expense' END,
                category_l1
            ) AS new_category_l1,
            -- L2 Category Logic now uses the cleansed fields
            CASE
                -- Transfer logic still uses raw description for more detail if needed
                WHEN REGEXP_CONTAINS(LOWER(description_raw), r'\\b(credit card payment|online payment thank you|mobile payment|card payment)\\b') THEN 'Credit Card Payment'
                WHEN REGEXP_CONTAINS(LOWER(description_raw), r'\\b(internal transfer|transfer from|transfer to)\\b') THEN 'Internal Transfer'
                ELSE
                    CASE
                        -- THE KEY CHANGE IS HERE: Using cleansed_merchant_name and cleansed_description
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(uber|lyft|doordash|instacart|upwork|fiverr|taskrabbit|gig)\\b') THEN 'Gig Income'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(payroll|paycheck|direct\\s*deposit|adp)\\b') THEN 'Payroll'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(uber|lyft|chevron|shell|mobil|gas|metro|parking|toll|auto\\s*repair|jiffy\\s*lube)\\b') THEN 'Auto & Transport'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(tmobile|at&t|verizon|comcast|spectrum|pge|internet|phone|utility|sewer|water)\\b') THEN 'Bills & Utilities'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(mcdonalds|starbucks|restaurant|cafe|dominos|deli|dunkin|wendys|taco\\s*bell|coffee|pizza|chipotle|eats)\\b') THEN 'Dining & Drinks'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(amc|theatres|dicks\\s*sporting|netflix|spotify|hulu|ticketmaster|eventbrite|steam\\s*games|disney\\s*plus)\\b') THEN 'Entertainment'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(whole\\s*foods|trader\\s*joes|safeway|kroger|costco|groceries|wal-mart)\\b') THEN 'Groceries'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(fitness|equinox|24\\s*hour|la\\s*fitness|yoga|gym|vitamin|gnc|peloton)\\b') THEN 'Health & Wellness'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(geico|state\\s*farm|progressive|allstate|metlife|insurance)\\b') THEN 'Insurance'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(hospital|doctor|medical|dr\\s*|dentist|pharmacy|cvs|walgreens|clinic)\\b') THEN 'Medical'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(amazon|amzn|target|walmart|best\\s*buy|macys|nike|shopping|home\\s*depot|lowes)\\b') THEN 'Shopping'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(adobe|google|microsoft|zoom|slack|software|aws|tech|openai|godaddy)\\b') THEN 'Software & Tech'
                        WHEN REGEXP_CONTAINS(CONCAT(IFNULL(cleansed_merchant_name, ''), ' ', IFNULL(cleansed_description, '')), r'\\b(delta|united|jetblue|southwest|hotel|resort|airport|marriott|hilton|airbnb|expedia)\\b') THEN 'Travel & Vacation'
                        ELSE category_l2
                    END
            END as new_category_l2
        FROM CleansedData
    ) AS U ON T.transaction_id = U.transaction_id
    WHEN MATCHED THEN
        UPDATE SET
            T.transaction_type = U.cleansed_transaction_type,
            T.description_cleaned = U.cleansed_description,
            T.merchant_name_cleaned = U.cleansed_merchant_name,
            T.category_l1 = U.new_category_l1,
            T.category_l2 = U.new_category_l2,
            T.categorization_method = 'rules-based',
            T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """
    try:
        query_job = bq_client.query(sql_combined_phase)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully ran cleansing and rules engine. %d rows affected.", affected_rows)
        return json.dumps({"status": "success", "processed_rows": affected_rows})
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during cleansing and rules phase: {e}")
        return json.dumps({"status": "error", "message": f"A BigQuery error occurred: {e}"})
    except Exception as e:
        logger.error(f"❌ Unexpected error during cleansing and rules phase: {e}")
        return json.dumps({"status": "error", "message": f"An unexpected error occurred: {e}"})

def fetch_batch_for_ai_l2_categorization(tool_context: ToolContext, batch_size: int = 100) -> str:
    """
    Fetches a batch of transactions that have a Level 1 category but are missing a
    Level 2 category. This prepares them for AI-based L2 processing.
    If no such transactions are found, it signals the LoopAgent to stop.
    """
    logger.info(
        "Fetching batch for AI L2 categorization. "
        f"Criteria: category_l2 IS NULL AND category_l1 IN ('Income', 'Expense'). Batch size: {batch_size}"
    )
    fetch_sql = f"""
        SELECT transaction_id, description_cleaned, merchant_name_cleaned
        FROM `{TABLE_ID}`
        WHERE category_l2 IS NULL AND category_l1 IN ('Income', 'Expense')
        LIMIT {batch_size};
    """
    try:
        df = bq_client.query(fetch_sql).to_dataframe()
        if df.empty:
            logger.info("✅ No more transactions found needing L2 categorization. Signaling loop to stop.")
            tool_context.actions.escalate = True # This stops the LoopAgent
            return json.dumps({"status": "complete", "message": "No more transactions to process."})

        logger.info(f"Fetched {len(df)} transactions for AI categorization.")
        return df.to_json(orient='records')

    except GoogleAPICallError as e:
        logger.error(f"❌ Failed to fetch batch from BigQuery: {e}")
        tool_context.actions.escalate = True # Stop the loop on error
        return json.dumps({"status": "error", "message": f"Failed to fetch data: {e}"})

def update_categorizations_in_bigquery(categorized_json_string: str) -> str:
    """
    Receives a JSON string of categorized transactions from the agent, validates it,
    and updates the main BigQuery table using a temporary table and a MERGE statement.
    """
    logger.info("Received AI results. Validating and preparing to update BigQuery.")
    validated_df = _validate_llm_results(categorized_json_string)

    if validated_df.empty:
        logger.warning("No valid categorizations received to update.")
        return json.dumps({"status": "success", "updated_count": 0, "message": "The provided data contained no valid updates."})

    logger.info(f"Validation successful. Found {len(validated_df)} valid records to update.")
    try:
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("transaction_id", "STRING"),
                bigquery.SchemaField("category_l2", "STRING"),
            ],
            write_disposition="WRITE_TRUNCATE",
        )
        bq_client.load_table_from_dataframe(validated_df, TEMP_TABLE_ID, job_config=job_config).result()
        logger.info(f"Successfully loaded {len(validated_df)} records into temporary table.")

        merge_sql = f"""
            MERGE `{TABLE_ID}` T
            USING `{TEMP_TABLE_ID}` U ON T.transaction_id = U.transaction_id
            WHEN MATCHED THEN
                UPDATE SET T.category_l2 = U.category_l2, T.categorization_method = 'llm-based', T.categorization_update_timestamp = CURRENT_TIMESTAMP();
        """
        merge_job = bq_client.query(merge_sql)
        merge_job.result()
        updated_count = merge_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully updated %d records in BigQuery via AI.", updated_count)
        return json.dumps({"status": "success", "updated_count": updated_count})
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during update: {e}")
        return json.dumps({"status": "error", "message": f"Failed to update BigQuery: {e}"})
    except Exception as e:
        logger.error(f"❌ Unexpected error during BigQuery update: {e}")
        return json.dumps({"status": "error", "message": f"An unexpected error occurred during update: {e}"})

def reset_all_categorizations(tool_context: ToolContext, confirm: bool = False) -> str:
    """
    A destructive tool to reset all categorization data in the table. Requires a two-step
    confirmation from the user to prevent accidental data loss.
    """
    if not confirm:
        logger.warning("Reset requested without confirmation. Awaiting user confirmation.")
        return json.dumps({"confirmation_required": True, "message": "This is a destructive action that will clear ALL categorization and cleansing data (categories, cleaned descriptions, etc.). This cannot be undone. Please confirm you want to proceed by replying 'yes'."})

    logger.info("Confirmation received. Proceeding with full data reset.")
    reset_sql = f"UPDATE `{TABLE_ID}` SET category_l1 = NULL, category_l2 = NULL, description_cleaned = NULL, merchant_name_cleaned = NULL, categorization_method = NULL, categorization_update_timestamp = NULL WHERE TRUE;"
    try:
        query_job = bq_client.query(reset_sql)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully reset %d rows.", affected_rows)
        return json.dumps({"status": "success", "reset_complete": True, "affected_rows": affected_rows})
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during reset operation: {e}")
        return json.dumps({"status": "error", "message": f"An error occurred during the reset operation: {e}"})

# --- 4. Agent Definitions ---

# --- REFINED & ROBUST CALLBACK FUNCTION ---
async def report_batch_progress(callback_context: CallbackContext) -> types.Content | None:
    """
    An after_agent_callback that inspects the last event to provide a real-time
    progress update to the user after each batch is processed.
    """
    logger.info("After-agent callback 'report_batch_progress' triggered.")
    try:
        # The invocation_context contains the current session, which holds event history.
        # This is the correct way to access session state within a callback.
        session = callback_context.invocation_context.session
        if not session or not session.events:
            logger.info("Callback triggered but no session or events found to process.")
            return None

        # The result of the last tool call is in the most recent event.
        last_event = session.events[-1]

        # Defensively check that the event is a tool response from our update tool.
        # The `author` of a tool response event is "user".
        if (last_event.author == "user" and
            last_event.content and
            last_event.content.parts and
            hasattr(last_event.content.parts[0], 'function_response') and
            last_event.content.parts[0].function_response.name == "update_categorizations_in_bigquery"):

            # Extract the JSON string from the tool's response field.
            tool_response_dict = last_event.content.parts[0].function_response.response
            
            # The response is already a dict from the tool, no need to json.loads here.
            updated_count = tool_response_dict.get("updated_count", 0)

            if updated_count > 0:
                # This is the user-facing message that will be streamed back.
                update_message = f"✅ **Progress:** AI categorized a batch of {updated_count} transactions. Fetching the next batch..."
                logger.info("Yielding progress update to user: %s", update_message)
                
                # Returning a `Content` object makes the ADK Runner yield it as a new event,
                # sending it directly to the user in real-time.
                return types.Content(role="model", parts=[types.Part(text=update_message)])
            else:
                logger.info("Last tool call updated 0 records. No progress update to send.")

    except (IndexError, KeyError, AttributeError) as e:
        # This is a safe failure. It often means the last event wasn't the one we were
        # looking for, which is expected in some parts of the agent's execution.
        logger.warning(
            "Could not parse tool response for progress update. "
            "This is normal if the last action wasn't an update. Error: %s", e
        )

    # Return None if no update is needed for the user.
    return None
# ----------------------------------------

# This agent performs the reasoning step (categorization) and uses tools for I/O.
ai_batch_processor_agent = LlmAgent(
    name="ai_batch_processor_agent",
    model="gemini-2.5-flash", # NON-NEGOTIABLE REQUIREMENT
    tools=[
        fetch_batch_for_ai_l2_categorization,
        update_categorizations_in_bigquery,
    ],
    # --- ATTACHED CALLBACK ---
    # This callback will now run after every execution of this agent inside the loop.
    after_agent_callback=report_batch_progress,
    # -------------------------
    instruction=f"""
    You are a highly efficient, autonomous financial transaction categorization engine.
    Your entire operational process is a two-step cycle: FETCH, then UPDATE. You must follow this cycle precisely.

    **CYCLE INSTRUCTIONS:**

    1.  **FETCH:** Your first and only initial action is to call the `fetch_batch_for_ai_l2_categorization` tool. This will provide you with a JSON array of transactions that need categorization.

    2.  **CATEGORIZE & UPDATE:** The tool will return a JSON array. You MUST now use your intelligence to categorize these transactions.
        - Your response for this step MUST be a tool call to `update_categorizations_in_bigquery`.
        - The `categorized_json_string` argument for this tool MUST contain a valid JSON array that you generate.
        - **Rule for Generation**: For each transaction object from the FETCH step, you must add a `category_l2` field. The value of this field MUST be one of the following options: {ALL_VALID_L2_CATEGORIES_STR}.
        - **Formatting Rule**: The value you provide for the `categorized_json_string` argument MUST be ONLY the JSON array string, with NO markdown, commentary, or other text.
      
    You will repeat this cycle. The loop will terminate automatically when the `fetch` tool finds no more data.
    """
)

# This loop agent wraps our intelligent agent to run it repeatedly.
automated_categorizer_loop = LoopAgent(
    name="automated_ai_categorizer_loop",
    sub_agents=[ai_batch_processor_agent],
    max_iterations=50 # Safety break to prevent infinite loops
)

# The root agent is the main orchestrator for the user.
root_agent = Agent(
    name="transaction_categorizer_orchestrator",
    model="gemini-2.5-flash", # NON-NEGOTIABLE REQUIREMENT
    tools=[
        audit_data_quality,
        run_cleansing_and_rules_categorization,
        reset_all_categorizations,
        AgentTool(agent=automated_categorizer_loop),
    ],
    instruction="""
    You are a friendly and professional financial data steward. Your purpose is to guide the user through a multi-step transaction categorization process. Be clear, concise, and proactive.

    **Your Standard Workflow:**
    1.  **Greeting & Audit:** Start by greeting the user. Your first recommendation should always be to run the `audit_data_quality` tool to understand the current state of the data. Present the markdown report from the tool directly to the user.
    2.  **Cleansing & Rules:** After the audit, the next logical step is to suggest running the `run_cleansing_and_rules_categorization` tool. This is a fast, deterministic step. When it's done, parse the JSON output and state clearly how many rows were processed. For example: "The data cleansing and initial rule-based categorization is complete, processing X rows."
    3.  **AI Categorization:** Once the rules have run, offer to start the automated AI categorization using the `automated_ai_categorizer_loop` tool.
        - Before calling it, inform the user: "I am now starting the automated AI categorization process. This will process transactions in batches and may take several minutes depending on the data volume. You will see real-time updates as each batch is completed. I will let you know when the entire process is finished."
        - You only need to **call this tool ONCE**. It will loop internally until all remaining transactions are categorized.
        - When the tool finishes, it will return a final status message. Relay this to the user, for example: "The automated AI categorization is now complete."

    **Special Commands (Handle with Care):**
    - **Resetting Data:** The `reset_all_categorizations` tool is destructive. You **MUST NOT** suggest using this tool unless the user explicitly asks to "start over," "reset the data," or uses similar phrasing.
        - If the user asks to reset, you MUST first call the tool with `confirm=False`.
        - You must then present the exact `message` from the tool's JSON response to the user for confirmation.
        - Only if the user explicitly confirms with "yes," "proceed," or a similar direct affirmative, should you call the tool a second time with `confirm=True`.
    """
)

# --- 5. Example of How to Run the Agent ---
# To run this agent, you would typically use the ADK Runner.
async def main():
    """Main function to run an example interaction with the agent."""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="transaction_categorization_app",
        session_service=session_service
    )

    session = await session_service.create_session(
        app_name="transaction_categorization_app",
        user_id="test_user_123"
    )

    # --- Start the conversation ---
    print("--- Starting Conversation with Agent ---")

    # 1. Start with a greeting, which should trigger the audit
    initial_message = "Hi, can you help me categorize my transactions?"
    print(f"\n👤 USER: {initial_message}")

    events = runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=initial_message)])
    )
    async for event in events:
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"🤖 AGENT: {event.content.parts[0].text}")

    # 2. Assume the user agrees to run the AI process
    ai_prompt = "Yes, please start the AI categorization"
    print(f"\n👤 USER: {ai_prompt}")

    events = runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=ai_prompt)])
    )
    async for event in events:
        # This will now print both the main agent responses AND the real-time updates from our callback!
        if event.content and event.content.parts and event.content.parts[0].text:
             print(f"🤖 AGENT: {event.content.parts[0].text}")

    print("\n--- Conversation Ended ---")

if __name__ == "__main__":
    # To run this script directly, execute `python your_script_name.py`
    # Make sure your .env file is configured correctly.
    asyncio.run(main())