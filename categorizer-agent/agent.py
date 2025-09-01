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
from google.adk.agents import Agent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext, AgentTool
from google.genai import types

# Import Runner and SessionService for a complete runnable example
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import asyncio


# --- 1. Configuration & Initialization ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    PROJECT_ID = os.environ["GCP_PROJECT_ID"]
    DATASET_ID = os.environ["BIGQUERY_DATASET"]
    TABLE_NAME = os.environ["BIGQUERY_TABLE"]
    TEMP_TABLE_NAME = os.environ["BIGQUERY_TEMP_TABLE"]
    LOCATION = os.getenv("GCP_LOCATION", "us-central1")

    TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
    TEMP_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TEMP_TABLE_NAME}"

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

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
        "Software & Tech", "Medical", "Insurance", "Bills & Utilities",
        "Fees & Charges", "Business Services"
    ],
    "Transfer": ["Credit Card Payment", "Internal Transfer"]
}
VALID_CATEGORIES_JSON_STR = json.dumps(VALID_CATEGORIES)

def is_valid_category(category_l1: str, category_l2: str) -> bool:
    """Checks if the L2 category is valid for the given L1 category."""
    return category_l1 in VALID_CATEGORIES and category_l2 in VALID_CATEGORIES.get(category_l1, [])

def _validate_llm_results(categorized_json_string: str) -> pd.DataFrame:
    """Parses and validates the JSON output from the LLM for transaction-level updates."""
    validated_updates = []
    try:
        if "```json" in categorized_json_string:
            cleaned_string = categorized_json_string.split("```json")[1].split("```")[0].strip()
        else:
            cleaned_string = categorized_json_string.strip()
        llm_results = json.loads(cleaned_string)
        if not isinstance(llm_results, list): return pd.DataFrame()

        for item in llm_results:
            if not isinstance(item, dict): continue
            transaction_id = item.get('transaction_id')
            category_l1 = item.get('category_l1')
            category_l2 = item.get('category_l2')
            if all([transaction_id, category_l1, category_l2]) and is_valid_category(category_l1, category_l2):
                validated_updates.append({'transaction_id': transaction_id, 'category_l1': category_l1, 'category_l2': category_l2})
            else:
                logger.warning("Skipping invalid category pair: L1='%s', L2='%s'", category_l1, category_l2)
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to decode or parse LLM JSON: {e}\nInput: {categorized_json_string}")
        return pd.DataFrame()
    return pd.DataFrame(validated_updates)

# --- 3. Tool Definitions ---

# --- Report & Reset Tools ---
def audit_data_quality() -> str:
    """
    Runs a comprehensive data quality audit on the main transaction table.
    Generates a user-friendly markdown report summarizing any findings.
    """
    logger.info("Starting data quality audit...")
    queries = {
        "Uncategorized Transactions": {
            "query": f"SELECT CASE WHEN category_l1 IS NULL AND category_l2 IS NULL THEN 'Missing L1 & L2' WHEN category_l1 IS NULL THEN 'Missing L1 Only' WHEN category_l2 IS NULL THEN 'Missing L2 Only' END AS issue_type, transaction_type, channel, COUNT(transaction_id) AS transaction_count FROM `{TABLE_ID}` WHERE category_l1 IS NULL OR category_l2 IS NULL GROUP BY 1, 2, 3 ORDER BY transaction_count DESC;",
            "summary": "Breakdown of transactions missing one or both category labels."
        },
        "Mismatched Transaction Types": {
            "query": f"SELECT transaction_type, category_l1, COUNT(transaction_id) AS transaction_count FROM `{TABLE_ID}` WHERE (category_l1 = 'Income' AND (transaction_type = 'Debit' OR amount < 0)) OR (category_l1 = 'Expense' AND (transaction_type = 'Credit' OR amount > 0)) GROUP BY 1, 2 ORDER BY transaction_count DESC;",
            "summary": "Highlights conflicts where transaction direction (Debit/Credit) contradicts the L1 category (Income/Expense)."
        },
        "Inconsistent Recurring Transactions": {
            "query": f"""
                SELECT
                    merchant_name_cleaned,
                    COUNT(DISTINCT category_l2) AS distinct_category_count,
                    ARRAY_AGG(DISTINCT category_l2 IGNORE NULLS) AS categories_assigned,
                    COUNT(t.transaction_id) AS total_inconsistent_transactions
                FROM `{TABLE_ID}` AS t
                WHERE t.is_recurring = TRUE AND t.merchant_name_cleaned IS NOT NULL
                GROUP BY 1
                HAVING COUNT(DISTINCT category_l2) > 1
                ORDER BY total_inconsistent_transactions DESC;
            """,
            "summary": "Lists recurring transactions from the same merchant that have been assigned multiple, conflicting L2 categories."
        }
    }
    results_markdown = "📊 **Data Quality Audit Report**\n\nHere's a summary of the data quality checks. Any tables below indicate areas that may need attention.\n"
    for title, data in queries.items():
        results_markdown += f"---\n\n### {title}\n*_{data['summary']}_*\n\n"
        try:
            df = bq_client.query(data["query"]).to_dataframe()
            if not df.empty:
                results_markdown += df.to_markdown(index=False) + "\n\n"
            else:
                results_markdown += "✅ **No issues found in this category.**\n\n"
        except GoogleAPICallError as e:
            logger.error(f"BigQuery error during '{title}' audit: {e}")
            results_markdown += f"❌ **Error executing query for `{title}`.** Please check the logs for details.\n\n"
        except Exception as e:
            logger.error(f"Unexpected error during '{title}' audit: {e}")
            results_markdown += f"❌ **An unexpected error occurred while checking `{title}`.** Please check the logs.\n\n"
    results_markdown += "---\n\nAudit complete. Based on these results, you may want to proceed with cleansing and categorization."
    logger.info("Data quality audit finished.")
    return results_markdown


def reset_all_categorizations(tool_context: ToolContext, confirm: bool = False) -> str:
    """
    A destructive tool to reset all categorization and cleansing fields in the table.
    Requires explicit confirmation to proceed.
    """
    if not confirm:
        logger.warning("Reset requested without confirmation. Awaiting user confirmation.")
        return "⚠️ **Confirmation Required**\n\nThis is a destructive action that will clear ALL categorization and cleansing data (categories, cleaned descriptions, etc.). This action **cannot be undone**.\n\nPlease confirm you want to proceed by replying with 'yes' or 'proceed'."
    logger.info("Confirmation received. Proceeding with full data reset.")
    reset_sql = f"UPDATE `{TABLE_ID}` SET category_l1 = NULL, category_l2 = NULL, description_cleaned = NULL, merchant_name_cleaned = NULL, is_recurring = NULL, categorization_method = NULL, categorization_update_timestamp = NULL WHERE TRUE;"
    try:
        query_job = bq_client.query(reset_sql)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully reset %d rows.", affected_rows)
        return f"✅ **Reset Complete**\n\nSuccessfully reset the categorization data for **{affected_rows}** transactions."
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during reset operation: {e}")
        return f"❌ **Error During Reset**\nA BigQuery error occurred during the reset operation. Please check the logs. Error: {e}"


# --- Phase 1: Rules-Based Categorization Tools ---
def run_cleansing_and_rules_categorization() -> str:
    """
    Performs data cleansing and applies a set of deterministic rules to categorize transactions.
    This step handles the most common and easily identifiable transactions.
    """
    logger.info("Starting data cleansing and rules-based categorization...")

    sql_combined_phase = f"""
    MERGE `{TABLE_ID}` AS T
    USING (
        WITH
        -- Step 1: Clean the raw data and create a single searchable text field.
        CleansedData AS (
            SELECT
                transaction_id,
                amount,
                -- Enhanced cleaning: remove special chars, extra spaces, and known prefixes
                TRIM(LOWER(REGEXP_REPLACE(REGEXP_REPLACE(CONCAT(IFNULL(description_raw, ''), ' ', IFNULL(merchant_name_raw, '')), r'[^a-zA-Z0-9\\\\s]', ' '), r'\\\\s+', ' '))) AS search_text,
                -- Correct inconsistent transaction type/amount data.
                CASE
                    WHEN transaction_type = 'Credit' AND amount < 0 THEN 'Debit'
                    WHEN transaction_type = 'Debit' AND amount > 0 THEN 'Credit'
                    ELSE transaction_type
                END AS cleansed_transaction_type
            FROM `{TABLE_ID}`
        ),

        -- Step 2: Assign the most specific Level 2 category based on keyword priority.
        L2_Categorized AS (
            SELECT
                *,
                CASE
                    -- Priority 1: Transfer Keywords
                    WHEN REGEXP_CONTAINS(search_text, r'credit card payment|online pmt|payment received|paymentthank you|mobile payment|autopay|e-payment|card pmt|amex epayment|dda transaction') THEN 'Credit Card Payment'
                    WHEN REGEXP_CONTAINS(search_text, r'internal transfer|transfer received|money in|money out|xfer|transfer|transfer from|transfer to|from chk|to chk|to savings|wealthfront|ally financial') THEN 'Internal Transfer'

                    -- Priority 2: Income Keywords (only apply to credit transactions)
                    WHEN (cleansed_transaction_type = 'Credit' AND amount > 0) AND REGEXP_CONTAINS(search_text, r'uber|lyft|doordash|instacart|upwork|fiverr|taskrabbit|gig|stripe payout|etsy deposit') THEN 'Gig Income'
                    WHEN (cleansed_transaction_type = 'Credit' AND amount > 0) AND REGEXP_CONTAINS(search_text, r'payroll|paycheck|direct\\s*deposit|adp|workday') THEN 'Payroll'
                    WHEN (cleansed_transaction_type = 'Credit' AND amount > 0) AND REGEXP_CONTAINS(search_text, r'interest|rewards|cash back|refund|reimbursement|dividend|invbanktran int') THEN 'Other Income'

                    -- Priority 3: Expense Keywords (only apply to debit transactions)
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'atm fee|overdraft|interest|service charge|late fee|foreign transaction|cash withdrawal') THEN 'Fees & Charges'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'uber|lyft|chevron|shell|mobil|gas|metro|parking|toll|auto\\s*repair|jiffy\\s*lube|autozone|pep boys|lime|bird|amtrak|bart|mta') THEN 'Auto & Transport'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'tmobile|at&t|verizon|comcast|spectrum|pge|internet|phone|utility|sewer|water|con edison|mint mobile|billpay|rent|mortgage') THEN 'Bills & Utilities'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'mcdonalds|starbucks|pizza|pizzeria|restaurant|cafe|dominos|deli|dunkin|wendys|taco\\s*bell|coffee|pizza|chipotle|eats|grubhub|seamless|postmates|caviar|doordash') THEN 'Dining & Drinks'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'amc|theatres|dicks\\s*sporting|netflix|spotify|hulu|ticketmaster|eventbrite|steam\\s*games|disney\\s*plus|peacock|max|playstation|xbox|nintendo|seatgeek') THEN 'Entertainment'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'whole\\s*foods|trader\\s*joes|safeway|kroger|costco|groceries|king kullen|kingkullen|stop and shop|instacart') THEN 'Groceries'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'fitness|equinox|24\\s*hour|la\\s*fitness|yoga|gym|vitamin|gnc|peloton') THEN 'Health & Wellness'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'geico|state\\s*farm|progressive|allstate|metlife|insurance|lemonade') THEN 'Insurance'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'hospital|doctor|medical|dr\\s*|dentist|pharmacy|cvs|walgreens|clinic') THEN 'Medical'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'amazon|amzn|target|walmart|best\\s*buy|macys|nike|shopping|home\\s*depot|lowes|etsy|ebay|nordstrom|kohls|zappos') THEN 'Shopping'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'adobe|google|microsoft|zoom|slack|software|aws|tech|openai|godaddy|[apple.com/bill](https://apple.com/bill)|icloud|notion|figma') THEN 'Software & Tech'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'delta|united|jetblue|southwest|hotel|resort|airport|marriott|hilton|airbnb|expedia|booking.com|vrbo|american|spirit|frontier') THEN 'Travel & Vacation'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'legalzoom|docusign|hellosign|upcounsel|clerky') THEN 'Business Services'
                    ELSE NULL
                END AS new_category_l2
            FROM CleansedData
        ),
        -- Step 3: Derive Level 1 category from the Level 2 category assigned.
        Final_Categorization AS (
            SELECT
                *,
                CASE
                    WHEN new_category_l2 IN ('Credit Card Payment', 'Internal Transfer') THEN 'Transfer'
                    WHEN new_category_l2 IN ('Gig Income', 'Payroll', 'Other Income') THEN 'Income'
                    WHEN new_category_l2 IS NOT NULL THEN 'Expense'
                    ELSE NULL
                END as new_category_l1
            FROM L2_Categorized
        )
        SELECT
            transaction_id,
            cleansed_transaction_type,
            new_category_l1,
            new_category_l2
        FROM Final_Categorization
        WHERE new_category_l2 IS NOT NULL
    ) AS U ON T.transaction_id = U.transaction_id
    WHEN MATCHED THEN
        UPDATE SET
            T.transaction_type = U.cleansed_transaction_type,
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
        return f"⚙️ **Cleansing & Rules Engine Complete**\n\nI've successfully processed the data. A total of **{affected_rows}** rows were cleansed and categorized based on predefined rules. We can now proceed with more advanced AI-based categorization."
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during cleansing and rules phase: {e}")
        return f"❌ **Error During Cleansing**\nA BigQuery error occurred. Please check the application logs for details. Error: {e}"
    except Exception as e:
        logger.error(f"❌ Unexpected error during cleansing and rules phase: {e}")
        return f"❌ **Unexpected Error**\nAn unexpected error occurred during the cleansing step. Please check the application logs. Error: {e}"

def identify_and_flag_recurring_transactions() -> str:
    """
    Identifies and flags recurring transactions based on consistent merchant, amount,
    and frequency. This helps improve categorization accuracy and consistency.
    """
    logger.info("Starting recurring transaction identification...")
    recurring_sql = f"""
    MERGE `{TABLE_ID}` T
    USING (
        WITH TransactionSequences AS (
            -- Step 1: For each merchant, order transactions by date and calculate the time difference between them.
            SELECT
                transaction_id,
                merchant_name_cleaned,
                transaction_date,
                amount,
                -- Calculate days since the last transaction from the same merchant
                DATE_DIFF(transaction_date, LAG(transaction_date, 1) OVER (PARTITION BY merchant_name_cleaned ORDER BY transaction_date), DAY) as days_since_last_txn,
                -- Calculate the percentage change in amount from the last transaction
                SAFE_DIVIDE(
                    amount - LAG(amount, 1) OVER (PARTITION BY merchant_name_cleaned ORDER BY transaction_date),
                    LAG(amount, 1) OVER (PARTITION BY merchant_name_cleaned ORDER BY transaction_date)
                ) as amount_change_pct
            FROM `{TABLE_ID}`
            WHERE merchant_name_cleaned IS NOT NULL AND transaction_type = 'Debit'
        ),
        RecurringCandidates AS (
            -- Step 2: Identify transactions that occur at a regular interval (e.g., monthly) and have a stable amount.
            SELECT
                transaction_id,
                merchant_name_cleaned
            FROM TransactionSequences
            WHERE
                -- Is the time interval consistent? (e.g., 28-32 days for monthly, 6-8 for weekly)
                (days_since_last_txn BETWEEN 28 AND 32 OR days_since_last_txn BETWEEN 6 AND 8)
                -- Is the amount stable? (e.g., less than 15% change)
                AND (amount_change_pct IS NULL OR ABS(amount_change_pct) < 0.15)
        ),
        MerchantsToFlag AS (
            -- Step 3: Only consider merchants that have at least 3 recurring transactions to establish a pattern.
            SELECT merchant_name_cleaned
            FROM RecurringCandidates
            GROUP BY 1
            HAVING COUNT(transaction_id) >= 3
        )
        -- Step 4: Select all transaction_ids for the identified recurring merchants
        SELECT t.transaction_id
        FROM `{TABLE_ID}` t
        JOIN MerchantsToFlag m ON t.merchant_name_cleaned = m.merchant_name_cleaned
    ) AS U ON T.transaction_id = U.transaction_id
    WHEN MATCHED AND T.is_recurring IS NOT TRUE THEN
        UPDATE SET T.is_recurring = TRUE;
    """
    try:
        query_job = bq_client.query(recurring_sql)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows or 0
        logger.info("✅ Recurring transaction identification complete. %d rows flagged.", affected_rows)
        return (f"🔍 **Recurring Transactions Identified**\n\nI've analyzed the data and flagged "
                f"**{affected_rows}** transactions as likely recurring payments. This will help improve the consistency of our next steps.")
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during recurring transaction identification: {e}")
        return f"❌ **Error Identifying Recurring**\nA BigQuery error occurred. Please check the logs. Error: {e}"
    except Exception as e:
        logger.error(f"❌ Unexpected error during recurring transaction identification: {e}")
        return f"❌ **Unexpected Error**\nAn unexpected error occurred. Please check the logs. Error: {e}"


def run_recurring_transaction_harmonization() -> str:
    """
    Applies trusted categories from categorized recurring transactions to uncategorized
    recurring transactions from the same merchant. This is a high-confidence,
    rules-based step to improve consistency.
    """
    logger.info("Starting recurring transaction harmonization...")
    harmonization_sql = f"""
    MERGE `{TABLE_ID}` T
    USING (
        WITH
        -- Step 1: Find the most common, trusted category for each recurring merchant.
        TrustedRecurringCategories AS (
            SELECT
                merchant_name_cleaned,
                transaction_type,
                ARRAY_AGG(
                    STRUCT(category_l1, category_l2)
                    ORDER BY category_count DESC LIMIT 1
                )[OFFSET(0)] AS most_common_category
            FROM (
                SELECT
                    merchant_name_cleaned,
                    transaction_type,
                    category_l1,
                    category_l2,
                    COUNT(*) AS category_count
                FROM `{TABLE_ID}`
                WHERE is_recurring = TRUE
                    AND merchant_name_cleaned IS NOT NULL
                    AND category_l1 IS NOT NULL
                    AND category_l2 IS NOT NULL
                GROUP BY 1, 2, 3, 4
            )
            GROUP BY 1, 2
        )
        -- Step 2: Select all uncategorized recurring transactions that have a trusted category.
        SELECT
            t.transaction_id,
            trc.most_common_category.category_l1 AS new_category_l1,
            trc.most_common_category.category_l2 AS new_category_l2
        FROM `{TABLE_ID}` t
        JOIN TrustedRecurringCategories trc
            ON t.merchant_name_cleaned = trc.merchant_name_cleaned
            AND t.transaction_type = trc.transaction_type
        WHERE
            t.is_recurring = TRUE
            AND (t.category_l1 IS NULL OR t.category_l2 IS NULL)
    ) AS U ON T.transaction_id = U.transaction_id
    WHEN MATCHED THEN
        UPDATE SET
            T.category_l1 = U.new_category_l1,
            T.category_l2 = U.new_category_l2,
            T.categorization_method = 'rules-based-recurring-harmonization',
            T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """
    try:
        query_job = bq_client.query(harmonization_sql)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows or 0
        logger.info("✅ Recurring transaction harmonization complete. %d rows affected.", affected_rows)
        if affected_rows > 0:
            return f"🔄 **Harmonization Complete**\n\nI've successfully standardized the categories for **{affected_rows}** recurring transactions, ensuring consistency with previously categorized entries. This improves overall data quality."
        else:
            return "✅ **Harmonization Complete**\n\nNo recurring transactions needed harmonization at this time. All recurring entries are consistent."

    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during harmonization phase: {e}")
        return f"❌ **Error During Harmonization**\nA BigQuery error occurred. Please check the logs. Error: {e}"
    except Exception as e:
        logger.error(f"❌ Unexpected error during harmonization phase: {e}")
        return f"❌ **Unexpected Error**\nAn unexpected error occurred during the harmonization step. Please check the logs. Error: {e}"


# --- Phase 2: Bulk Merchant AI Tools ---
def get_next_merchant_to_categorize(tool_context: ToolContext) -> str:
    """
    Fetches the single most frequent uncategorized merchant, including a sample of
    transactions to provide rich context for categorization.
    """
    logger.info("Fetching next merchant group for bulk categorization...")
    query = f"""
        WITH TopMerchantGroup AS (
            -- Step 1: Find the top merchant group by transaction count
            SELECT
                merchant_name_cleaned,
                transaction_type
            FROM `{TABLE_ID}`
            WHERE (category_l1 IS NULL OR category_l2 IS NULL)
                AND merchant_name_cleaned IS NOT NULL AND merchant_name_cleaned != ''
                AND category_l1 IS DISTINCT FROM 'Transfer'
            GROUP BY 1, 2
            HAVING COUNT(transaction_id) >= 2
            ORDER BY COUNT(transaction_id) DESC
            LIMIT 1
        )
        -- Step 2: Get the stats and transaction examples for that specific group
        SELECT
            t.merchant_name_cleaned,
            t.transaction_type,
            (SELECT COUNT(*) FROM `{TABLE_ID}`
                WHERE merchant_name_cleaned = t.merchant_name_cleaned
                AND transaction_type = t.transaction_type
                AND (category_l1 IS NULL OR category_l2 IS NULL)) as uncategorized_count,
            ARRAY_AGG(
                STRUCT(
                    t.description_cleaned,
                    t.amount,
                    t.channel,
                    t.is_recurring
                ) ORDER BY t.transaction_date DESC LIMIT 5
            ) AS example_transactions
        FROM `{TABLE_ID}` t
        JOIN TopMerchantGroup g
            ON t.merchant_name_cleaned = g.merchant_name_cleaned
            AND t.transaction_type = g.transaction_type
        WHERE (t.category_l1 IS NULL OR t.category_l2 IS NULL)
        GROUP BY 1, 2;
    """
    try:
        df = bq_client.query(query).to_dataframe()
        if df.empty:
            logger.info("No more merchant groups to process. Escalating.")
            tool_context.actions.escalate = True # Signal to the LoopAgent to stop
            return json.dumps({"status": "complete", "message": "No more merchants to process."})

        result_json = df.to_json(orient='records')
        logger.info(f"Found next merchant: {result_json}")
        return result_json

    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error fetching next merchant: {e}")
        tool_context.actions.escalate = True
        return json.dumps({"status": "error", "message": str(e)})


def apply_bulk_update_by_merchant(merchant_name: str, transaction_type: str, category_l1: str, category_l2: str) -> str:
    """Applies categories to all uncategorized transactions for a given merchant."""
    logger.info(f"Applying bulk update for merchant '{merchant_name}' ({transaction_type}) to '{category_l1} > {category_l2}'.")

    if not is_valid_category(category_l1, category_l2):
        logger.warning(f"Invalid category pair L1:'{category_l1}', L2:'{category_l2}' provided for bulk update. Aborting.")
        return json.dumps({"status": "error", "message": f"Invalid category pair: {category_l1} > {category_l2}"})

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("merchant_name", "STRING", merchant_name),
            bigquery.ScalarQueryParameter("transaction_type", "STRING", transaction_type),
            bigquery.ScalarQueryParameter("category_l1", "STRING", category_l1),
            bigquery.ScalarQueryParameter("category_l2", "STRING", category_l2),
        ]
    )
    query = f"""
        MERGE `{TABLE_ID}` T
        USING (SELECT @merchant_name AS merchant_name, @transaction_type AS transaction_type) AS S
        ON T.merchant_name_cleaned = S.merchant_name AND T.transaction_type = S.transaction_type
        WHEN MATCHED AND (T.category_l1 IS NULL OR T.category_l2 IS NULL) THEN
            UPDATE SET
                T.category_l1 = @category_l1,
                T.category_l2 = @category_l2,
                T.categorization_method = 'llm-bulk-merchant-based',
                T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """
    try:
        merge_job = bq_client.query(query, job_config=job_config)
        merge_job.result()
        updated_count = merge_job.num_dml_affected_rows or 0
        logger.info(f"✅ Successfully bulk-updated {updated_count} records for merchant '{merchant_name}'.")
        return json.dumps({"status": "success", "updated_count": updated_count, "merchant": merchant_name})
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during bulk update for merchant '{merchant_name}': {e}")
        return json.dumps({"status": "error", "message": str(e)})


# --- Phase 3: Bulk Pattern AI Tools ---
def get_next_pattern_to_categorize(tool_context: ToolContext) -> str:
    """Fetches the most frequent uncategorized transaction pattern, including transaction samples for context."""
    logger.info("Fetching next pattern group for bulk categorization...")
    query = f"""
        WITH PatternGroups AS (
            SELECT
                SUBSTR(description_cleaned, 1, 40) AS description_prefix,
                transaction_type,
                channel,
                COUNT(transaction_id) as uncategorized_count,
                ARRAY_AGG(
                    STRUCT(
                        description_cleaned,
                        merchant_name_cleaned,
                        amount,
                        is_recurring
                    ) LIMIT 5 -- Limit to 5 examples per group
                ) AS example_transactions
            FROM `{TABLE_ID}`
            WHERE (category_l1 IS NULL OR category_l2 IS NULL)
                AND description_cleaned IS NOT NULL AND description_cleaned != ''
                AND category_l1 IS DISTINCT FROM 'Transfer'
            GROUP BY 1, 2, 3
            HAVING COUNT(transaction_id) >= 2
        )
        SELECT *
        FROM PatternGroups
        ORDER BY uncategorized_count DESC
        LIMIT 1;
    """
    try:
        df = bq_client.query(query).to_dataframe()
        if df.empty:
            logger.info("No more pattern groups to process. Escalating.")
            tool_context.actions.escalate = True
            return json.dumps({"status": "complete", "message": "No more patterns to process."})

        result_json = df.to_json(orient='records')
        logger.info(f"Found next pattern: {result_json}")
        return result_json

    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error fetching next pattern: {e}")
        tool_context.actions.escalate = True
        return json.dumps({"status": "error", "message": str(e)})


def apply_bulk_update_by_pattern(description_prefix: str, transaction_type: str, channel: str, category_l1: str, category_l2: str) -> str:
    """Applies categories to all uncategorized transactions matching a given pattern."""
    logger.info(f"Applying bulk update for pattern '{description_prefix}' | '{transaction_type}' | '{channel}' to '{category_l1} > {category_l2}'.")

    if not is_valid_category(category_l1, category_l2):
        logger.warning(f"Invalid category pair L1:'{category_l1}', L2:'{category_l2}' provided for pattern bulk update. Aborting.")
        return json.dumps({"status": "error", "message": f"Invalid category pair: {category_l1} > {category_l2}"})

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("description_prefix", "STRING", description_prefix),
            bigquery.ScalarQueryParameter("transaction_type", "STRING", transaction_type),
            bigquery.ScalarQueryParameter("channel", "STRING", channel),
            bigquery.ScalarQueryParameter("category_l1", "STRING", category_l1),
            bigquery.ScalarQueryParameter("category_l2", "STRING", category_l2),
        ]
    )
    query = f"""
        MERGE `{TABLE_ID}` T
        USING (
            SELECT
                @description_prefix as description_prefix,
                @transaction_type as transaction_type,
                @channel as channel
        ) S ON SUBSTR(T.description_cleaned, 1, 40) = S.description_prefix
            AND T.transaction_type = S.transaction_type
            AND T.channel = S.channel
        WHEN MATCHED AND (T.category_l1 IS NULL OR T.category_l2 IS NULL) THEN
            UPDATE SET
                T.category_l1 = @category_l1,
                T.category_l2 = @category_l2,
                T.categorization_method = 'llm-bulk-pattern-based',
                T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """
    try:
        merge_job = bq_client.query(query, job_config=job_config)
        merge_job.result()
        updated_count = merge_job.num_dml_affected_rows or 0
        logger.info(f"✅ Successfully bulk-updated {updated_count} records for pattern '{description_prefix}'.")
        return json.dumps({"status": "success", "updated_count": updated_count, "pattern": description_prefix})
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during pattern bulk update for pattern '{description_prefix}': {e}")
        return json.dumps({"status": "error", "message": str(e)})


# --- Phase 4: Transaction-Level AI Tools ---
def fetch_batch_for_ai_categorization(tool_context: ToolContext, batch_size: int = 500) -> str:
    """Fetches a batch of individual uncategorized transactions for detailed, row-by-row AI processing."""
    logger.info(f"Fetching batch for AI categorization with enriched context. Batch size: {batch_size}")
    fetch_sql = f"""
        SELECT
            transaction_id, description_cleaned, merchant_name_cleaned, amount,
            transaction_type, channel, account_type, institution_name
        FROM `{TABLE_ID}`
        WHERE (category_l1 IS NULL OR category_l2 IS NULL)
            AND category_l1 IS DISTINCT FROM 'Transfer'
        LIMIT {batch_size};
    """
    try:
        df = bq_client.query(fetch_sql).to_dataframe()
        if df.empty:
            logger.info("✅ No more transactions found needing categorization. Signaling loop to stop.")
            tool_context.actions.escalate = True
            return json.dumps({"status": "complete", "message": "No more transactions to process."})
        logger.info(f"Fetched {len(df)} transactions for AI categorization.")
        return df.to_json(orient='records')
    except GoogleAPICallError as e:
        logger.error(f"❌ Failed to fetch batch from BigQuery: {e}")
        tool_context.actions.escalate = True
        return json.dumps({"status": "error", "message": f"Failed to fetch data: {e}"})


def update_categorizations_in_bigquery(categorized_json_string: str) -> str:
    """Updates the main BigQuery table with the L1 and L2 categories provided by the AI agent."""
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
                bigquery.SchemaField("category_l1", "STRING"),
                bigquery.SchemaField("category_l2", "STRING")
            ],
            write_disposition="WRITE_TRUNCATE",
        )
        bq_client.load_table_from_dataframe(validated_df, TEMP_TABLE_ID, job_config=job_config).result()
        logger.info(f"Successfully loaded {len(validated_df)} records into temporary table.")

        merge_sql = f"""
            MERGE `{TABLE_ID}` T
            USING `{TEMP_TABLE_ID}` U ON T.transaction_id = U.transaction_id
            WHEN MATCHED THEN
                UPDATE SET
                    T.category_l1 = U.category_l1,
                    T.category_l2 = U.category_l2,
                    T.categorization_method = 'llm-transaction-based',
                    T.categorization_update_timestamp = CURRENT_TIMESTAMP();
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


# --- 4. Agent Definitions ---

# --- Agent Callbacks ---
async def report_progress_callback(callback_context: CallbackContext) -> types.Content | None:
    """After-agent callback to provide progress updates during loops."""
    logger.info("Callback 'report_progress_callback' triggered.")
    try:
        session = callback_context.session
        if not session or not session.events: return None
        last_event = session.events[-1]
        if (last_event.author == "user" and last_event.content and last_event.content.parts and
            hasattr(last_event.content.parts[0], 'function_response')):

            response = last_event.content.parts[0].function_response
            tool_name = response.name
            # Safely parse the response content which might be a JSON string
            response_data = {}
            if isinstance(response.response, str):
                try:
                    response_data = json.loads(response.response)
                except json.JSONDecodeError:
                    pass # Not a json string, proceed with empty dict
            elif isinstance(response.response, dict):
                response_data = response.response

            updated_count = response_data.get("updated_count", 0)

            update_message = None
            if tool_name == "apply_bulk_update_by_merchant" and updated_count > 0:
                merchant = response_data.get("merchant", "an unknown merchant")
                update_message = f"🛒 Processed **{updated_count}** transactions for **{merchant}**. Looking for the next group..."
            elif tool_name == "apply_bulk_update_by_pattern" and updated_count > 0:
                pattern = response_data.get("pattern", "an unknown pattern")
                update_message = f"🧾 Processed **{updated_count}** transactions for pattern starting with **'{pattern}'**. Looking for the next group..."
            elif tool_name == "update_categorizations_in_bigquery" and updated_count > 0:
                update_message = f"✅ **Batch Processed:** The AI just categorized **{updated_count}** transactions. Fetching the next batch..."

            if update_message:
                logger.info("Yielding progress update: %s", update_message)
                return types.Content(role="model", parts=[types.Part(text=update_message)])

    except (IndexError, KeyError, AttributeError) as e:
        logger.warning(f"Could not parse tool response for progress update. Error: {e}")
    return None


# --- Bulk Merchant Processing Loop ---
merchant_selector_agent = LlmAgent(
    name="merchant_selector",
    model="gemini-2.5-flash",
    tools=[get_next_merchant_to_categorize],
    instruction="Your only job is to call the `get_next_merchant_to_categorize` tool and output its result.",
    output_key="merchant_to_process" # Save the result to session state
)

merchant_categorizer_agent = LlmAgent(
    name="merchant_categorizer",
    model="gemini-2.5-flash",
    tools=[apply_bulk_update_by_merchant],
    after_agent_callback=report_progress_callback,
    instruction=f"""
    Your only job is to categorize the single merchant group provided in the session state variable `merchant_to_process`.
    1. Analyze the `merchant_name_cleaned`, `transaction_type`, and MOST IMPORTANTLY the `example_transactions` from the input. The examples provide critical context like descriptions, amounts, and whether the transaction is known to be recurring.
    2. Determine the correct `category_l1` and `category_l2` from the valid categories: {VALID_CATEGORIES_JSON_STR}.
    3. Use the `transaction_type` to help determine `category_l1`. 'Credit' maps to 'Income', 'Debit' maps to 'Expense'.
    4. Call the `apply_bulk_update_by_merchant` tool with all required parameters based on your analysis.
    5. If you cannot confidently categorize the merchant based on the examples, do nothing.
    """,
)

merchant_categorization_loop = LoopAgent(
    name="merchant_categorization_loop",
    description="This agent starts an automated, context-aware bulk categorization process. It repeatedly finds the most common uncategorized merchant, analyzes a sample of its transactions for context (like descriptions and amounts), and then applies the determined category to all transactions from that merchant. It continues until no more merchant groups are found.",
    sub_agents=[
        SequentialAgent(
            name="merchant_processing_sequence",
            sub_agents=[merchant_selector_agent, merchant_categorizer_agent]
        )
    ],
    max_iterations=25 # Safety break
)

# --- Bulk Pattern Processing Loop ---
pattern_selector_agent = LlmAgent(
    name="pattern_selector",
    model="gemini-2.5-flash",
    tools=[get_next_pattern_to_categorize],
    instruction="Your only job is to call the `get_next_pattern_to_categorize` tool and output its result.",
    output_key="pattern_to_process"
)

pattern_categorizer_agent = LlmAgent(
    name="pattern_categorizer",
    model="gemini-2.5-flash",
    tools=[apply_bulk_update_by_pattern],
    after_agent_callback=report_progress_callback,
    instruction=f"""
    Your only job is to categorize the single transaction pattern provided in `pattern_to_process`.
    1. Analyze the `description_prefix`, `transaction_type`, `channel`, and MOST IMPORTANTLY the `example_transactions` to understand the context.
    2. Determine the correct `category_l1` and `category_l2` from the valid categories: {VALID_CATEGORIES_JSON_STR}.
    3. Use `transaction_type` to help determine `category_l1`. 'Credit' is 'Income', 'Debit' is 'Expense'.
    4. Call the `apply_bulk_update_by_pattern` tool with all required parameters.
    5. If you cannot confidently categorize the pattern based on the examples, do nothing.
    """,
)

pattern_categorization_loop = LoopAgent(
    name="pattern_categorization_loop",
    description="This agent starts an advanced, context-aware bulk categorization based on common transaction description patterns. It analyzes samples of matching transactions to understand their context (merchant, amount) before applying a category, making it more accurate than the merchant-only tool. This is best used after the merchant-based tool.",
    sub_agents=[
        SequentialAgent(
            name="pattern_processing_sequence",
            sub_agents=[pattern_selector_agent, pattern_categorizer_agent]
        )
    ],
    max_iterations=25 # Safety break
)

# --- Transaction-Level Processing Loop ---
transactional_ai_agent = LlmAgent(
    name="transactional_ai_agent",
    model="gemini-2.5-flash",
    tools=[fetch_batch_for_ai_categorization, update_categorizations_in_bigquery],
    after_agent_callback=report_progress_callback,
    instruction=f"""
    You are a highly efficient, autonomous financial transaction categorization engine.
    Your entire operational process is a two-step cycle: FETCH, then UPDATE.
    **CYCLE INSTRUCTIONS:**
    1.  **FETCH:** Call `fetch_batch_for_ai_categorization`.
    2.  **CATEGORIZE & UPDATE:** The tool will return a JSON array of transactions. You MUST now categorize these transactions.
        - Your response MUST be a tool call to `update_categorizations_in_bigquery`.
        - The `categorized_json_string` argument MUST contain a valid JSON array you generate.
        - Each transaction object in your JSON MUST have three keys: `transaction_id`, `category_l1`, and `category_l2`.
        - **Crucially, you must base your categorization decision on a holistic analysis of ALL available fields (`merchant_name_cleaned`, `description_cleaned`, `amount`, `transaction_type`, `channel`, `account_type`, `institution_name`).**
        - Use the `transaction_type` and `amount` to determine the correct `category_l1` ('Income' or 'Expense').
        - Then, select the most appropriate `category_l2` from the valid list: {VALID_CATEGORIES_JSON_STR}.
        - The value for `categorized_json_string` MUST be ONLY the JSON array string, with NO markdown or commentary.
    The loop terminates automatically when the `fetch` tool finds no more data.
    """
)

automated_transactional_loop = LoopAgent(
    name="automated_transactional_categorizer_loop",
    description="This agent is used for the final step. It starts a fully automated, transaction-by-transaction categorization process for any remaining records. It runs in a loop and provides progress updates.",
    sub_agents=[transactional_ai_agent],
    max_iterations=50 # Safety break
)

# --- Root Orchestrator Agent ---
root_agent = Agent(
    name="transaction_categorizer_orchestrator",
    model="gemini-2.5-flash",
    tools=[
        audit_data_quality,
        run_cleansing_and_rules_categorization,
        identify_and_flag_recurring_transactions, # <-- NEW TOOL
        run_recurring_transaction_harmonization,
        reset_all_categorizations,
        AgentTool(agent=merchant_categorization_loop),
        AgentTool(agent=pattern_categorization_loop),
        AgentTool(agent=automated_transactional_loop),
    ],
    instruction="""
    You are a friendly and professional financial transaction data analyst 🤖. Your purpose is to guide the user through a multi-step transaction categorization process. Be clear, concise, proactive, and use markdown and emojis to make your responses easy to read.

    **Your Standard Workflow:**
    1.  **Greeting & Audit:** Start by greeting the user warmly. Always recommend running an `audit_data_quality` check first to get a baseline of the data's health. Present the full report from the tool.
    2.  **Cleansing & Rules:** After the audit, suggest running `run_cleansing_and_rules_categorization`. When it's done, present the tool's formatted response clearly to the user.
    3.  **Identify Recurring:** Next, recommend running `identify_and_flag_recurring_transactions`. Explain that this tool intelligently finds subscriptions and regular bills to improve consistency.
    4.  **Harmonize Recurring:** After identifying recurring payments, recommend running `run_recurring_transaction_harmonization`. Explain that this is a high-confidence step that makes sure all recurring payments from the same merchant have the same category.
    5.  **Bulk AI (Merchant-Based):** After harmonization, it's time for the first AI step. Offer to start the `merchant_categorization_loop`. Explain that this is an efficient automated process that will categorize transactions one merchant group at a time, providing progress updates as it goes.
    6.  **Bulk AI (Pattern-Based):** After the merchant-based step is complete, propose the next level of bulk categorization using the `pattern_categorization_loop`. Explain that this is a more advanced scan that will automatically find and categorize transactions based on common description patterns.
    7.  **Transactional AI Categorization:** Once all bulk updates are done, offer to begin the final, most detailed step by starting the `automated_transactional_categorizer_loop`.
        -   **Before** starting this loop, you MUST inform the user with this exact message: "Great! I am now starting the final automated categorization agent. 🕵️‍♂️ This may take several minutes depending on the number of transactions. **You will see real-time updates below as each batch is completed.** I will let you know when the process is finished."
        -   When the loop finishes and control returns to you, you must inform the user that the process is complete with a concluding message, for example: "🎉 **All Done!** The automated categorization process has successfully completed. All remaining transactions have now been processed by the AI. You can run another data quality audit to see the results."

    **Special Commands (Handle with Extreme Care):**
    - **Resetting Data:** The `reset_all_categorizations` tool is highly destructive. **Never** suggest it unless the user explicitly asks to "start over," "reset everything," or uses similar direct language.
        -   First, call the tool with `confirm=False`.
        -   Present the exact markdown `message` from the tool's response to the user.
        -   Only if the user confirms with "yes", "proceed", or similar affirmative language, call the tool a second time with `confirm=True`. Present the final confirmation message from the tool.
    """
)