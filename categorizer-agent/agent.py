from __future__ import annotations
import os
import json
import logging
from dotenv import load_dotenv

import pandas as pd
import vertexai
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError

# ADK Core Components
from google.adk.agents import Agent, LlmAgent, LoopAgent
from google.adk.tools import ToolContext, AgentTool

# --- 1. Configuration & Initialization ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

def _validate_and_clean_llm_json(json_string: str) -> list:
    """Cleans, parses, and performs basic validation on a JSON string from an LLM."""
    try:
        if "```json" in json_string:
            cleaned_string = json_string.split("```json")[1].split("```")[0].strip()
        else:
            cleaned_string = json_string.strip()
        
        data = json.loads(cleaned_string)
        if isinstance(data, list):
            return data
        else:
            logger.warning("LLM output was valid JSON but not a list as expected.")
            return []
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to decode or parse LLM JSON: {e}\nInput: {json_string}")
        return []

def _validate_transaction_level_results(categorized_json_string: str) -> pd.DataFrame:
    """Parses and validates the JSON output from the LLM for transaction-level updates."""
    llm_results = _validate_and_clean_llm_json(categorized_json_string)
    validated_updates = []
    for item in llm_results:
        if not isinstance(item, dict): continue
        transaction_id = item.get('transaction_id')
        category_l1 = item.get('category_l1')
        category_l2 = item.get('category_l2')
        if all([transaction_id, category_l1, category_l2]) and is_valid_category(category_l1, category_l2):
            validated_updates.append({
                'transaction_id': transaction_id,
                'category_l1': category_l1,
                'category_l2': category_l2
            })
        else:
            logger.warning("Skipping invalid category pair: L1='%s', L2='%s'", category_l1, category_l2)
    return pd.DataFrame(validated_updates)

def _validate_bulk_llm_results(categorized_json_string: str, required_keys: list[str]) -> pd.DataFrame:
    """Parses and validates the JSON output from the LLM for bulk updates (merchant or pattern)."""
    llm_results = _validate_and_clean_llm_json(categorized_json_string)
    validated_updates = []
    for item in llm_results:
        if not isinstance(item, dict): continue
        category_l1 = item.get('category_l1')
        category_l2 = item.get('category_l2')
    
        # Check for required keys and valid categories
        if all(key in item for key in required_keys) and is_valid_category(category_l1, category_l2):
            validated_updates.append(item)
        else:
            logger.warning("Skipping invalid bulk update item: %s", item)
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
    Performs comprehensive data cleansing, including timestamp validation,
    merchant name extraction, and applies deterministic rules to categorize
    transactions.
    """
    logger.info("Starting data cleansing and rules-based categorization...")

    sql_combined_phase = f"""
    MERGE `{TABLE_ID}` AS T
    USING (
        WITH
        CleansedData AS (
            SELECT
                transaction_id,
                amount,
                SAFE_CAST(transaction_date AS TIMESTAMP) AS new_transaction_date,
                TRIM(LOWER(REGEXP_REPLACE(REGEXP_REPLACE(REPLACE(IFNULL(description_raw, ''), '-', ''), r'[^a-zA-Z0-9\\s]', ' '), r'\\s+', ' '))) AS new_description_cleaned,
                TRIM(LOWER(
                    REGEXP_EXTRACT(
                        REGEXP_REPLACE(REGEXP_REPLACE(REPLACE(IFNULL(merchant_name_raw, ''), '-', ''), r'[^a-zA-Z0-9\\s\\*#-]', ''), r'\\s+', ' '),
                        r'^(?:sq\\s*\\*|pypl\\s*\\*|cl\\s*\\*|\\*\\s*)?([^*#-]+)'
                    )
                )) AS new_merchant_name_cleaned,
                CASE
                    WHEN transaction_type = 'Credit' AND amount < 0 THEN 'Debit'
                    WHEN transaction_type = 'Debit' AND amount > 0 THEN 'Credit'
                    ELSE transaction_type
                END AS cleansed_transaction_type
            FROM `{TABLE_ID}`
        ),
        CombinedSearchText AS (
            SELECT
                *,
                CONCAT(IFNULL(new_description_cleaned, ''), ' ', IFNULL(new_merchant_name_cleaned, '')) as search_text
            FROM CleansedData
        ),
        L2_Categorized AS (
            SELECT
                *,
                CASE
                    WHEN REGEXP_CONTAINS(search_text, r'credit card payment|online pmt|payment received|paymentthank you|mobile payment|autopay|e-payment|card pmt|amex epayment|dda transaction') THEN 'Credit Card Payment'
                    WHEN REGEXP_CONTAINS(search_text, r'internal transfer|transfer received|money in|money out|xfer|transfer|transfer from|transfer to|from chk|to chk|to savings|wealthfront|ally financial') THEN 'Internal Transfer'
                    WHEN (cleansed_transaction_type = 'Credit' AND amount > 0) AND REGEXP_CONTAINS(search_text, r'uber|lyft|doordash|instacart|upwork|fiverr|taskrabbit|gig|stripe payout|etsy deposit') THEN 'Gig Income'
                    WHEN (cleansed_transaction_type = 'Credit' AND amount > 0) AND REGEXP_CONTAINS(search_text, r'payroll|paycheck|direct\\s*deposit|adp|workday') THEN 'Payroll'
                    WHEN (cleansed_transaction_type = 'Credit' AND amount > 0) AND REGEXP_CONTAINS(search_text, r'interest|rewards|cash back|refund|reimbursement|dividend|invbanktran int') THEN 'Other Income'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'atm fee|overdraft|interest|service charge|late fee|foreign transaction|cash withdrawal') THEN 'Fees & Charges'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'uber|lyft|chevron|shell|mobil|lirr|gas|metro|parking|toll|auto\\s*repair|jiffy\\s*lube|autozone|pep boys|lime|bird|amtrak|bart|mta') THEN 'Auto & Transport'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'tmobile|at&t|verizon|comcast|spectrum|pge|internet|phone|utility|sewer|water|con edison|mint mobile|billpay|rent|mortgage') THEN 'Bills & Utilities'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'mcdonalds|starbucks|pizza|pizzeria|restaurant|cafe|dominos|deli|dunkin|wendys|taco\\s*bell|coffee|pizza|chipotle|brewery|burger\\s*king|eats|grubhub|seamless|postmates|caviar|doordash') THEN 'Dining & Drinks'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'amc|theatres|dicks\\s*sporting|netflix|spotify|hulu|ticketmaster|eventbrite|steam\\s*games|disney\\s*plus|peacock|max|playstation|xbox|nintendo|seatgeek') THEN 'Entertainment'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'whole\\s*foods|factor75|trader\\s*joes|safeway|kroger|costco|groceries|king kullen|kingkullen|stop and shop|instacart') THEN 'Groceries'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'fitness|equinox|24\\s*hour|la\\s*fitness|yoga|gym|vitamin|gnc|peloton') THEN 'Health & Wellness'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'geico|state\\s*farm|progressive|allstate|metlife|insurance|lemonade') THEN 'Insurance'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'hospital|doctor|medical|dr\\s*|dentist|pharmacy|cvs|walgreens|clinic') THEN 'Medical'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'amazon|amzn|target|walmart|best\\s*buy|marshalls|kohls|macys|nike|shopping|home\\s*depot|lowes|etsy|ebay|nordstrom|kohls|zappos') THEN 'Shopping'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'adobe|google|audible|spotify|netflix|hulu|microsoft|zoom|slack|software|aws|tech|openai|godaddy|[apple.com/bill](https://apple.com/bill)|icloud|notion|figma') THEN 'Software & Tech'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'delta|united|jetblue|southwest|hotel|resort|airport|marriott|hilton|airbnb|expedia|booking.com|vrbo|american|spirit|frontier') THEN 'Travel & Vacation'
                    WHEN (cleansed_transaction_type = 'Debit' AND amount < 0) AND REGEXP_CONTAINS(search_text, r'legalzoom|docusign|hellosign|upcounsel|clerky') THEN 'Business Services'
                    ELSE NULL
                END AS new_category_l2
            FROM CombinedSearchText
        ),
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
            new_transaction_date,
            cleansed_transaction_type,
            new_description_cleaned,
            new_merchant_name_cleaned,
            new_category_l1,
            new_category_l2
        FROM Final_Categorization
    ) AS U ON T.transaction_id = U.transaction_id
    WHEN MATCHED THEN
        UPDATE SET
            T.transaction_date = U.new_transaction_date,
            T.transaction_type = U.cleansed_transaction_type,
            T.description_cleaned = U.new_description_cleaned,
            T.merchant_name_cleaned = U.new_merchant_name_cleaned,
            T.category_l1 = COALESCE(U.new_category_l1, T.category_l1),
            T.category_l2 = COALESCE(U.new_category_l2, T.category_l2),
            T.categorization_method = CASE WHEN U.new_category_l2 IS NOT NULL THEN 'rules-based' ELSE T.categorization_method END,
            T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """

    try:
        query_job = bq_client.query(sql_combined_phase)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully ran cleansing and rules engine. %d rows affected.", affected_rows)
        return f"⚙️ **Cleansing & Rules Engine Complete**\n\nI've successfully processed the data. A total of **{affected_rows}** rows were cleansed and/or categorized based on predefined rules. We can now proceed with the next steps."
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during cleansing and rules phase: {e}")
        return f"❌ **Error During Cleansing**\nA BigQuery error occurred. Please check the application logs for details. Error: {e}"
    except Exception as e:
        logger.error(f"❌ Unexpected error during cleansing and rules phase: {e}")
        return f"❌ **Unexpected Error**\nAn unexpected error occurred during the cleansing step. Please check the application logs. Error: {e}"

def run_recurring_transaction_harmonization() -> str:
    """
    Applies trusted categories from categorized recurring transactions to uncategorized
    recurring transactions from the same merchant.
    """
    logger.info("Starting recurring transaction harmonization...")
    harmonization_sql = f"""
    MERGE `{TABLE_ID}` T
    USING (
        WITH
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

# --- Phase 2 & 3: AI-Based Bulk & Recurring Tools ---

# --- Tools for AI Recurring Identification ---
def get_recurring_candidates_batch(tool_context: ToolContext, batch_size: int = 15) -> str:
    """
    Fetches a batch of merchants that are potential candidates for being recurring.
    Gathers evidence like transaction counts, amount stability, time intervals, and example descriptions.
    This version is optimized to only select candidates with 3+ transactions.
    """
    logger.info(f"Fetching batch of {batch_size} recurring merchant candidates...")
    query = f"""
        WITH TransactionIntervals AS (
            SELECT
                merchant_name_cleaned,
                transaction_type,
                transaction_id,
                transaction_date,
                DATE_DIFF(
                    transaction_date,
                    LAG(transaction_date, 1) OVER (PARTITION BY merchant_name_cleaned, transaction_type ORDER BY transaction_date),
                    DAY
                ) as days_since_last_txn
            FROM `{TABLE_ID}`
            WHERE (is_recurring IS NULL OR is_recurring = FALSE)
                AND merchant_name_cleaned IS NOT NULL AND merchant_name_cleaned != ''
                AND transaction_type = 'Debit'
        ),
        MerchantStats AS (
            SELECT
                t.merchant_name_cleaned,
                t.transaction_type,
                COUNT(t.transaction_id) as transaction_count,
                ARRAY_AGG(ti.days_since_last_txn IGNORE NULLS) AS transaction_intervals_days,
                LOGICAL_OR(REGEXP_CONTAINS(LOWER(IFNULL(t.description_cleaned, '')), r'monthly|weekly|annual|subscription|membership|recurring|plan')) AS has_recurring_keywords,
                STDDEV(ABS(t.amount)) as stddev_amount,
                AVG(ABS(t.amount)) as avg_amount,
                ARRAY_AGG(
                    STRUCT(
                        t.description_cleaned,
                        t.amount,
                        t.transaction_date
                    ) ORDER BY t.transaction_date DESC LIMIT 5
                ) AS example_transactions
            FROM `{TABLE_ID}` t
            JOIN TransactionIntervals ti ON t.transaction_id = ti.transaction_id
            GROUP BY 1, 2
            HAVING COUNT(t.transaction_id) >= 3
        )
        SELECT *
        FROM MerchantStats
        ORDER BY has_recurring_keywords DESC, transaction_count DESC
        LIMIT {batch_size};
    """
    try:
        df = bq_client.query(query).to_dataframe()
        if df.empty:
            logger.info("No more recurring candidates to process. Escalating.")
            tool_context.actions.escalate = True
            return json.dumps({"status": "complete", "message": "No more recurring candidates to process."})
        
        # Clean up data types before converting to JSON
        df['stddev_amount'] = pd.to_numeric(df['stddev_amount'], errors='coerce').fillna(0)
        df['avg_amount'] = pd.to_numeric(df['avg_amount'], errors='coerce').fillna(0)
        
        result_json = df.to_json(orient='records')
        logger.info("Found next recurring candidate batch: %s", result_json)
        return result_json
    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error fetching recurring candidates batch: {e}")
        tool_context.actions.escalate = True
        return json.dumps({"status": "error", "message": str(e)})

def apply_bulk_recurring_flags(categorized_json_string: str) -> str:
    """Applies the 'is_recurring' flag to merchants identified by the LLM."""
    logger.info("Applying bulk recurring flags...")
    
    llm_results = _validate_and_clean_llm_json(categorized_json_string)
    validated_updates = []
    for item in llm_results:
        if not isinstance(item, dict): continue
        if item.get('is_recurring') is True and item.get('merchant_name_cleaned') and item.get('transaction_type'):
            validated_updates.append({
                'merchant_name_cleaned': item.get('merchant_name_cleaned'),
                'transaction_type': item.get('transaction_type')
            })

    if not validated_updates:
        return json.dumps({"status": "success", "updated_count": 0, "message": "No valid recurring merchants were identified to apply."})

    validated_df = pd.DataFrame(validated_updates)

    try:
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("merchant_name_cleaned", "STRING"),
                bigquery.SchemaField("transaction_type", "STRING"),
            ],
            write_disposition="WRITE_TRUNCATE"
        )
        bq_client.load_table_from_dataframe(validated_df, TEMP_TABLE_ID, job_config=job_config).result()
        logger.info(f"Loaded {len(validated_df)} recurring merchant flags into temporary table.")

        merge_sql = f"""
            MERGE `{TABLE_ID}` T
            USING `{TEMP_TABLE_ID}` U
                ON T.merchant_name_cleaned = U.merchant_name_cleaned
                AND T.transaction_type = U.transaction_type
            WHEN MATCHED AND (T.is_recurring IS NULL OR T.is_recurring = FALSE) THEN
                UPDATE SET
                    T.is_recurring = TRUE,
                    T.categorization_method = 'llm-recurring-identification';
        """
        merge_job = bq_client.query(merge_sql)
        merge_job.result()
        updated_count = merge_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully bulk-flagged %d records as recurring.", updated_count)
    
        summary = validated_df[['merchant_name_cleaned']].to_dict(orient='records')
        return json.dumps({"status": "success", "updated_count": updated_count, "summary": summary})

    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during bulk recurring flag update: {e}")
        return json.dumps({"status": "error", "message": str(e)})

# --- Tools for AI Bulk Categorization ---
def get_merchant_batch_to_categorize(tool_context: ToolContext, batch_size: int = 10) -> str:
    """
    Fetches a batch of the most frequent uncategorized merchants for efficient bulk processing.
    """
    logger.info(f"Fetching batch of {batch_size} merchants for bulk categorization...")
    query = f"""
        WITH TopMerchantGroups AS (
            SELECT
                merchant_name_cleaned,
                transaction_type,
                COUNT(transaction_id) as uncategorized_count
            FROM `{TABLE_ID}`
            WHERE (category_l1 IS NULL OR category_l2 IS NULL)
                AND merchant_name_cleaned IS NOT NULL AND merchant_name_cleaned != ''
                AND category_l1 IS DISTINCT FROM 'Transfer'
            GROUP BY 1, 2
            HAVING COUNT(transaction_id) >= 2
            ORDER BY uncategorized_count DESC
            LIMIT {batch_size}
        )
        SELECT
            g.merchant_name_cleaned,
            g.transaction_type,
            g.uncategorized_count,
            ARRAY_AGG(
                STRUCT(
                    t.description_cleaned,
                    t.amount,
                    t.channel,
                    t.is_recurring
                ) ORDER BY t.transaction_date DESC LIMIT 5
            ) AS example_transactions
        FROM `{TABLE_ID}` t
        JOIN TopMerchantGroups g
            ON t.merchant_name_cleaned = g.merchant_name_cleaned
            AND t.transaction_type = g.transaction_type
        WHERE (t.category_l1 IS NULL OR t.category_l2 IS NULL)
        GROUP BY 1, 2, 3;
    """
    try:
        df = bq_client.query(query).to_dataframe()
        if df.empty:
            logger.info("No more merchant groups to process. Escalating.")
            tool_context.actions.escalate = True
            return json.dumps({"status": "complete", "message": "No more merchant groups to process."})
        result_json = df.to_json(orient='records')
        logger.info("Found next merchant batch: %s", result_json)
        return result_json
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error fetching merchant batch: {e}")
        tool_context.actions.escalate = True
        return json.dumps({"status": "error", "message": str(e)})

def apply_bulk_merchant_update(categorized_json_string: str) -> str:
    """Applies categories to a batch of merchants from a validated JSON string."""
    logger.info("Applying bulk merchant update...")
    validated_df = _validate_bulk_llm_results(categorized_json_string, ['merchant_name_cleaned', 'transaction_type'])
    if validated_df.empty:
        return json.dumps({"status": "success", "updated_count": 0, "message": "No valid merchant categorizations to apply."})

    try:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        bq_client.load_table_from_dataframe(validated_df, TEMP_TABLE_ID, job_config=job_config).result()
        logger.info(f"Loaded {len(validated_df)} merchant categorizations into temporary table.")

        merge_sql = f"""
            MERGE `{TABLE_ID}` T
            USING `{TEMP_TABLE_ID}` U
                ON T.merchant_name_cleaned = U.merchant_name_cleaned
                AND T.transaction_type = U.transaction_type
            WHEN MATCHED AND (T.category_l1 IS NULL OR T.category_l2 IS NULL) THEN
                UPDATE SET
                    T.category_l1 = U.category_l1,
                    T.category_l2 = U.category_l2,
                    T.categorization_method = 'llm-bulk-merchant-based',
                    T.categorization_update_timestamp = CURRENT_TIMESTAMP();
        """
        merge_job = bq_client.query(merge_sql)
        merge_job.result()
        updated_count = merge_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully bulk-updated %d records for merchants.", updated_count)
    
        summary = validated_df[['merchant_name_cleaned', 'category_l1', 'category_l2']].to_dict(orient='records')
        return json.dumps({"status": "success", "updated_count": updated_count, "summary": summary})

    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during bulk merchant update: {e}")
        return json.dumps({"status": "error", "message": str(e)})

def get_pattern_batch_to_categorize(tool_context: ToolContext, batch_size: int = 10) -> str:
    """Fetches a batch of the most frequent uncategorized transaction patterns for efficient bulk processing."""
    logger.info(f"Fetching batch of {batch_size} patterns for bulk categorization...")
    query = f"""
        WITH PatternGroups AS (
            SELECT
                SUBSTR(description_cleaned, 1, 40) AS description_prefix,
                transaction_type,
                channel,
                COUNT(transaction_id) as uncategorized_count
            FROM `{TABLE_ID}`
            WHERE (category_l1 IS NULL OR category_l2 IS NULL)
                AND description_cleaned IS NOT NULL AND description_cleaned != ''
                AND category_l1 IS DISTINCT FROM 'Transfer'
            GROUP BY 1, 2, 3
            HAVING COUNT(transaction_id) >= 2
            ORDER BY uncategorized_count DESC
            LIMIT {batch_size}
        )
        SELECT
            g.description_prefix,
            g.transaction_type,
            g.channel,
            g.uncategorized_count,
            ARRAY_AGG(
                STRUCT(
                    t.description_cleaned,
                    t.merchant_name_cleaned,
                    t.amount,
                    t.is_recurring
                ) LIMIT 5
            ) AS example_transactions
        FROM `{TABLE_ID}` t
        JOIN PatternGroups g
            ON SUBSTR(t.description_cleaned, 1, 40) = g.description_prefix
            AND t.transaction_type = g.transaction_type
            AND t.channel = g.channel
        WHERE (t.category_l1 IS NULL OR t.category_l2 IS NULL)
        GROUP BY 1, 2, 3, 4;
    """
    try:
        df = bq_client.query(query).to_dataframe()
        if df.empty:
            logger.info("No more pattern groups to process. Escalating.")
            tool_context.actions.escalate = True
            return json.dumps({"status": "complete", "message": "No more patterns to process."})

        result_json = df.to_json(orient='records')
        logger.info("Found next pattern batch: %s", result_json)
        return result_json
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error fetching next pattern batch: {e}")
        tool_context.actions.escalate = True
        return json.dumps({"status": "error", "message": str(e)})

def apply_bulk_pattern_update(categorized_json_string: str) -> str:
    """Applies categories to a batch of transaction patterns from a validated JSON string."""
    logger.info("Applying bulk pattern update...")
    validated_df = _validate_bulk_llm_results(categorized_json_string, ['description_prefix', 'transaction_type', 'channel'])
    if validated_df.empty:
        return json.dumps({"status": "success", "updated_count": 0, "message": "No valid pattern categorizations to apply."})

    try:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        bq_client.load_table_from_dataframe(validated_df, TEMP_TABLE_ID, job_config=job_config).result()
        logger.info(f"Loaded {len(validated_df)} pattern categorizations into temporary table.")
    
        merge_sql = f"""
            MERGE `{TABLE_ID}` T
            USING `{TEMP_TABLE_ID}` U
                ON SUBSTR(T.description_cleaned, 1, 40) = U.description_prefix
                AND T.transaction_type = U.transaction_type
                AND T.channel = U.channel
            WHEN MATCHED AND (T.category_l1 IS NULL OR T.category_l2 IS NULL) THEN
                UPDATE SET
                    T.category_l1 = U.category_l1,
                    T.category_l2 = U.category_l2,
                    T.categorization_method = 'llm-bulk-pattern-based',
                    T.categorization_update_timestamp = CURRENT_TIMESTAMP();
        """
        merge_job = bq_client.query(merge_sql)
        merge_job.result()
        updated_count = merge_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully bulk-updated %d records for patterns.", updated_count)
    
        summary = validated_df[['description_prefix', 'category_l1', 'category_l2']].to_dict(orient='records')
        return json.dumps({"status": "success", "updated_count": updated_count, "summary": summary})

    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during bulk pattern update: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# --- Phase 4: Transaction-Level AI Tools ---
def fetch_batch_for_ai_categorization(tool_context: ToolContext, batch_size: int = 200) -> str:
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
    """
    Updates the main BigQuery table with categories provided by the AI, and returns a detailed summary.
    """
    logger.info("Received AI results. Validating and preparing to update BigQuery.")
    validated_df = _validate_transaction_level_results(categorized_json_string)
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

        # Generate a summary of the applied categories
        category_summary = validated_df.groupby(['category_l1', 'category_l2']).size().reset_index(name='count').to_dict('records')

        return json.dumps({
            "status": "success",
            "updated_count": updated_count,
            "summary": category_summary
        })
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during update: {e}")
        return json.dumps({"status": "error", "message": f"Failed to update BigQuery: {e}"})
    except Exception as e:
        logger.error(f"❌ Unexpected error during BigQuery update: {e}")
        return json.dumps({"status": "error", "message": f"An unexpected error occurred during update: {e}"})


# --- 4. Agent Definitions ---

# --- Bulk Recurring Identification Loop ---
single_recurring_batch_agent = LlmAgent(
    name="single_recurring_batch_agent",
    model="gemini-2.5-flash",
    tools=[get_recurring_candidates_batch, apply_bulk_recurring_flags],
    instruction="""
    Your purpose is to perform one complete cycle of BATCH recurring transaction identification.

    **Your process is a strict, three-step sequence:**
    1.  **FETCH BATCH:** First, call `get_recurring_candidates_batch` to get a list of potential recurring merchants.
        - If the tool returns a "complete" status, you must stop and escalate.
    2.  **ANALYZE & UPDATE BATCH:** Analyze the JSON data for ALL merchants. Based on the evidence, decide if the merchant represents a recurring charge. Look for clues like:
        - `transaction_intervals_days`: A list of numbers showing the days between transactions. Look for consistent patterns (e.g., `[30, 31, 29]` for monthly, or `[7, 7, 7]` for weekly). This is a very strong signal.
        - `has_recurring_keywords` being true.
        - A low standard deviation (`stddev_amount`) relative to the average amount (`avg_amount`), indicating consistent payment amounts.
        - Keywords in the `example_transactions` like 'monthly', 'subscription', etc.
        
        Then, call `apply_bulk_recurring_flags` ONCE with a single JSON array containing your decisions. Your JSON output should be a list of objects, each with `merchant_name_cleaned`, `transaction_type`, and a boolean `is_recurring`. **Only include merchants you confidently identify as recurring.**
    3.  **REPORT BATCH:** The update tool will return a JSON object with `updated_count` and a `summary`. Use this to create a user-friendly markdown report. For example: "🔍 Identified 'spotify' and 'netflix' as recurring, flagging 24 new transactions."
    """,
)

recurring_identification_loop = LoopAgent(
    name="recurring_identification_loop",
    description="This agent starts an intelligent, AI-driven process to find and flag recurring transactions like subscriptions and bills. It processes merchants in batches and provides real-time summaries.",
    sub_agents=[single_recurring_batch_agent],
    max_iterations=10
)

# --- Bulk Merchant Processing Loop ---
single_merchant_batch_agent = LlmAgent(
    name="single_merchant_batch_agent",
    model="gemini-2.5-flash",
    tools=[get_merchant_batch_to_categorize, apply_bulk_merchant_update],
    instruction=f"""
    Your purpose is to perform one complete cycle of BATCH merchant-based transaction categorization.

    **Your process is a strict, three-step sequence:**
    1.  **FETCH BATCH:** First, call `get_merchant_batch_to_categorize` to get a batch of up to 10 merchant groups.
        - If the tool returns a "complete" status, you must stop and escalate.
    2.  **ANALYZE & UPDATE BATCH:** Analyze the JSON data for ALL merchants in the batch. For each one, determine the correct `category_l1` and `category_l2` from the valid list: {VALID_CATEGORIES_JSON_STR}.
        
        Then, call `apply_bulk_merchant_update` ONCE with a single JSON string. **Crucially, your JSON for each merchant MUST include the `merchant_name_cleaned`, the original `transaction_type`, `category_l1`, and `category_l2`.**
    3.  **REPORT BATCH:** The update tool will return a JSON object with `updated_count` and a `summary`. Use this to create a user-friendly markdown report summarizing the batch. For example: "🛒 Processed a batch of 3 merchants, updating 112 transactions. Key updates include 'grubhub' to Dining & Drinks and 'uber' to Auto & Transport."
    """,
)

merchant_categorization_loop = LoopAgent(
    name="merchant_categorization_loop",
    description="This agent starts an efficient, automated categorization process. In each cycle, it processes a BATCH of the most common uncategorized merchants, providing a real-time summary for each batch. It continues until no more merchant groups are found.",
    sub_agents=[single_merchant_batch_agent],
    max_iterations=10
)


# --- Bulk Pattern Processing Loop ---
single_pattern_batch_agent = LlmAgent(
    name="single_pattern_batch_agent",
    model="gemini-2.5-flash",
    tools=[get_pattern_batch_to_categorize, apply_bulk_pattern_update],
    instruction=f"""
    Your purpose is to perform one complete cycle of BATCH pattern-based transaction categorization.

    **Your process is a strict, three-step sequence:**
    1.  **FETCH BATCH:** First, you MUST call `get_pattern_batch_to_categorize` to get a batch of up to 10 pattern groups.
        - If the tool returns a "complete" status, you must stop and escalate.
    2.  **ANALYZE & UPDATE BATCH:** Analyze the JSON data for ALL patterns. For each one, determine the correct `category_l1` and `category_l2` from the valid list: {VALID_CATEGORIES_JSON_STR}. Then, call `apply_bulk_pattern_update` ONCE with a single JSON string for the entire batch.
    3.  **REPORT BATCH:** The update tool returns `updated_count` and a `summary`. Use this to create a user-friendly markdown report. For example: "🧾 Processed a batch of 5 patterns, updating 88 transactions. This included patterns like 'payment thank you' being set to Credit Card Payment."
    """,
)

pattern_categorization_loop = LoopAgent(
    name="pattern_categorization_loop",
    description="This agent starts an advanced, batch-based categorization on common transaction description patterns. It repeatedly finds a batch of common patterns, categorizes them, and provides a real-time summary. This is best used after the merchant-based tool.",
    sub_agents=[single_pattern_batch_agent],
    max_iterations=10
)


# --- Transaction-Level Processing Loop ---
single_transaction_categorizer_agent = LlmAgent(
    name="single_transaction_categorizer_agent",
    model="gemini-2.5-flash",
    tools=[fetch_batch_for_ai_categorization, update_categorizations_in_bigquery],
    instruction=f"""
    Your purpose is to perform one complete cycle of detailed, transaction-by-transaction categorization and report the result with enhanced detail.

    **Your process is a strict, three-step sequence:**
    1.  **FETCH:** First, you MUST call the `fetch_batch_for_ai_categorization` tool to get a batch of individual transactions.
        - If the tool returns a "complete" status, you must stop and escalate immediately. Do not proceed.
    2.  **CATEGORIZE & UPDATE:** Second, for the batch of transactions you just fetched, you MUST call `update_categorizations_in_bigquery`.
        - You will generate a `categorized_json_string` by analyzing each transaction.
        - This string MUST be a JSON array of objects.
        - Each object MUST have exactly three keys: `transaction_id`, `category_l1`, and `category_l2`.
        - You MUST use a valid L1 and L2 category from this list: {VALID_CATEGORIES_JSON_STR}.
        - Example: `[ {{"transaction_id": "123", "category_l1": "Expense", "category_l2": "Shopping"}}, {{"transaction_id": "456", "category_l1": "Expense", "category_l2": "Groceries"}} ]`
    3.  **REPORT:** Third, the update tool will return a JSON object containing `updated_count` and a `summary` of the categories. You MUST present this information clearly in a user-friendly markdown format. For example:
        "✅ Processed a batch of 198 transactions.
        - **Shopping**: 75 transactions
        - **Groceries**: 50 transactions
        - **Dining & Drinks**: 73 transactions
        Now moving to the next batch..."

    Execute this cycle repeatedly. Your primary goal is accuracy and adherence to the specified JSON format.
    """,
)

transaction_categorization_loop = LoopAgent(
    name="transaction_categorization_loop",
    description="This agent starts the final, most granular categorization. It automatically processes any remaining transactions one by one, providing a detailed, real-time summary for each batch until the job is complete.",
    sub_agents=[single_transaction_categorizer_agent],
    # A high max_iterations allows the loop to process a large number of transactions.
    # With a batch size of 200, 50 iterations can process up to 10,000 transactions.
    max_iterations=50
)

# --- Root Orchestrator Agent ---
root_agent = Agent(
    name="transaction_categorizer_orchestrator",
    model="gemini-2.5-flash",
    tools=[
        audit_data_quality,
        run_cleansing_and_rules_categorization,
        run_recurring_transaction_harmonization,
        reset_all_categorizations,
        AgentTool(agent=recurring_identification_loop),
        AgentTool(agent=merchant_categorization_loop),
        AgentTool(agent=pattern_categorization_loop),
        AgentTool(agent=transaction_categorization_loop),
    ],
    instruction="""
    You are an elite financial transaction data analyst 🤖. Your purpose is to guide the user through a multi-step transaction categorization process. Be clear, concise, proactive, and use markdown and emojis to make your responses easy to read.

    **Your Standard Workflow:**
    1.  **Greeting & Audit:** Start by greeting the user warmly. Always recommend running an `audit_data_quality` check first to get a baseline of the data's health. Present the full report from the tool.
    2.  **Cleansing & Rules:** After the audit, suggest running `run_cleansing_and_rules_categorization`. When it's done, present the tool's formatted response clearly to the user.
    3.  **Identify Recurring (AI-Powered):** Next, recommend starting the `recurring_identification_loop`. Explain that this agent uses AI to intelligently find subscriptions and regular bills, processing them in batches and providing real-time updates.
    4.  **Harmonize Recurring:** After identifying recurring payments, recommend running `run_recurring_transaction_harmonization`. Explain that this is a high-confidence step that makes sure all recurring payments from the same merchant have the same category.
    5.  **Bulk AI (Merchant-Based):** After harmonization, it's time for the first AI step. Offer to start the `merchant_categorization_loop`. Explain that this is an efficient automated process that will categorize transactions in large batches, providing clear, step-by-step updates for each batch.
    6.  **Bulk AI (Pattern-Based):** After the merchant-based step is complete, propose the next level of bulk categorization using the `pattern_categorization_loop`. Explain that this is a more advanced scan that will automatically find and categorize transactions based on common description patterns, also in efficient batches.
    7.  **Transactional AI Categorization:** Once all bulk updates are done, offer to begin the final, most detailed step by starting the `transaction_categorization_loop`.
        -   **Before** starting this loop, you MUST inform the user with this exact message: "Great! I am now starting the final automated categorization agent. 🕵️‍♂️ This may take several minutes depending on the number of transactions. **You will see real-time updates below as each batch is completed.** I will let you know when the process is finished."
        -   When the loop finishes and control returns to you, you must inform the user that the process is complete with a concluding message, for example: "🎉 **All Done!** The automated categorization process has successfully completed. All remaining transactions have now been processed by the AI. You can run another data quality audit to see the results."

    **Special Commands (Handle with Extreme Care):**
    - **Resetting Data:** The `reset_all_categorizations` tool is highly destructive. **Never** suggest it unless the user explicitly asks to "start over," "reset everything," or uses similar direct language.
        -   First, call the tool with `confirm=False`.
        -   Present the exact markdown `message` from the tool's response to the user.
        -   Only if the user confirms with "yes", "proceed", or similar affirmative language, call the tool a second time with `confirm=True`. Present the final confirmation message from the tool.
    """
)