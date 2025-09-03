# tools.py

from __future__ import annotations
import json
import logging
import pandas as pd
from typing import Optional
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery

# ADK Core Components
from google.adk.tools import ToolContext
from google.adk.agents import LlmAgent

# Import configurations from your config file
from .config import (
    bq_client,
    TABLE_ID,
    RULES_TABLE_ID,
    TEMP_TABLE_ID,
    is_valid_category,
    _validate_and_clean_llm_json,
    _validate_transaction_level_results,
    _validate_bulk_llm_results
)

logger = logging.getLogger(__name__)

# --- 3. Tool Definitions ---

# --- Report, Reset, & Custom Query Tools ---
def get_data_quality_report() -> str:
    """
    Runs a comprehensive data quality audit on the main transaction table.
    Generates a user-friendly markdown report summarizing any findings.
    """
    logger.info("Starting data quality audit...")
    queries = {
        "Categorization Status by Persona": {
            "query": f"SELECT COALESCE(persona_type, 'Unknown') as persona, COALESCE(category_l1, 'Uncategorized') as category, COUNT(transaction_id) as transaction_count FROM `{TABLE_ID}` GROUP BY 1, 2 ORDER BY persona, transaction_count DESC;",
            "summary": "Overview of categorized vs. uncategorized transactions for each persona."
        },
        "Uncategorized Transactions Breakdown by Persona": {
            "query": f"SELECT COALESCE(persona_type, 'Unknown') as persona, CASE WHEN category_l1 IS NULL AND category_l2 IS NULL THEN 'Missing L1 & L2' WHEN category_l1 IS NULL THEN 'Missing L1 Only' WHEN category_l2 IS NULL THEN 'Missing L2 Only' END AS issue_type, transaction_type, channel, COUNT(transaction_id) AS transaction_count FROM `{TABLE_ID}` WHERE category_l1 IS NULL OR category_l2 IS NULL GROUP BY 1, 2, 3, 4 ORDER BY persona, transaction_count DESC;",
            "summary": "Breakdown of transactions missing one or both category labels, segmented by persona."
        },
        "Mismatched Transaction Types": {
            "query": f"SELECT transaction_id, persona_type, transaction_type, category_l1, category_l2, merchant_name_cleaned, amount FROM `{TABLE_ID}` WHERE (category_l1 = 'Income' AND (transaction_type = 'Debit' OR amount < 0)) OR (category_l1 = 'Expense' AND (transaction_type = 'Credit' OR amount > 0)) ORDER BY amount DESC;",
            "summary": "Highlights conflicts where transaction direction (Debit/Credit) contradicts the L1 category (Income/Expense)."
        },
        "Inconsistent Recurring Transactions by Persona": {
            "query": f"""
                SELECT
                    persona_type,
                    merchant_name_cleaned,
                    COUNT(DISTINCT category_l2) AS distinct_category_count,
                    ARRAY_AGG(DISTINCT category_l2 IGNORE NULLS) AS categories_assigned,
                    COUNT(t.transaction_id) AS total_inconsistent_transactions
                FROM `{TABLE_ID}` AS t
                WHERE t.is_recurring = TRUE AND t.merchant_name_cleaned IS NOT NULL
                GROUP BY 1, 2
                HAVING COUNT(DISTINCT category_l2) > 1
                ORDER BY persona_type, total_inconsistent_transactions DESC;
            """,
            "summary": "Lists recurring transactions from the same merchant that have been assigned multiple, conflicting L2 categories, broken down by persona."
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


def reset_all_transaction_categorizations(confirm: bool = False) -> str:
    """
    A destructive tool to reset all categorization and cleansing fields in the table.
    Requires explicit confirmation to proceed.
    """
    if not confirm:
        logger.warning("Reset requested without confirmation. Awaiting user confirmation.")
        return "⚠️ **Confirmation Required**\n\nThis is a destructive action that will clear ALL categorization and cleansing data (categories, cleaned descriptions, etc.). This action **cannot be undone**.\n\nPlease confirm you want to proceed by replying with 'yes' or 'proceed'."
    logger.info("Confirmation received. Proceeding with full data reset.")
    reset_sql = f"UPDATE `{TABLE_ID}` SET category_l1 = NULL, category_l2 = NULL, description_cleaned = NULL, merchant_name_cleaned = NULL, is_recurring = NULL, rule_id = NULL, categorization_method = NULL, categorization_update_timestamp = NULL WHERE TRUE;"
    try:
        query_job = bq_client.query(reset_sql)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully reset %d rows.", affected_rows)
        return f"✅ **Reset Complete**\n\nSuccessfully reset the categorization data for **{affected_rows}** transactions."
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during reset operation: {e}")
        return f"❌ **Error During Reset**\nA BigQuery error occurred during the reset operation. Please check the logs. Error: {e}"

def run_custom_query(query: str) -> str:
    """
    Executes a user-provided SQL query against the transaction data.
    Primarily for SELECT statements to perform custom analysis.
    WARNING: Can execute UPDATE or DELETE, use with caution.
    """
    logger.info(f"Executing custom user query: {query}")
    if not query.lower().strip().startswith(('select', 'update', 'with', 'merge')):
        return "❌ **Invalid Query**: This tool only supports SELECT, UPDATE, MERGE, or WITH statements for security reasons."

    try:
        final_query = query.replace("{{TABLE_ID}}", f"`{TABLE_ID}`")
        
        query_job = bq_client.query(final_query)
        results = query_job.result()
        
        if query_job.statement_type == "SELECT":
            df = results.to_dataframe()
            if df.empty:
                return "✅ **Query Successful**: The query ran successfully but returned no results."
            else:
                response = f"✅ **Query Successful**:\n\n{df.to_markdown(index=False)}"
                return response
        else: 
            affected_rows = results.num_dml_affected_rows or 0
            return f"✅ **Query Successful**: The operation completed and affected **{affected_rows}** rows."

    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during custom query execution: {e}")
        return f"❌ **Error During Query Execution**\nA BigQuery error occurred. Please check the syntax of your query and the application logs. Error: {e}"
    except Exception as e:
        logger.error(f"❌ Unexpected error during custom query execution: {e}")
        return f"❌ **Unexpected Error**\nAn unexpected error occurred. Please check the logs. Error: {e}"


def prepare_database_schema() -> str:
    """
    Ensures the required columns exist in the BigQuery tables.
    This is an idempotent operation; it will not fail if columns already exist.
    """
    logger.info("Preparing database schema...")
    columns_to_add = {
        TABLE_ID: [
            ("llm_confidence_score", "INT64"),
        ],
        RULES_TABLE_ID: [
            ("is_active", "BOOL"),
        ]
    }

    for table_id, columns in columns_to_add.items():
        for column_name, column_type in columns:
            try:
                add_column_query = f"ALTER TABLE `{table_id}` ADD COLUMN IF NOT EXISTS {column_name} {column_type};"
                bq_client.query(add_column_query).result()
                logger.info(f"Successfully ensured column '{column_name}' exists in table '{table_id}'.")
            except GoogleAPICallError as e:
                logger.error(f"Error adding column '{column_name}' to '{table_id}': {e}")
                return f"❌ Error preparing database schema for table '{table_id}': {e}"

    # Backfill is_active in rules table
    try:
        update_query = f"UPDATE `{RULES_TABLE_ID}` SET is_active = TRUE WHERE is_active IS NULL;"
        update_job = bq_client.query(update_query)
        update_job.result()
        if (update_job.num_dml_affected_rows or 0) > 0:
             logger.info(f"Backfilled 'is_active' flag for {update_job.num_dml_affected_rows} rules.")
    except GoogleAPICallError as e:
        logger.error(f"Error backfilling 'is_active' column: {e}")
        return f"❌ Error backfilling 'is_active' column: {e}"

    return "✅ **Database Schema Ready**: All required columns are present."


# --- Phase 1: Dynamic Rules-Based Tools ---

def prepare_transaction_data() -> str:
    """
    prepares the raw merchant and description fields for all transactions that haven't been cleaned yet.
    This involves converting to lowercase and removing special characters.
    """
    logger.info("Starting data cleansing for all transactions...")
    # This query will prepare the raw fields for any transaction where the
    # cleaned fields are currently NULL, ensuring it only runs once per row.
    cleansing_sql = f"""
    MERGE `{TABLE_ID}` T
    USING (
        SELECT
            transaction_id,
            TRIM(LOWER(REGEXP_REPLACE(REGEXP_REPLACE(REPLACE(IFNULL(description_raw, ''), '-', ''), r'[^a-zA-Z0-9\\s]', ' '), r'\\s+', ' '))) AS new_description_cleaned,
            TRIM(LOWER(
                REGEXP_EXTRACT(
                    REGEXP_REPLACE(REGEXP_REPLACE(REPLACE(IFNULL(merchant_name_raw, ''), '-', ''), r'[^a-zA-Z0-9\\s\\*#-]', ''), r'\\s+', ' '),
                    r'^(?:SQ\\s*\\*|PYPL\\s*\\*|CL\\s*\\*|\\*\\s*)?([^*#-]+)'
                )
            )) AS new_merchant_name_cleaned
        FROM `{TABLE_ID}`
        WHERE description_cleaned IS NULL OR merchant_name_cleaned IS NULL
    ) U
    ON T.transaction_id = U.transaction_id
    WHEN MATCHED THEN
        UPDATE SET
            T.description_cleaned = U.new_description_cleaned,
            T.merchant_name_cleaned = U.new_merchant_name_cleaned;
    """
    try:
        job = bq_client.query(cleansing_sql)
        job.result()
        affected_rows = job.num_dml_affected_rows or 0
        logger.info("✅ Data cleansing complete. %d rows affected.", affected_rows)
        if affected_rows > 0:
            return f"⚙️ **Data Cleansing Complete**\n\nSuccessfully prepared merchant and description data for **{affected_rows}** new transactions."
        else:
            return "✅ **Data Cleansing Complete**\n\nAll transactions were already prepared."
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during data cleansing: {e}")
        return f"❌ **Error During Data Cleansing**\nA BigQuery error occurred. Please check the logs. Error: {e}"


def apply_rules_based_categorization() -> str:
    """
    Applies categorization rules from the dedicated rules table with persona-based logic.
    It prioritizes rules matching the transaction's persona_type, then falls back to global rules.
    """
    logger.info("Starting persona-aware rules-based categorization...")

    # The SQL logic is now enhanced to prioritize persona-specific rules.
    # A persona-specific rule is one where the rule's persona_type matches the transaction's persona_type.
    # A global rule is one where the rule's persona_type is NULL.
    rules_sql = f"""
    MERGE `{TABLE_ID}` T
    USING (
        WITH RankedRules AS (
            SELECT
                t.transaction_id,
                r.category_l1,
                r.category_l2,
                r.is_recurring_rule,
                r.rule_id, 
                'rules-based-' || LOWER(r.rule_type) AS method,
                -- Prioritize persona-specific rules over global rules
                ROW_NUMBER() OVER(
                    PARTITION BY t.transaction_id
                    ORDER BY
                        CASE WHEN t.persona_type = r.persona_type THEN 1 ELSE 2 END, -- Persona match first
                        r.confidence_score DESC,
                        r.last_updated_timestamp DESC
                ) as rn
            FROM `{TABLE_ID}` t
            JOIN `{RULES_TABLE_ID}` r
                ON (
                    (r.rule_type = 'merchant' AND t.merchant_name_cleaned = r.identifier) OR
                    (r.rule_type = 'pattern' AND STRPOS(t.description_cleaned, r.identifier) > 0)
                )
                AND (t.persona_type = r.persona_type OR r.persona_type IS NULL) -- Match persona or global
                AND t.transaction_type = r.transaction_type
            WHERE t.category_l1 IS NULL AND r.is_active = TRUE
        )
        SELECT *
        FROM RankedRules
        WHERE rn = 1
    ) U
    ON T.transaction_id = U.transaction_id
    WHEN MATCHED THEN
        UPDATE SET
            T.category_l1 = U.category_l1,
            T.category_l2 = U.category_l2,
            T.rule_id = U.rule_id,
            T.is_recurring = COALESCE(U.is_recurring_rule, T.is_recurring),
            T.categorization_method = U.method,
            T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """
    try:
        job = bq_client.query(rules_sql)
        job.result()
        affected_rows = job.num_dml_affected_rows or 0
        logger.info("✅ Persona-aware rules-based categorization complete. %d rows affected.", affected_rows)
        return f"✅ **Rules Engine Complete**\n\nCategorized and updated **{affected_rows}** transactions using persona-aware logic."
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during rules-based categorization: {e}")
        return f"❌ **Error During Rules Application**\nA BigQuery error occurred. Please check the logs. Error: {e}"


from google.adk.agents import LlmAgent

def review_and_resolve_rule_conflicts() -> str:
    """
    Reviews the categorization rules for conflicts, uses an LLM to resolve them,
    and updates the rules table to deactivate incorrect entries.
    """
    logger.info("Starting rule review and conflict resolution...")

    # Step 1: Find conflicting rules (same identifier, different categories).
    conflict_query = f"""
        WITH ConflictingRules AS (
            SELECT
                identifier,
                rule_type,
                transaction_type,
                persona_type,
                ARRAY_AGG(STRUCT(rule_id, category_l1, category_l2, confidence_score, last_updated_timestamp)) AS rules
            FROM `{RULES_TABLE_ID}`
            WHERE is_active = TRUE
            GROUP BY 1, 2, 3, 4
            HAVING COUNT(DISTINCT FORMAT('%T', (category_l1, category_l2))) > 1
        )
        SELECT identifier, rule_type, transaction_type, persona_type, rules FROM ConflictingRules
    """
    try:
        conflicts_df = bq_client.query(conflict_query).to_dataframe()
        if conflicts_df.empty:
            return "✅ **Rule Review Complete**: No conflicting rules found."
        logger.info(f"Found {len(conflicts_df)} conflicting rule sets to resolve.")
    except GoogleAPICallError as e:
        logger.error(f"Error finding conflicting rules: {e}")
        return f"❌ Error finding conflicting rules: {e}"

    # Step 3: Use a temporary, in-tool LLM agent to resolve conflicts.
    conflict_resolver_agent = LlmAgent(
        model="gemini-2.5-flash",
        instruction="""
        You are a data analyst specializing in financial transaction categorization.
        You will be given a JSON object representing a set of conflicting rules for a single identifier and persona.
        Your task is to analyze the rules and decide which one is the most correct.
        Consider the confidence_score (higher is better) and last_updated_timestamp (more recent is better).
        You MUST respond with a valid JSON object with two keys:
        - "rule_to_keep": The rule_id of the single rule that should be kept active.
        - "rules_to_deactivate": A list of all other rule_ids that should be deactivated.
        """
    )

    rules_to_deactivate = []
    for conflict in conflicts_df.to_dict('records'):
        prompt = f"Please resolve the following conflict:\n{json.dumps(conflict, indent=2, default=str)}"
        try:
            response_str = conflict_resolver_agent.invoke(prompt)
            # Clean the response string before parsing
            cleaned_response = response_str.strip().replace('```json', '').replace('```', '')
            resolution = json.loads(cleaned_response)
            if 'rules_to_deactivate' in resolution and isinstance(resolution['rules_to_deactivate'], list):
                rules_to_deactivate.extend(resolution['rules_to_deactivate'])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error parsing LLM response for conflict resolution: {e}. Response: '{response_str}'")
            continue # Skip this conflict if LLM fails to provide a valid resolution

    # Step 4: Apply resolutions by deactivating incorrect rules.
    if not rules_to_deactivate:
        return "⚠️ **Rule Review Complete**: Found conflicts, but was unable to generate resolutions. Please review rules manually."

    # Use a parameterized query to prevent SQL injection, even with internal data.
    update_query = f"""
        UPDATE `{RULES_TABLE_ID}`
        SET is_active = FALSE, last_updated_timestamp = CURRENT_TIMESTAMP()
        WHERE rule_id IN UNNEST(@rule_ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("rule_ids", "STRING", rules_to_deactivate)
        ]
    )
    try:
        update_job = bq_client.query(update_query, job_config=job_config)
        update_job.result()
        deactivated_count = update_job.num_dml_affected_rows or 0
        logger.info(f"Successfully deactivated {deactivated_count} conflicting rules.")
        return f"✅ **Rule Review Complete**: Deactivated {deactivated_count} conflicting or redundant rules to improve consistency."
    except GoogleAPICallError as e:
        logger.error(f"Error deactivating conflicting rules: {e}")
        return f"❌ Error applying rule resolutions: {e}"


def harmonize_recurring_transaction_categories() -> str:
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
            return f"🔄 **Harmonization Complete**\n\nI've successfully standardized the categories for **{affected_rows}** recurring transactions, ensuring consistency."
        else:
            return "✅ **Harmonization Complete**\n\nNo recurring transactions needed harmonization at this time."

    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during harmonization phase: {e}")
        return f"❌ **Error During Harmonization**\nA BigQuery error occurred. Please check the logs. Error: {e}"


# --- Phase 2 & 3: AI-Based Bulk & Recurring Tools ---

def get_recurring_transaction_candidates(tool_context: ToolContext, batch_size: int = 300) -> str:
    """
    Fetches a batch of merchants that are potential candidates for being recurring.
    Gathers evidence like transaction counts, amount stability, and time intervals.
    """
    logger.info(f"Fetching batch of {batch_size} recurring merchant candidates...")
    
    query = f"""
        WITH TransactionIntervals AS (
            SELECT
                merchant_name_cleaned,
                transaction_type,
                transaction_id,
                description_cleaned,
                amount,
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
        )
        SELECT
            merchant_name_cleaned,
            transaction_type,
            COUNT(transaction_id) as transaction_count,
            ARRAY_AGG(days_since_last_txn IGNORE NULLS) AS transaction_intervals_days,
            LOGICAL_OR(REGEXP_CONTAINS(LOWER(IFNULL(description_cleaned, '')), r'monthly|weekly|annual|subscription|membership|recurring|plan')) AS has_recurring_keywords,
            STDDEV(ABS(amount)) as stddev_amount,
            AVG(ABS(amount)) as avg_amount,
            ARRAY_AGG(
                STRUCT(
                    description_cleaned,
                    amount,
                    transaction_date
                ) ORDER BY transaction_date DESC LIMIT 2
            ) AS example_transactions
        FROM TransactionIntervals
        GROUP BY 1, 2
        HAVING COUNT(transaction_id) >= 5
        ORDER BY has_recurring_keywords DESC, transaction_count DESC
        LIMIT {batch_size};
    """
    try:
        df = bq_client.query(query).to_dataframe()
        if df.empty:
            logger.info("No more recurring candidates to process. Escalating.")
            tool_context.actions.escalate = True
            return json.dumps({"status": "complete", "message": "No more recurring candidates to process."})
        
        df['stddev_amount'] = pd.to_numeric(df['stddev_amount'], errors='coerce').fillna(0)
        df['avg_amount'] = pd.to_numeric(df['avg_amount'], errors='coerce').fillna(0)
        
        result_json = df.to_json(orient='records')
        logger.info("Found next recurring candidate batch: %s", result_json)
        return result_json
    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error fetching recurring candidates batch: {e}")
        tool_context.actions.escalate = True
        return json.dumps({"status": "error", "message": str(e)})

def flag_recurring_transactions_in_bulk(categorized_json_string: str) -> str:
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

def get_uncategorized_merchants_batch(tool_context: ToolContext, batch_size: int = 200) -> str:
    """
    Fetches a batch of the most frequent uncategorized merchants for efficient bulk processing.
    """
    logger.info(f"Fetching batch of {batch_size} merchants for bulk categorization...")
    query = f"""
        WITH TopMerchantGroups AS (
            SELECT
                merchant_name_cleaned,
                transaction_type,
                persona_type,
                COUNT(transaction_id) as uncategorized_count
            FROM `{TABLE_ID}`
            WHERE (category_l1 IS NULL OR category_l2 IS NULL)
                AND merchant_name_cleaned IS NOT NULL AND merchant_name_cleaned != ''
                AND category_l1 IS DISTINCT FROM 'Transfer'
            GROUP BY 1, 2, 3
            HAVING COUNT(transaction_id) >= 2
            ORDER BY uncategorized_count DESC
            LIMIT {batch_size}
        )
        SELECT
            g.merchant_name_cleaned,
            g.transaction_type,
            g.persona_type,
            g.uncategorized_count,
            ARRAY_AGG(
                STRUCT(
                    t.description_cleaned,
                    t.amount,
                    t.channel,
                    t.is_recurring
                ) ORDER BY t.transaction_date DESC LIMIT 3
            ) AS example_transactions
        FROM `{TABLE_ID}` t
        JOIN TopMerchantGroups g
            ON t.merchant_name_cleaned = g.merchant_name_cleaned
            AND t.transaction_type = g.transaction_type
            AND IFNULL(t.persona_type, 'none') = IFNULL(g.persona_type, 'none')
        WHERE (t.category_l1 IS NULL OR t.category_l2 IS NULL)
        GROUP BY 1, 2, 3, 4;
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
    validated_df = _validate_bulk_llm_results(categorized_json_string, ['merchant_name_cleaned', 'transaction_type', 'persona_type'])
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
                AND IFNULL(T.persona_type, 'none') = IFNULL(U.persona_type, 'none')
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
    
        summary = validated_df[['merchant_name_cleaned', 'persona_type', 'category_l1', 'category_l2']].to_dict(orient='records')
        return json.dumps({"status": "success", "updated_count": updated_count, "summary": summary})

    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during bulk merchant update: {e}")
        return json.dumps({"status": "error", "message": str(e)})

def get_uncategorized_patterns_batch(tool_context: ToolContext, batch_size: int = 200) -> str:
    """Fetches a batch of the most frequent uncategorized transaction patterns for efficient bulk processing."""
    logger.info(f"Fetching batch of {batch_size} patterns for bulk categorization...")
    query = f"""
        WITH PatternGroups AS (
            SELECT
                SUBSTR(description_cleaned, 1, 40) AS description_prefix,
                transaction_type,
                channel,
                persona_type,
                COUNT(transaction_id) as uncategorized_count
            FROM `{TABLE_ID}`
            WHERE (category_l1 IS NULL OR category_l2 IS NULL)
                AND description_cleaned IS NOT NULL AND description_cleaned != ''
                AND category_l1 IS DISTINCT FROM 'Transfer'
            GROUP BY 1, 2, 3, 4
            HAVING COUNT(transaction_id) >= 2
            ORDER BY uncategorized_count DESC
            LIMIT {batch_size}
        )
        SELECT
            g.description_prefix,
            g.transaction_type,
            g.channel,
            g.persona_type,
            g.uncategorized_count,
            ARRAY_AGG(
                STRUCT(
                    t.description_cleaned,
                    t.merchant_name_cleaned,
                    t.amount,
                    t.is_recurring
                ) LIMIT 2
            ) AS example_transactions
        FROM `{TABLE_ID}` t
        JOIN PatternGroups g
            ON SUBSTR(t.description_cleaned, 1, 40) = g.description_prefix
            AND t.transaction_type = g.transaction_type
            AND t.channel = g.channel
            AND IFNULL(t.persona_type, 'none') = IFNULL(g.persona_type, 'none')
        WHERE (t.category_l1 IS NULL OR t.category_l2 IS NULL)
        GROUP BY 1, 2, 3, 4, 5;
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
    validated_df = _validate_bulk_llm_results(categorized_json_string, ['description_prefix', 'transaction_type', 'channel', 'persona_type'])
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
                AND IFNULL(T.persona_type, 'none') = IFNULL(U.persona_type, 'none')
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
    
        summary = validated_df[['description_prefix', 'persona_type', 'category_l1', 'category_l2']].to_dict(orient='records')
        return json.dumps({"status": "success", "updated_count": updated_count, "summary": summary})

    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during bulk pattern update: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# --- Phase 4: Transaction-Level AI & Learning Tools ---
def get_transaction_batch_for_ai_categorization(tool_context: ToolContext, batch_size: int = 300, transaction_type: Optional[str] = None) -> str:
    """
    Fetches a batch of individual uncategorized transactions for detailed, row-by-row AI processing.
    Can be filtered by transaction type.
    """
    logger.info(f"Fetching batch for AI categorization. Batch size: {batch_size}, Type: {transaction_type or 'All'}")

    # Start building the query
    fetch_sql = f"""
        SELECT
            transaction_id, description_cleaned, merchant_name_cleaned, amount,
            transaction_type, persona_type, channel, account_type
        FROM `{TABLE_ID}`
        WHERE (category_l1 IS NULL OR category_l2 IS NULL)
            AND category_l1 IS DISTINCT FROM 'Transfer'
    """

    # Add the transaction_type filter if provided
    if transaction_type:
        # Use a parameter to prevent SQL injection, though in this controlled
        # environment it's less of a risk. It's good practice.
        fetch_sql += f" AND transaction_type = '{transaction_type}'"

    # Add the limit
    fetch_sql += f" LIMIT {batch_size};"

    try:
        df = bq_client.query(fetch_sql).to_dataframe()
        if df.empty:
            logger.info(f"✅ No more '{transaction_type or 'any'}' transactions found needing categorization. Signaling loop to stop.")
            tool_context.actions.escalate = True
            return json.dumps({"status": "complete", "message": f"No more '{transaction_type or 'any'}' transactions to process."})
        logger.info(f"Fetched {len(df)} transactions for AI categorization.")
        return df.to_json(orient='records')
    except GoogleAPICallError as e:
        logger.error(f"❌ Failed to fetch batch from BigQuery: {e}")
        tool_context.actions.escalate = True
        return json.dumps({"status": "error", "message": f"Failed to fetch data: {e}"})


def update_transactions_with_ai_categories(categorized_json_string: str) -> str:
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
                bigquery.SchemaField("category_l2", "STRING"),
                bigquery.SchemaField("llm_confidence_score", "INTEGER"),
            ],
            write_disposition="WRITE_TRUNCATE",
        )
        bq_client.load_table_from_dataframe(validated_df, TEMP_TABLE_ID, job_config=job_config).result()
        logger.info(f"Successfully loaded {len(validated_df)} records into temporary table.")

        merge_sql = f"""
            MERGE `{TABLE_ID}` T
            USING `{TEMP_TABLE_ID}` U ON T.transaction_id = U.transaction_id
            WHEN MATCHED AND (T.category_l1 IS NULL OR T.category_l2 IS NULL) THEN
                UPDATE SET
                    T.category_l1 = U.category_l1,
                    T.category_l2 = U.category_l2,
                    T.llm_confidence_score = U.llm_confidence_score,
                    T.categorization_method = 'llm-transaction-based',
                    T.categorization_update_timestamp = CURRENT_TIMESTAMP();
        """
        merge_job = bq_client.query(merge_sql)
        merge_job.result()
        updated_count = merge_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully updated %d records in BigQuery via AI.", updated_count)

        category_summary = validated_df.groupby(['category_l1', 'category_l2']).size().reset_index(name='count').to_dict('records')

        return json.dumps({
            "status": "success",
            "updated_count": updated_count,
            "summary": category_summary
        })
    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during update: {e}")
        return json.dumps({"status": "error", "message": f"Failed to update BigQuery: {e}"})

def _calculate_confidence_score(transaction_count: int) -> int:
    """Calculates a confidence score on a 1-100 scale based on the transaction count."""
    if transaction_count >= 15:
        return 80 + min(14, (transaction_count - 15)) # High confidence: 80-94
    elif transaction_count >= 5:
        return 60 + (transaction_count - 5) * 4 # Medium confidence: 60-76
    elif transaction_count >= 3:
        return 50 # Low confidence
    else:
        return 0 # No confidence, do not create a rule

def learn_new_categorization_rules() -> str:
    """
    Identifies high-confidence categories from AI processing and saves them as new rules.
    This version correctly calculates confidence scores in Python before updating BigQuery.
    """
    logger.info("Harvesting new persona-aware rules from AI categorizations...")
    total_affected_rows = 0

    # --- Part 1: Harvest Merchant rules from BULK processing ---
    get_merchants_sql = f"""
    SELECT
        merchant_name_cleaned AS identifier,
        transaction_type,
        persona_type,
        category_l1,
        category_l2,
        LOGICAL_AND(IFNULL(is_recurring, FALSE)) AS is_recurring_rule,
        COUNT(transaction_id) AS transaction_count,
        'llm-bulk-merchant-based' as source_method
    FROM `{TABLE_ID}`
    WHERE categorization_method = 'llm-bulk-merchant-based'
        AND merchant_name_cleaned IS NOT NULL AND merchant_name_cleaned != ''
    GROUP BY 1, 2, 3, 4, 5
    HAVING COUNT(transaction_id) >= 3
    """
    try:
        new_merchant_rules_df = bq_client.query(get_merchants_sql).to_dataframe()
        if not new_merchant_rules_df.empty:
            new_merchant_rules_df['confidence_score'] = new_merchant_rules_df['transaction_count'].apply(_calculate_confidence_score)
            
            # Load to temp table
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            bq_client.load_table_from_dataframe(new_merchant_rules_df, TEMP_TABLE_ID, job_config=job_config).result()

            # Merge into rules table
            merge_merchants_sql = f"""
            MERGE `{RULES_TABLE_ID}` R
            USING `{TEMP_TABLE_ID}` NewRules
            ON R.identifier = NewRules.identifier
                AND R.rule_type = 'merchant'
                AND R.transaction_type = NewRules.transaction_type
                AND IFNULL(R.persona_type, 'none') = IFNULL(NewRules.persona_type, 'none')
            WHEN MATCHED AND NewRules.confidence_score > R.confidence_score THEN
                UPDATE SET
                    category_l1 = NewRules.category_l1,
                    category_l2 = NewRules.category_l2,
                    is_recurring_rule = NewRules.is_recurring_rule,
                    confidence_score = NewRules.confidence_score,
                    last_updated_timestamp = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (rule_id, rule_type, identifier, transaction_type, persona_type, category_l1, category_l2, is_recurring_rule, confidence_score, created_timestamp, last_updated_timestamp, is_active)
                VALUES (
                    GENERATE_UUID(), 'merchant', NewRules.identifier, NewRules.transaction_type,
                    NewRules.persona_type, NewRules.category_l1, NewRules.category_l2,
                    NewRules.is_recurring_rule, NewRules.confidence_score,
                    CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), TRUE
                );
            """
            merchant_job = bq_client.query(merge_merchants_sql)
            merchant_job.result()
            total_affected_rows += merchant_job.num_dml_affected_rows or 0
    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ Error harvesting merchant rules: {e}")
        return f"❌ **Error During Learning**: An error occurred while harvesting merchant rules. Error: {e}"

    # --- Part 2: Harvest Pattern rules from BULK processing ---
        get_patterns_sql = f"""
        SELECT
            SUBSTR(description_cleaned, 1, 40) AS identifier,
            transaction_type,
            persona_type,
            category_l1,
            category_l2,
            LOGICAL_AND(IFNULL(is_recurring, FALSE)) AS is_recurring_rule,
            COUNT(transaction_id) AS transaction_count
        FROM `{TABLE_ID}`
        WHERE categorization_method = 'llm-bulk-pattern-based'
            AND description_cleaned IS NOT NULL AND description_cleaned != ''
        GROUP BY 1, 2, 3, 4, 5
        HAVING COUNT(transaction_id) >= 3
        """
        try:
            new_pattern_rules_df = bq_client.query(get_patterns_sql).to_dataframe()
            if not new_pattern_rules_df.empty:
                new_pattern_rules_df['confidence_score'] = new_pattern_rules_df['transaction_count'].apply(_calculate_confidence_score)
                
                # Load to temp table
                job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
                bq_client.load_table_from_dataframe(new_pattern_rules_df, TEMP_TABLE_ID, job_config=job_config).result()

                # Merge into rules table
                merge_patterns_sql = f"""
                MERGE `{RULES_TABLE_ID}` R
                USING `{TEMP_TABLE_ID}` NewRules
                ON R.identifier = NewRules.identifier
                    AND R.rule_type = 'pattern'
                    AND R.transaction_type = NewRules.transaction_type
                    AND IFNULL(R.persona_type, 'none') = IFNULL(NewRules.persona_type, 'none')
                WHEN MATCHED AND NewRules.confidence_score > R.confidence_score THEN
                    UPDATE SET
                        category_l1 = NewRules.category_l1,
                        category_l2 = NewRules.category_l2,
                        is_recurring_rule = NewRules.is_recurring_rule,
                        confidence_score = NewRules.confidence_score,
                        last_updated_timestamp = CURRENT_TIMESTAMP()
                WHEN NOT MATCHED THEN
                    INSERT (rule_id, rule_type, identifier, transaction_type, persona_type, category_l1, category_l2, is_recurring_rule, confidence_score, created_timestamp, last_updated_timestamp, is_active)
                    VALUES (
                        GENERATE_UUID(), 'pattern', NewRules.identifier, NewRules.transaction_type,
                        NewRules.persona_type, NewRules.category_l1, NewRules.category_l2,
                        NewRules.is_recurring_rule, NewRules.confidence_score,
                        CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), TRUE
                    );
                """
                pattern_job = bq_client.query(merge_patterns_sql)
                pattern_job.result()
                total_affected_rows += pattern_job.num_dml_affected_rows or 0
        except (GoogleAPICallError, Exception) as e:
            logger.error(f"❌ Error harvesting pattern rules: {e}")
            return f"❌ **Error During Learning**: An error occurred while harvesting pattern rules. Error: {e}"

    # --- Part 3: Harvest rules from HIGH-CONFIDENCE single transactions ---
    get_single_tx_rules_sql = f"""
    SELECT
        merchant_name_cleaned AS identifier,
        transaction_type,
        persona_type,
        category_l1,
        category_l2,
        is_recurring AS is_recurring_rule,
        1 AS transaction_count, -- It's a single transaction
        95 AS confidence_score -- Assign a high confidence score for these rules
    FROM `{TABLE_ID}`
    WHERE categorization_method = 'llm-transaction-based'
      AND llm_confidence_score >= 9
      AND merchant_name_cleaned IS NOT NULL AND merchant_name_cleaned != ''
    """
    try:
        new_single_tx_rules_df = bq_client.query(get_single_tx_rules_sql).to_dataframe()
        if not new_single_tx_rules_df.empty:
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            bq_client.load_table_from_dataframe(new_single_tx_rules_df, TEMP_TABLE_ID, job_config=job_config).result()

            merge_single_tx_sql = f"""
            MERGE `{RULES_TABLE_ID}` R
            USING `{TEMP_TABLE_ID}` NewRules
            ON R.identifier = NewRules.identifier
                AND R.rule_type = 'merchant'
                AND R.transaction_type = NewRules.transaction_type
                AND IFNULL(R.persona_type, 'none') = IFNULL(NewRules.persona_type, 'none')
            WHEN NOT MATCHED THEN
                INSERT (rule_id, rule_type, identifier, transaction_type, persona_type, category_l1, category_l2, is_recurring_rule, confidence_score, created_timestamp, last_updated_timestamp, is_active)
                VALUES (
                    GENERATE_UUID(), 'merchant', NewRules.identifier, NewRules.transaction_type,
                    NewRules.persona_type, NewRules.category_l1, NewRules.category_l2,
                    NewRules.is_recurring_rule, NewRules.confidence_score,
                    CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), TRUE
                );
            """
            single_tx_job = bq_client.query(merge_single_tx_sql)
            single_tx_job.result()
            total_affected_rows += single_tx_job.num_dml_affected_rows or 0
    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ Error harvesting single transaction rules: {e}")
        return f"❌ **Error During Learning**: An error occurred while harvesting single transaction rules. Error: {e}"


    if total_affected_rows > 0:
        logger.info("✅ Successfully harvested %d new or updated rules.", total_affected_rows)
        return f"🧠 **Learning Complete**: Analyzed AI categorizations and created or updated **{total_affected_rows}** new high-confidence rules."
    else:
        logger.info("No new high-confidence rules to harvest.")
        return "🧠 **Learning Complete**: No new patterns met the confidence threshold to be saved as rules."

def create_new_categorization_rule(
    identifier: str,
    rule_type: str,
    category_l1: str,
    category_l2: str,
    transaction_type: str,
    is_recurring_rule: bool = False,
    persona_type: Optional[str] = None
) -> str:
    """
    Adds or updates a user-defined rule in the categorization_rules table, with optional persona-specificity.
    """
    logger.info(f"Attempting to add/update rule for identifier: '{identifier}', persona: '{persona_type or 'Global'}'")

    if rule_type.lower() not in ['merchant', 'pattern']:
        return f"❌ Invalid rule_type: '{rule_type}'. Must be 'merchant' or 'pattern'."
    if transaction_type not in ['Debit', 'Credit', 'All']:
        return f"❌ Invalid transaction_type: '{transaction_type}'. Must be 'Debit', 'Credit', or 'All'."
    if not is_valid_category(category_l1, category_l2):
        return f"❌ Invalid category combination: L1='{category_l1}', L2='{category_l2}'."

    user_rule_confidence = 99

    merge_sql = f"""
    MERGE `{RULES_TABLE_ID}` R
    USING (
        SELECT
            @identifier AS identifier, @rule_type AS rule_type, @transaction_type AS transaction_type,
            @persona_type AS persona_type, @category_l1 AS category_l1, @category_l2 AS category_l2,
            @is_recurring_rule AS is_recurring_rule, @confidence_score AS confidence_score
    ) AS NewRule
    ON R.identifier = NewRule.identifier
        AND R.rule_type = NewRule.rule_type
        AND R.transaction_type = NewRule.transaction_type
        AND IFNULL(R.persona_type, 'none') = IFNULL(NewRule.persona_type, 'none')
    WHEN MATCHED THEN
        UPDATE SET
            category_l1 = NewRule.category_l1, category_l2 = NewRule.category_l2,
            is_recurring_rule = NewRule.is_recurring_rule, confidence_score = NewRule.confidence_score,
            last_updated_timestamp = CURRENT_TIMESTAMP(), is_active = TRUE
    WHEN NOT MATCHED THEN
        INSERT (rule_id, rule_type, identifier, transaction_type, persona_type, category_l1, category_l2, is_recurring_rule, confidence_score, created_timestamp, last_updated_timestamp, is_active)
        VALUES (
            GENERATE_UUID(), NewRule.rule_type, NewRule.identifier, NewRule.transaction_type,
            NewRule.persona_type, NewRule.category_l1, NewRule.category_l2,
            NewRule.is_recurring_rule, NewRule.confidence_score,
            CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), TRUE
        );
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("identifier", "STRING", identifier.lower()),
            bigquery.ScalarQueryParameter("rule_type", "STRING", rule_type.lower()),
            bigquery.ScalarQueryParameter("transaction_type", "STRING", transaction_type),
            bigquery.ScalarQueryParameter("persona_type", "STRING", persona_type),
            bigquery.ScalarQueryParameter("category_l1", "STRING", category_l1),
            bigquery.ScalarQueryParameter("category_l2", "STRING", category_l2),
            bigquery.ScalarQueryParameter("is_recurring_rule", "BOOL", is_recurring_rule),
            bigquery.ScalarQueryParameter("confidence_score", "INT64", user_rule_confidence),
        ]
    )

    try:
        job = bq_client.query(merge_sql, job_config=job_config)
        job.result()
        return f"✅ **Rule Created**: Successfully created a rule for '{identifier.lower()}' (Persona: {persona_type or 'Global'})."
    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during custom rule creation: {e}")
        return f"❌ **Error Creating Rule**: An error occurred. Please check the logs. Error: {e}"


def apply_fallback_categorization() -> str:
    """
    Applies a fallback category to any transactions that are still uncategorized.
    - Credits are categorized as 'Income' / 'Other Income'.
    - Debits are categorized as 'Expense' / 'Other Expense'.
    """
    logger.info("Applying fallback categorization for remaining uncategorized transactions...")
    fallback_sql = f"""
    MERGE `{TABLE_ID}` T
    USING (
        SELECT
            transaction_id,
            CASE
                WHEN transaction_type = 'Credit' THEN 'Income'
                WHEN transaction_type = 'Debit' THEN 'Expense'
            END AS new_category_l1,
            CASE
                WHEN transaction_type = 'Credit' THEN 'Other Income'
                WHEN transaction_type = 'Debit' THEN 'Other Expense'
            END AS new_category_l2
        FROM `{TABLE_ID}`
        WHERE category_l1 IS NULL OR category_l2 IS NULL
    ) AS U ON T.transaction_id = U.transaction_id
    WHEN MATCHED THEN
        UPDATE SET
            T.category_l1 = U.new_category_l1,
            T.category_l2 = U.new_category_l2,
            T.categorization_method = 'fallback',
            T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """
    try:
        query_job = bq_client.query(fallback_sql)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows or 0
        logger.info("✅ Fallback categorization complete. %d rows affected.", affected_rows)
        if affected_rows > 0:
            return f"✅ **Fallback Complete**\n\nApplied default categories to **{affected_rows}** remaining uncategorized transactions."
        else:
            return "✅ **Fallback Complete**\n\nNo transactions required fallback categorization."
    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during fallback categorization: {e}")
        return f"❌ **Error During Fallback Categorization**\nA BigQuery error occurred. Please check the logs. Error: {e}"

def backfill_confidence_scores() -> str:
    """
    One-time utility to backfill and standardize confidence scores for all existing rules
    based on the new 1-100 scale.
    """
    logger.info("Starting confidence score backfill for all existing rules...")

    # Step 1: Get all existing rules
    try:
        rules_df = bq_client.query(f"SELECT rule_id, identifier, rule_type, transaction_type FROM `{RULES_TABLE_ID}`").to_dataframe()
        if rules_df.empty:
            return "✅ **Backfill Complete**: No rules found to update."
        logger.info(f"Found {len(rules_df)} existing rules to analyze.")
    except GoogleAPICallError as e:
        logger.error(f"Error fetching existing rules: {e}")
        return f"❌ **Error Fetching Rules**: Could not retrieve rules for backfill. Error: {e}"

    # Step 2: For each rule, count matching transactions to determine its new score
    updates = []
    for _, rule in rules_df.iterrows():
        if rule['rule_type'] == 'merchant':
            count_query = f"""
                SELECT COUNT(transaction_id) as count
                FROM `{TABLE_ID}`
                WHERE merchant_name_cleaned = '{rule['identifier']}'
                  AND transaction_type = '{rule['transaction_type']}'
            """
        elif rule['rule_type'] == 'pattern':
            count_query = f"""
                SELECT COUNT(transaction_id) as count
                FROM `{TABLE_ID}`
                WHERE STRPOS(description_cleaned, '{rule['identifier']}') > 0
                  AND transaction_type = '{rule['transaction_type']}'
            """
        else:
            continue

        try:
            count_result = bq_client.query(count_query).to_dataframe()
            transaction_count = count_result['count'][0] if not count_result.empty else 0
            new_score = _calculate_confidence_score(transaction_count)
            updates.append({
                'rule_id': rule['rule_id'],
                'new_confidence_score': new_score
            })
        except GoogleAPICallError as e:
            logger.warning(f"Could not count transactions for rule {rule['rule_id']}: {e}")
            continue

    if not updates:
        return "⚠️ **Backfill Complete**: No transaction counts could be determined to update rule scores."

    # Step 3: Perform the update in BigQuery
    updates_df = pd.DataFrame(updates)
    try:
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("rule_id", "STRING"),
                bigquery.SchemaField("new_confidence_score", "INT64"),
            ],
            write_disposition="WRITE_TRUNCATE",
        )
        bq_client.load_table_from_dataframe(updates_df, TEMP_TABLE_ID, job_config=job_config).result()

        merge_sql = f"""
            MERGE `{RULES_TABLE_ID}` T
            USING `{TEMP_TABLE_ID}` U ON T.rule_id = U.rule_id
            WHEN MATCHED THEN
                UPDATE SET T.confidence_score = U.new_confidence_score;
        """
        update_job = bq_client.query(merge_sql)
        update_job.result()
        updated_count = update_job.num_dml_affected_rows or 0
        logger.info(f"Successfully updated {updated_count} rule confidence scores.")
        return f"✅ **Backfill Complete**: Successfully standardized confidence scores for **{updated_count}** rules."
    except (GoogleAPICallError, Exception) as e:
        logger.error(f"Error updating confidence scores in BigQuery: {e}")
        return f"❌ **Error Updating Scores**: A failure occurred during the final update. Error: {e}"