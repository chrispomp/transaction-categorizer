# tools.py

from __future__ import annotations
import json
import logging
import pandas as pd
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
    _validate_and_clean_llm_json,
    _validate_transaction_level_results,
    _validate_bulk_llm_results
)

logger = logging.getLogger(__name__)

# --- 3. Tool Definitions ---

# --- Report, Reset, & Custom Query Tools ---
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
    results_markdown = "📊 **Data Quality Audit Report**\\n\\nHere's a summary of the data quality checks. Any tables below indicate areas that may need attention.\\n"
    for title, data in queries.items():
        results_markdown += f"---\\n\\n### {title}\\n*_{data['summary']}_*\\n\\n"
        try:
            df = bq_client.query(data["query"]).to_dataframe()
            if not df.empty:
                results_markdown += df.to_markdown(index=False) + "\\n\\n"
            else:
                results_markdown += "✅ **No issues found in this category.**\\n\\n"
        except GoogleAPICallError as e:
            logger.error(f"BigQuery error during '{title}' audit: {e}")
            results_markdown += f"❌ **Error executing query for `{title}`.** Please check the logs for details.\\n\\n"
        except Exception as e:
            logger.error(f"Unexpected error during '{title}' audit: {e}")
            results_markdown += f"❌ **An unexpected error occurred while checking `{title}`.** Please check the logs.\\n\\n"
    results_markdown += "---\\n\\nAudit complete. Based on these results, you may want to proceed with cleansing and categorization."
    logger.info("Data quality audit finished.")
    return results_markdown


def reset_all_categorizations(confirm: bool = False) -> str:
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

def execute_custom_query(query: str) -> str:
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
                response = f"✅ **Query Successful**:\\n\\n{df.to_markdown(index=False)}"
                return response
        else: 
            affected_rows = results.num_dml_affected_rows or 0
            return f"✅ **Query Successful**: The operation completed and affected **{affected_rows}** rows."

    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during custom query execution: {e}")
        return f"❌ **Error During Query Execution**\\nA BigQuery error occurred. Please check the syntax of your query and the application logs. Error: {e}"
    except Exception as e:
        logger.error(f"❌ Unexpected error during custom query execution: {e}")
        return f"❌ **Unexpected Error**\\nAn unexpected error occurred. Please check the logs. Error: {e}"


# --- Phase 1: Dynamic Rules-Based Tools ---

def run_cleansing_and_dynamic_rules() -> str:
    """
    Performs data cleansing and applies categorization rules dynamically
    from the dedicated rules table in BigQuery. This enhanced version uses a two-pass
    approach, first matching on merchant, then on description patterns for any remainder.
    """
    logger.info("Starting enhanced data cleansing and DYNAMIC rules-based categorization...")

    sql_cleansing_and_rules = f"""
    MERGE `{TABLE_ID}` AS T
    USING (
        WITH
        -- Step 1: Clean the raw description and merchant name for all uncategorized transactions.
        CleansedData AS (
            SELECT
                transaction_id,
                transaction_type,
                TRIM(LOWER(REGEXP_REPLACE(REGEXP_REPLACE(REPLACE(IFNULL(description_raw, ''), '-', ''), r'[^a-zA-Z0-9\\\\s]', ' '), r'\\\\s+', ' '))) AS new_description_cleaned,
                TRIM(LOWER(
                    REGEXP_EXTRACT(
                        REGEXP_REPLACE(REGEXP_REPLACE(REPLACE(IFNULL(merchant_name_raw, ''), '-', ''), r'[^a-zA-Z0-9\\\\s\\\\*#-]', ''), r'\\\\s+', ' '),
                        r'^(?:sq\\\\s*\\\\*|pypl\\\\s*\\\\*|cl\\\\s*\\\\*|\\\\*\\\\s*)?([^*#-]+)'
                    )
                )) AS new_merchant_name_cleaned
            FROM `{TABLE_ID}`
            WHERE category_l1 IS NULL
        ),
        -- Step 2: Get all rules, deduplicated by identifier, type, and transaction_type to get the best one.
        DeduplicatedRules AS (
            SELECT
                *,
                ROW_NUMBER() OVER(PARTITION BY identifier, rule_type, transaction_type ORDER BY confidence_score DESC, last_updated_timestamp DESC) as rn
            FROM `{RULES_TABLE_ID}`
            WHERE is_active = TRUE -- Only use active rules
        ),
        -- Step 3: First pass categorization based on MERCHANT rules.
        CategorizedByMerchant AS (
            SELECT
                cd.transaction_id,
                cd.new_description_cleaned,
                cd.new_merchant_name_cleaned,
                rules.category_l1,
                rules.category_l2,
                'rules-based-merchant' AS method
            FROM CleansedData cd
            JOIN DeduplicatedRules rules
                ON cd.new_merchant_name_cleaned = rules.identifier
                AND cd.transaction_type = rules.transaction_type
            WHERE rules.rn = 1 AND rules.rule_type = 'MERCHANT'
        ),
        -- Step 4: Second pass categorization based on DESCRIPTION rules for remaining transactions.
        CategorizedByDescription AS (
            SELECT
                cd.transaction_id,
                cd.new_description_cleaned,
                cd.new_merchant_name_cleaned,
                rules.category_l1,
                rules.category_l2,
                'rules-based-description' AS method
            FROM CleansedData cd
            LEFT JOIN CategorizedByMerchant cbm ON cd.transaction_id = cbm.transaction_id
            JOIN DeduplicatedRules rules
                ON STRPOS(cd.new_description_cleaned, rules.identifier) > 0 -- Use STRPOS for 'contains' logic
                AND cd.transaction_type = rules.transaction_type
            WHERE cbm.transaction_id IS NULL -- Only process those not caught by the merchant pass
              AND rules.rn = 1
              AND rules.rule_type = 'DESCRIPTION'
        )
        -- Step 5: Combine results from both passes
        SELECT * FROM CategorizedByMerchant
        UNION ALL
        SELECT * FROM CategorizedByDescription

    ) AS U
    ON T.transaction_id = U.transaction_id
    WHEN MATCHED THEN
        UPDATE SET
            T.description_cleaned = U.new_description_cleaned,
            T.merchant_name_cleaned = U.new_merchant_name_cleaned,
            T.category_l1 = U.category_l1,
            T.category_l2 = U.category_l2,
            T.categorization_method = U.method,
            T.categorization_update_timestamp = CURRENT_TIMESTAMP();
    """

    try:
        query_job = bq_client.query(sql_cleansing_and_rules)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows or 0
        logger.info("✅ Successfully ran enhanced cleansing and dynamic rules engine. %d rows affected.", affected_rows)
        return f"⚙️ **Cleansing & Dynamic Rules Engine Complete**\n\nSuccessfully processed the data. A total of **{affected_rows}** rows were cleansed and categorized based on the enhanced dynamic rules table (merchant and description)."
    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during enhanced cleansing and dynamic rules phase: {e}")
        return f"❌ **Error During Cleansing & Rules**\nA BigQuery error occurred. Please check the logs. Error: {e}"


from google.adk.agents import LlmAgent

def review_and_resolve_rule_conflicts() -> str:
    """
    Reviews the categorization rules for conflicts, uses an LLM to resolve them,
    and updates the rules table to deactivate incorrect entries.
    """
    logger.info("Starting rule review and conflict resolution...")

    # Step 1: Ensure 'is_active' column exists and is populated.
    try:
        # Attempt to add the column. This will fail if it already exists.
        add_column_query = f"ALTER TABLE `{RULES_TABLE_ID}` ADD COLUMN is_active BOOL;"
        bq_client.query(add_column_query).result()
        logger.info("Successfully added 'is_active' column.")
    except GoogleAPICallError as e:
        # Ignore the error if the column already exists, otherwise return an error.
        if "Duplicate column name is_active" in str(e):
            logger.info("'is_active' column already exists. No action needed.")
        else:
            logger.error(f"Error adding 'is_active' column: {e}")
            return f"❌ Error preparing rules table for review: {e}"

    try:
        # Backfill any NULL values for the is_active flag. This makes the operation idempotent.
        update_query = f"UPDATE `{RULES_TABLE_ID}` SET is_active = TRUE WHERE is_active IS NULL;"
        update_job = bq_client.query(update_query)
        update_job.result()
        if (update_job.num_dml_affected_rows or 0) > 0:
             logger.info(f"Backfilled 'is_active' flag for {update_job.num_dml_affected_rows} rules.")
        else:
            logger.info("'is_active' column is fully populated.")
    except GoogleAPICallError as e:
        logger.error(f"Error populating 'is_active' column: {e}")
        return f"❌ Error populating 'is_active' column: {e}"

    # Step 2: Find conflicting rules (same identifier, different categories).
    conflict_query = f"""
        WITH ConflictingRules AS (
            SELECT
                identifier,
                rule_type,
                transaction_type,
                ARRAY_AGG(STRUCT(rule_id, category_l1, category_l2, confidence_score, last_updated_timestamp)) AS rules
            FROM `{RULES_TABLE_ID}`
            WHERE is_active = TRUE
            GROUP BY 1, 2, 3
            HAVING COUNT(DISTINCT (category_l1, category_l2)) > 1
        )
        SELECT identifier, rule_type, transaction_type, rules FROM ConflictingRules
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
        You will be given a JSON object representing a set of conflicting rules for a single identifier.
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
            return f"🔄 **Harmonization Complete**\\n\\nI've successfully standardized the categories for **{affected_rows}** recurring transactions, ensuring consistency."
        else:
            return "✅ **Harmonization Complete**\\n\\nNo recurring transactions needed harmonization at this time."

    except GoogleAPICallError as e:
        logger.error(f"❌ BigQuery error during harmonization phase: {e}")
        return f"❌ **Error During Harmonization**\\nA BigQuery error occurred. Please check the logs. Error: {e}"


# --- Phase 2 & 3: AI-Based Bulk & Recurring Tools ---

def get_recurring_candidates_batch(tool_context: ToolContext, batch_size: int = 15) -> str:
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
                ) ORDER BY transaction_date DESC LIMIT 5
            ) AS example_transactions
        FROM TransactionIntervals
        GROUP BY 1, 2
        HAVING COUNT(transaction_id) >= 3
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


# --- Phase 4: Transaction-Level AI & Learning Tools ---
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

        category_summary = validated_df.groupby(['category_l1', 'category_l2']).size().reset_index(name='count').to_dict('records')

        return json.dumps({
            "status": "success",
            "updated_count": updated_count,
            "summary": category_summary
        })
    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during update: {e}")
        return json.dumps({"status": "error", "message": f"Failed to update BigQuery: {e}"})

def harvest_new_rules() -> str:
    """
    Identifies high-confidence categories from AI processing and saves them as new rules
    for future use, making the agent smarter over time.
    """
    logger.info("Harvesting new rules from AI categorizations...")
    
    harvest_sql = f"""
    MERGE `{RULES_TABLE_ID}` R
    USING (
        -- Find merchants consistently categorized by the AI
        SELECT
            merchant_name_cleaned,
            transaction_type,
            category_l1,
            category_l2,
            COUNT(transaction_id) AS confidence_score
        FROM `{TABLE_ID}`
        WHERE categorization_method IN ('llm-bulk-merchant-based', 'llm-transaction-based')
            AND merchant_name_cleaned IS NOT NULL
        GROUP BY 1, 2, 3, 4
        -- Only create rules for high-confidence patterns (e.g., seen at least 5 times)
        HAVING COUNT(transaction_id) >= 5
    ) AS NewRules
    ON R.identifier = NewRules.merchant_name_cleaned
        AND R.rule_type = 'MERCHANT'
        AND R.transaction_type = NewRules.transaction_type
    -- If a rule for this merchant already exists, update it if the new one is more confident
    WHEN MATCHED AND NewRules.confidence_score > R.confidence_score THEN
        UPDATE SET
            category_l1 = NewRules.category_l1,
            category_l2 = NewRules.category_l2,
            confidence_score = NewRules.confidence_score,
            last_updated_timestamp = CURRENT_TIMESTAMP()
    -- If no rule exists, insert a new one
    WHEN NOT MATCHED THEN
        INSERT (rule_id, rule_type, identifier, transaction_type, category_l1, category_l2, confidence_score, created_timestamp, last_updated_timestamp)
        VALUES (
            GENERATE_UUID(),
            'MERCHANT',
            NewRules.merchant_name_cleaned,
            NewRules.transaction_type,
            NewRules.category_l1,
            NewRules.category_l2,
            NewRules.confidence_score,
            CURRENT_TIMESTAMP(),
            CURRENT_TIMESTAMP()
        );
    """
    try:
        harvest_job = bq_client.query(harvest_sql)
        harvest_job.result()
        new_rules_count = harvest_job.num_dml_affected_rows or 0
        if new_rules_count > 0:
            logger.info("✅ Successfully harvested %d new rules.", new_rules_count)
            return f"🧠 **Learning Complete**: I've analyzed the recent AI categorizations and created or updated **{new_rules_count}** new high-confidence rules. The agent is now smarter for the next run!"
        else:
            logger.info("No new high-confidence rules to harvest.")
            return "🧠 **Learning Complete**: No new patterns met the confidence threshold to be saved as rules at this time."

    except (GoogleAPICallError, Exception) as e:
        logger.error(f"❌ BigQuery error during rule harvesting: {e}")
        return f"❌ **Error During Learning**: An error occurred while trying to save new rules. Please check the logs. Error: {e}"