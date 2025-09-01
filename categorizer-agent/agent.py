# agent.py

from __future__ import annotations
import logging

# ADK Core Components
from google.adk.agents import Agent, LlmAgent, LoopAgent
from google.adk.tools import AgentTool

# Import configurations and tools
from .config import VALID_CATEGORIES_JSON_STR
from .tools import (
    audit_data_quality,
    run_data_cleansing,
    apply_categorization_rules,
    add_rule_to_table,
    run_recurring_transaction_harmonization,
    reset_all_categorizations,
    execute_custom_query,
    harvest_new_rules,
    get_recurring_candidates_batch,
    apply_bulk_recurring_flags,
    get_merchant_batch_to_categorize,
    apply_bulk_merchant_update,
    get_pattern_batch_to_categorize,
    apply_bulk_pattern_update,
    fetch_batch_for_ai_categorization,
    update_categorizations_in_bigquery,
    review_and_resolve_rule_conflicts,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 5. Agent Definitions ---

# --- Loop Agents for Batch Processing ---
single_recurring_batch_agent = LlmAgent(
    name="single_recurring_batch_agent",
    model="gemini-2.5-flash",
    tools=[get_recurring_candidates_batch, apply_bulk_recurring_flags],
    instruction="""
    Your purpose is to perform one cycle of BATCH recurring transaction identification.
    1. **FETCH**: Call `get_recurring_candidates_batch` to get potential recurring merchants. Escalate if complete.
    2. **ANALYZE & UPDATE**: Analyze the JSON for ALL merchants. Decide if a merchant is recurring based on `transaction_intervals_days` (strong signal for patterns like `[30, 31, 29]`), `has_recurring_keywords`, and low `stddev_amount`. Then, call `apply_bulk_recurring_flags` ONCE with a JSON list of objects for merchants you are confident are recurring. Each object must have `merchant_name_cleaned`, `transaction_type`, and `is_recurring: true`.
    3. **REPORT**: The update tool returns `updated_count` and a `summary`. Create a markdown report, e.g., "🔍 Identified 'spotify' and 'netflix' as recurring, flagging 24 new transactions."
    """,
)

recurring_identification_loop = LoopAgent(
    name="recurring_identification_loop",
    description="This agent starts an AI-driven process to find and flag recurring transactions. It processes merchants in batches and provides real-time summaries.",
    sub_agents=[single_recurring_batch_agent],
    max_iterations=10
)

single_merchant_batch_agent = LlmAgent(
    name="single_merchant_batch_agent",
    model="gemini-2.5-flash",
    tools=[get_merchant_batch_to_categorize, apply_bulk_merchant_update],
    instruction=f"""
    Your purpose is to perform one cycle of BATCH merchant-based transaction categorization.
    1. **FETCH**: Call `get_merchant_batch_to_categorize`. Escalate if complete.
    2. **ANALYZE & UPDATE**: For ALL merchants in the batch, determine the correct `category_l1` and `category_l2` from: {VALID_CATEGORIES_JSON_STR}. Then, call `apply_bulk_merchant_update` ONCE with a single JSON string. Each merchant object MUST include `merchant_name_cleaned`, `transaction_type`, `category_l1`, and `category_l2`.
    3. **REPORT**: The tool returns `updated_count` and a `summary`. Create a markdown report, e.g., "🛒 Processed a batch of 3 merchants, updating 112 transactions. Key updates include 'grubhub' to Food & Dining."
    """,
)

merchant_categorization_loop = LoopAgent(
    name="merchant_categorization_loop",
    description="This agent starts an efficient, automated categorization by processing BATCHES of common uncategorized merchants, providing a summary for each batch.",
    sub_agents=[single_merchant_batch_agent],
    max_iterations=10
)

single_pattern_batch_agent = LlmAgent(
    name="single_pattern_batch_agent",
    model="gemini-2.5-flash",
    tools=[get_pattern_batch_to_categorize, apply_bulk_pattern_update],
    instruction=f"""
    Your purpose is to perform one complete cycle of BATCH pattern-based transaction categorization.

    **Your process is a strict, three-step sequence:**
    1.  **FETCH BATCH:** First, you MUST call `get_pattern_batch_to_categorize` to get a batch of up to 10 pattern groups.
        - If the tool returns a "complete" status, you must stop and escalate.
    2.  **ANALYZE & UPDATE BATCH:** Analyze the JSON data for ALL patterns. For each one, determine the correct `category_l1` and `category_l2` from the valid list: {VALID_CATEGORIES_JSON_STR}.
        Then, call `apply_bulk_pattern_update` ONCE. **Your output must be a single JSON array** that includes an entry for every pattern in the batch you received.
    3.  **REPORT BATCH:** The update tool returns `updated_count` and a `summary`. Use this to create a user-friendly markdown report. For example: "🧾 Processed a batch of 5 patterns, updating 88 transactions. This included patterns like 'payment thank you' being set to Credit Card Payment."
    """,
)

pattern_categorization_loop = LoopAgent(
    name="pattern_categorization_loop",
    description="This agent starts an advanced, batch-based categorization on common transaction description patterns, providing real-time summaries.",
    sub_agents=[single_pattern_batch_agent],
    max_iterations=10
)

single_transaction_categorizer_agent = LlmAgent(
    name="single_transaction_categorizer_agent",
    model="gemini-2.5-flash",
    tools=[fetch_batch_for_ai_categorization, update_categorizations_in_bigquery],
    instruction=f"""
    Your purpose is to perform one cycle of detailed, transaction-by-transaction categorization and report the result with enhanced detail.
    1.  **FETCH**: Call `fetch_batch_for_ai_categorization`. If it returns "complete", escalate immediately.
    2.  **CATEGORIZE & UPDATE**: Call `update_categorizations_in_bigquery` with a `categorized_json_string`. This string MUST be a JSON array of objects, each with `transaction_id`, `category_l1`, and `category_l2` from the valid list: {VALID_CATEGORIES_JSON_STR}.
    3.  **REPORT**: The update tool returns `updated_count` and a `summary`. Present this clearly in markdown. Example:
        "✅ Processed a batch of 198 transactions.
        - **Shopping**: 75 transactions
        - **Groceries**: 50 transactions
        Now moving to the next batch..."
    """,
)

transaction_categorization_loop = LoopAgent(
    name="transaction_categorization_loop",
    description="This agent starts the final, granular categorization. It automatically processes remaining transactions in batches, providing a detailed summary for each.",
    sub_agents=[single_transaction_categorizer_agent],
    max_iterations=50
)

# --- Root Orchestrator Agent ---
root_agent = Agent(
    name="transaction_categorizer_orchestrator",
    model="gemini-2.5-flash",
    tools=[
        audit_data_quality,
        reset_all_categorizations,
        execute_custom_query,
        review_and_resolve_rule_conflicts,
        run_data_cleansing,
        apply_categorization_rules,
        add_rule_to_table,
        run_recurring_transaction_harmonization,
        harvest_new_rules,
        AgentTool(agent=recurring_identification_loop),
        AgentTool(agent=merchant_categorization_loop),
        AgentTool(agent=pattern_categorization_loop),
        AgentTool(agent=transaction_categorization_loop),
    ],
    instruction="""
    You are an elite financial transaction data analyst 🤖. Your purpose is to help users categorize their financial transactions using a powerful and efficient workflow.

    **Primary Workflow:**
    1.  **Initial Greeting**: Unless the user asks a specific question, greet them with this exact message:
        "Hello! I'm ready to help you categorize your financial transactions using a powerful and efficient workflow. 🚀

        Please choose an option to begin:

        1.  📊 **Audit Data Quality**: Get a high-level overview and identify issues in your data.
        2.  ⚙️ **Run Categorization**: Cleanse, classify, and categorize your data using rules and AI.
        3.  🔎 **Conduct Custom Research**: Analyze transactional data using natural language.
        4.  ➕ **Create Custom Rule**: Create a custom transaction categorizaton rule.
        4.  🔄 **Reset All Categorizations**: Clear all data cleansing and category assignments."

    2.  **Executing User's Choice**:
        - If the user chooses **1 (Audit)**, call `audit_data_quality` and present the results.
        - If the user chooses **2 (Run Categorization)**, you must execute the full, end-to-end categorization workflow. This involves calling a sequence of tools and agents in a specific order. You MUST follow this order and report on the outcome of each step before proceeding to the next.
            - **Step 1: Rule Conflict Review:** Call `review_and_resolve_rule_conflicts`.
            - **Step 2: Data Cleansing:** Call `run_data_cleansing`.
            - **Step 3: Rules Application:** Call `apply_categorization_rules`.
            - **Step 4: AI Recurring Identification:** Call the `recurring_identification_loop` agent.
            - **Step 5: Harmonize Recurring:** Call `run_recurring_transaction_harmonization`.
            - **Step 6: AI Merchant Categorization:** Call the `merchant_categorization_loop` agent.
            - **Step 7: AI Pattern Categorization:** Call the `pattern_categorization_loop` agent.
            - **Step 8: AI Transaction-Level Categorization:** Call the `transaction_categorization_loop` agent.
            - **Step 9: Learn New Rules:** Call `harvest_new_rules` to learn from the AI's work.
            - After the final step, provide a concluding summary.
        - If the user chooses **3 (Reset)**, call `reset_all_categorizations`. You must first call with `confirm=False`, show the user the warning, and only proceed if they explicitly confirm.

    3.  **Handling Follow-up Questions**: After completing a task, ask the user what they would like to do next.

    **Custom Rule Creation:**
    - If the user asks to create a new rule (e.g., "set up a rule for medical expenses"), you must guide them through the process.
    - **Analyze the Request:** Determine the key information from the user's request.
    - **Gather Information:** Ask clarifying questions to get all the required parameters for the `add_rule_to_table` tool: `identifier`, `rule_type` ('MERCHANT' or 'PATTERN'), `category_l1`, `category_l2`, and `transaction_type` ('Debit', 'Credit', or 'All'). Also ask if it should be a recurring rule.
    - **Propose and Confirm:** Propose the complete rule to the user in a clear format. Example: "Great! I'm ready to create a rule. Does this look correct?\n\n- **Match On**: 'MEDICAL'\n- **Rule Type**: PATTERN\n- **Set Category To**: Expense / Medical\n- **For Transaction Type**: All\n- **Mark as Recurring**: No"
    - **Execute:** Once the user confirms, call the `add_rule_to_table` tool with the confirmed parameters.

    **Ad-hoc Queries:**
    - If the user asks a specific question about their data (e.g., "how many transactions from Starbucks?"), use the `execute_custom_query` tool to answer them.
    """
)