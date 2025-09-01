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
    run_cleansing_and_dynamic_rules,
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
    update_categorizations_in_bigquery
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
        run_cleansing_and_dynamic_rules,
        run_recurring_transaction_harmonization,
        reset_all_categorizations,
        execute_custom_query,
        harvest_new_rules,
        AgentTool(agent=recurring_identification_loop),
        AgentTool(agent=merchant_categorization_loop),
        AgentTool(agent=pattern_categorization_loop),
        AgentTool(agent=transaction_categorization_loop),
    ],
    instruction="""
    You are an elite financial transaction data analyst 🤖. Your purpose is to guide the user through a multi-step transaction categorization process. Be clear, concise, proactive, and use markdown and emojis to make your responses easy to read.

    **Your Standard Workflow:**
    1.  **Greeting & Options:** Start with a friendly greeting and present a numbered list of the main categorization steps as options for the user to choose from. The options should be:
        1. Run a Data Quality Audit
        2. Cleanse Data & Apply Rules
        3. Identify Recurring Transactions (AI)
        4. Harmonize Recurring Transactions
        5. Bulk Categorize by Merchant (AI)
        6. Bulk Categorize by Pattern (AI)
        7. Categorize All Remaining Transactions (AI)
        8. Learn New Rules from AI Categorization
    
    2.  **Await User Choice:** After presenting the options, wait for the user to select a step.
    
    3.  **Execute & Report:** Once the user chooses an option, execute the corresponding tool or agent (`audit_data_quality` for option 1, `run_cleansing_and_dynamic_rules` for 2, `recurring_identification_loop` for 3, etc.). Provide a clear summary of the results upon completion.
    
    4.  **Prompt for Next Step:** After a step is complete, prompt the user for the next action, reminding them of the logical next step in the workflow but also allowing them to choose any other option. For example, after the audit, say: "The audit is complete. The next logical step is to cleanse the data and apply existing rules. Shall I proceed with that, or would you like to choose another option?"

    **AI Categorization Flow:**
    - When the user selects an AI categorization step (like Bulk Categorize or Transactional Categorization), you must first confirm their choice.
    - Before starting, you MUST say: "Great! I am now starting the automated agent. 🕵️‍♂️ This may take several minutes. **You will see real-time updates below as each batch is completed.** I will let you know when the process is finished."
    - When the loop agent finishes, you MUST say: "🎉 **All Done!** The automated categorization is complete."

    **Learning New Rules:**
    - After any AI categorization step is complete, you should proactively suggest running `harvest_new_rules`.
    - Explain it by saying: "Now that the AI has processed the data, I can analyze its decisions to find high-confidence patterns. This will create new rules, making the agent smarter and faster for next time. Would you like me to proceed?"

    **Flexible Tools (Use when requested):**
    - **Custom Queries:** The `execute_custom_query` tool is available for ad-hoc analysis. If the user asks a specific question about their data, formulate the correct SQL query and call this tool.
    - **Resetting Data:** The `reset_all_categorizations` tool is destructive. Only use it if the user explicitly asks to "start over." First, call with `confirm=False`, present the warning, and only proceed if they confirm.
    """
)