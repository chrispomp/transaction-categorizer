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

# --- New Categorization Flow Agent ---
# This new LoopAgent will orchestrate the entire categorization process end-to-end.

# Define simple agents to wrap the tool calls
rule_review_agent = Agent(
    name="rule_review_agent",
    model="gemini-2.5-flash",
    tools=[review_and_resolve_rule_conflicts],
    instruction="Call the `review_and_resolve_rule_conflicts` tool and report its output.",
)

cleansing_agent = Agent(
    name="cleansing_agent",
    model="gemini-2.5-flash",
    tools=[run_cleansing_and_dynamic_rules],
    instruction="Call the `run_cleansing_and_dynamic_rules` tool and report its output.",
)

harmonization_agent = Agent(
    name="harmonization_agent",
    model="gemini-2.5-flash",
    tools=[run_recurring_transaction_harmonization],
    instruction="Call the `run_recurring_transaction_harmonization` tool and report its output.",
)

rule_harvester_agent = Agent(
    name="rule_harvester_agent",
    model="gemini-2.5-flash",
    tools=[harvest_new_rules],
    instruction="Your job is to call the `harvest_new_rules` tool to learn from the AI's recent work. Report the result back to the user.",
)

categorization_flow_loop = LoopAgent(
    name="categorization_flow_loop",
    model="gemini-2.5-flash",
    description="Runs the full, end-to-end categorization process. This includes data cleansing, rule-based categorization, AI-driven categorization, and learning new rules.",
    sub_agents=[
        AgentTool(agent=rule_review_agent),
        AgentTool(agent=cleansing_agent),
        AgentTool(agent=recurring_identification_loop),
        AgentTool(agent=harmonization_agent),
        AgentTool(agent=merchant_categorization_loop),
        AgentTool(agent=pattern_categorization_loop),
        AgentTool(agent=transaction_categorization_loop),
        AgentTool(agent=rule_harvester_agent),
    ],
    max_iterations=1, # This loop runs once to execute the sequence of sub-agents.
    instruction="""
    You are orchestrating a multi-step financial transaction categorization process. Execute the following agents in this exact order and do not proceed to the next step until the current one is complete. Report on the outcome of each step.

    **Workflow:**
    1.  **`rule_review_agent`**: Resolve any conflicts in the existing rule set.
    2.  **`cleansing_agent`**: Cleanse the data and apply all existing high-confidence rules.
    3.  **`recurring_identification_loop`**: Identify and flag recurring transactions.
    4.  **`harmonization_agent`**: Ensure recurring transactions from the same merchant have the same category.
    5.  **`merchant_categorization_loop`**: Use AI to categorize transactions in bulk based on merchant names.
    6.  **`pattern_categorization_loop`**: Use AI to categorize transactions in bulk based on description patterns.
    7.  **`transaction_categorization_loop`**: Use AI to categorize any remaining individual transactions.
    8.  **`rule_harvester_agent`**: Analyze the AI's work to create new rules for the future.

    After the final step is complete, provide a concluding summary to the user.
    """
)

# --- Root Orchestrator Agent ---
root_agent = Agent(
    name="transaction_categorizer_orchestrator",
    model="gemini-2.5-flash",
    tools=[
        audit_data_quality,
        reset_all_categorizations,
        execute_custom_query,
        AgentTool(agent=categorization_flow_loop),
    ],
    instruction="""
    You are an elite financial transaction data analyst 🤖. Your purpose is to help users categorize their financial transactions using a powerful and efficient workflow.

    **Primary Workflow:**
    1.  **Initial Greeting**: Unless the user asks a specific question, greet them with this exact message:
        "Hello! I'm ready to help you categorize your financial transactions using a powerful and efficient workflow. 🚀

        Please choose an option to begin:

        1.  📊 **Audit Data Quality**: Get a high-level overview and identify issues in your data.
        2.  ⚙️ **Run Categorization**: Cleanse, classify, and categorize your data using rules and AI.
        3.  🔄 **Reset All Categorizations**: Clear all data cleansing and category assignments."

    2.  **Executing User's Choice**:
        - If the user chooses **1 (Audit)**, call the `audit_data_quality` tool and present the results.
        - If the user chooses **2 (Run Categorization)**, call the `categorization_flow_loop` agent. Provide a brief intro before starting and a confirmation when it's complete.
        - If the user chooses **3 (Reset)**, call the `reset_all_categorizations` tool. Since this is a destructive action, you must first call it with `confirm=False`, show the user the warning, and only proceed if they explicitly confirm.

    3.  **Handling Follow-up Questions**: After completing a task, ask the user what they would like to do next.

    **Ad-hoc Queries:**
    - If the user asks a specific question about their data (e.g., "how many transactions from Starbucks?"), use the `execute_custom_query` tool to answer them.
    """
)