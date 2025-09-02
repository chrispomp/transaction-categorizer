# agent.py

from __future__ import annotations
import logging

# ADK Core Components
from google.adk.agents import Agent, LlmAgent, LoopAgent
from google.adk.tools import AgentTool

# Import configurations and tools
from .config import VALID_CATEGORIES_JSON_STR, PROJECT_ID, DATASET_ID, TABLE_ID, RULES_TABLE_ID
from .tools import (
    get_data_quality_report,
    prepare_transaction_data,
    apply_rules_based_categorization,
    create_new_categorization_rule,
    harmonize_recurring_transaction_categories,
    reset_all_transaction_categorizations,
    run_custom_query,
    learn_new_categorization_rules,
    get_recurring_transaction_candidates,
    flag_recurring_transactions_in_bulk,
    get_uncategorized_merchants_batch,
    apply_bulk_merchant_update,
    get_uncategorized_patterns_batch,
    apply_bulk_pattern_update,
    get_transaction_batch_for_ai_categorization,
    update_transactions_with_ai_categories,
    review_and_resolve_rule_conflicts,
    apply_fallback_categorization,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 5. Agent Definitions ---

# --- Loop Agents for Batch Processing ---
recurring_transaction_identifier_agent = LlmAgent(
    name="recurring_transaction_identifier_agent",
    model="gemini-2.5-flash",
    tools=[get_recurring_transaction_candidates, flag_recurring_transactions_in_bulk],
    instruction="""
    Your purpose is to perform one cycle of BATCH recurring transaction identification.
    1.  **FETCH**: Call `get_recurring_transaction_candidates` to get potential recurring merchants. Escalate if complete.
    2.  **ANALYZE & UPDATE**: Analyze the JSON for ALL merchants. Decide if a merchant is recurring based on `transaction_intervals_days` (strong signal for patterns like `[30, 31, 29]`), `has_recurring_keywords`, and low `stddev_amount`. Then, call `flag_recurring_transactions_in_bulk` ONCE with a JSON list of objects for merchants you are confident are recurring. Each object must have `merchant_name_cleaned`, `transaction_type`, and `is_recurring: true`.
    3.  **REPORT**: The update tool returns `updated_count` and a `summary`. Create a visually appealing markdown report. Example: "**🔍 Recurring Transactions Identified**\n\nI found and flagged **24** new recurring transactions for merchants like 'spotify' and 'netflix'. This helps in better tracking of subscriptions and regular bills."
    """,
)

recurring_transaction_identification_workflow = LoopAgent(
    name="recurring_transaction_identification_workflow",
    description="This agent starts an AI-driven process to find and flag recurring transactions. It processes merchants in batches and provides real-time summaries.",
    sub_agents=[recurring_transaction_identifier_agent],
    max_iterations=20
)

merchant_categorization_agent = LlmAgent(
    name="merchant_categorization_agent",
    model="gemini-2.5-flash",
    tools=[get_uncategorized_merchants_batch, apply_bulk_merchant_update],
    instruction=f"""
    Your purpose is to perform one cycle of BATCH merchant-based transaction categorization. You are an expert at analyzing transaction data and assigning precise categories.

    **Here are some examples of excellent categorization:**

    * **Input Merchant**: 'starbucks'
        * **Examples**: [{{"description_cleaned": "starbucks corp", "amount": -5.75}}, {{"description_cleaned": "starbucks 123 main st", "amount": -12.40}}]
        * **Correct Output**: {{"merchant_name_cleaned": "starbucks", "transaction_type": "Debit", "persona_type": "Salaried Tech Professional", "category_l1": "Expense", "category_l2": "Food & Dining"}}

    * **Input Merchant**: 'amzn mktp us'
        * **Examples**: [{{"description_cleaned": "amzn mktp us item 123", "amount": -25.99}}, {{"description_cleaned": "amazon marketplace", "amount": -9.99}}]
        * **Correct Output**: {{"merchant_name_cleaned": "amzn mktp us", "transaction_type": "Debit", "persona_type": "Salaried Tech Professional", "category_l1": "Expense", "category_l2": "Shopping"}}

    * **Input Merchant**: 'adp payroll'
        * **Examples**: [{{"description_cleaned": "direct deposit adp", "amount": 2500.00}}]
        * **Correct Output**: {{"merchant_name_cleaned": "adp payroll", "transaction_type": "Credit", "persona_type": "Salaried Tech Professional", "category_l1": "Income", "category_l2": "Payroll"}}

    **Your process is a strict, two-step sequence:**
    1.  **FETCH BATCH**: Call `get_uncategorized_merchants_batch`. If the tool returns a "complete" status, you must stop and escalate.
    2.  **ANALYZE & UPDATE BATCH**: Analyze the JSON data for ALL merchants in the batch.
        - **CRITICAL**: You **MUST ONLY** use `category_l1` and `category_l2` from the valid list below. Do not invent, create, or use any category not explicitly provided.
            - **`category_l1`: "Income"**
                - `category_l2`: "Gig Income", "Payroll", "Other Income", "Refund"
            - **`category_l1`: "Expense"**
                - `category_l2`: "Groceries", "Food & Dining", "Shopping", "Entertainment", "Health & Wellness", "Auto & Transport", "Travel & Vacation", "Software & Tech", "Medical", "Insurance", "Bills & Utilities", "Fees & Charges", "Business Services", "Other Expense", "Loan Payment"
            - **`category_l1`: "Transfer"**
                - `category_l2`: "Credit Card Payment", "Internal Transfer", "ATM Withdrawal"
        - Then, call `apply_bulk_merchant_update` ONCE with a single JSON array. Each merchant object MUST include `merchant_name_cleaned`, `transaction_type`, `persona_type`, `category_l1`, and `category_l2`.
    3.  **REPORT BATCH**: The tool returns `updated_count` and a `summary`. Create an insightful, data-driven markdown report. Example: "**🛒 Merchant Batch Categorized**\n\nI processed a batch of **3** merchants, updating **112** transactions. Key updates include:\n- 'grubhub' -> Food & Dining\n- 'shell' -> Auto & Transport"
    """,
)

merchant_categorization_workflow = LoopAgent(
    name="merchant_categorization_workflow",
    description="This agent starts an efficient, automated categorization by processing BATCHES of common uncategorized merchants, providing a summary for each batch.",
    sub_agents=[merchant_categorization_agent],
    max_iterations=20
)

pattern_categorization_agent = LlmAgent(
    name="pattern_categorization_agent",
    model="gemini-2.5-flash",
    tools=[get_uncategorized_patterns_batch, apply_bulk_pattern_update],
    instruction=f"""
    You are a meticulous data analyst specializing in identifying and categorizing financial transaction patterns. Your purpose is to process one batch of transaction patterns accurately and efficiently.

    **Your process is a strict, sequential workflow:**

    ### 1. Fetch Batch of Patterns
    * You **MUST** begin by calling the `get_uncategorized_patterns_batch` tool. This will provide a batch of up to 20 pattern groups.
    * **Completion Check**: If the tool returns a "complete" status, your work is done. You must stop and escalate immediately.
    * **Empty Batch Check**: If the tool returns an empty list, report that there were no patterns to process and stop.

    ### 2. Analyze and Categorize
    * For each pattern group in the batch, you must determine the correct `category_l1` and `category_l2`.
    * **CRITICAL**: Your categorization choices are strictly limited to the following structure. Any deviation will result in a tool error.

        * **`category_l1`: "Income"**
            * `category_l2`: "Gig Income", "Payroll", "Other Income", "Refund"
        * **`category_l1`: "Expense"**
            * `category_l2`: "Groceries", "Food & Dining", "Shopping", "Entertainment", "Health & Wellness", "Auto & Transport", "Travel & Vacation", "Software & Tech", "Medical", "Insurance", "Bills & Utilities", "Fees & Charges", "Business Services", "Other Expense", "Loan Payment"
        * **`category_l1`: "Transfer"**
            * `category_l2`: "Credit Card Payment", "Internal Transfer", "ATM Withdrawal"
    * Use the `persona_type` to inform your categorization. For example, a "Full-Time Rideshare Driver" might have income from "UBER" or "LYFT" that should be categorized as "Gig Income", not "Payroll".

    ### 3. Apply Bulk Update
    * After analyzing **ALL** patterns in the batch, you **MUST** call the `apply_bulk_pattern_update` tool **only once**.
    * Your input to this tool must be a single JSON array containing an entry for every pattern you received in the batch. Each object in the array MUST include `description_prefix`, `transaction_type`, `channel`, `persona_type`, `category_l1`, and `category_l2`.

    ### 4. Report on Batch
    * The `apply_bulk_pattern_update` tool will return an `updated_count` and a `summary`.
    * Use this information to create a clear, user-friendly, and insightful markdown report.
    * **Example Report:**
        ```markdown
        ### Pattern Categorization Batch Report
        * **Status:** 🧾 Success
        * **Patterns Processed:** 5
        * **Transactions Updated:** 88
        * **Key Insights:** Applied categories to common patterns like 'payment thank you' (Credit Card Payment) and 'uber trip' (Auto & Transport), improving categorization accuracy for many transactions at once.
        ```
    """,
)

pattern_categorization_workflow = LoopAgent(
    name="pattern_categorization_workflow",
    description="This agent starts an advanced, batch-based categorization on common transaction description patterns, providing real-time summaries.",
    sub_agents=[pattern_categorization_agent],
    max_iterations=20
)

# --- OPTIMIZATION: Reusable Transaction Categorizer Agent ---
# This single agent is now smart enough to handle either 'Credit' or 'Debit'
# transactions based on the initial prompt it receives from its parent controller.
individual_transaction_categorizer_agent = LlmAgent(
    name="individual_transaction_categorizer_agent",
    model="gemini-2.5-flash",
    tools=[get_transaction_batch_for_ai_categorization, update_transactions_with_ai_categories],
    instruction=f"""
    You are an expert financial analyst. Your goal is to categorize a batch of transactions.
    
    **Your process is determined by the initial prompt you receive:**
    1.  **Determine Transaction Type:** The initial prompt will tell you to process either 'Credit' or 'Debit' transactions. You must use this to inform your next step.
    2.  **Fetch Batch:** Call the `get_transaction_batch_for_ai_ categorization` tool, passing the correct `transaction_type` ('Credit' or 'Debit') that you determined in the previous step.
    3.  **Analyze & Categorize:**
        * If the tool returns "complete", escalate immediately.
        * For each transaction, assign `category_l1` and `category_l2` based **strictly** on the list below.
        * **CRITICAL**: You **MUST ONLY** use `category_l1` and `category_l2` from this valid list. Do not invent, create, or use any category not explicitly provided.
            - **`category_l1`: "Income"**
                - `category_l2`: "Gig Income", "Payroll", "Other Income", "Refund"
            - **`category_l1`: "Expense"**
                - `category_l2`: "Groceries", "Food & Dining", "Shopping", "Entertainment", "Health & Wellness", "Auto & Transport", "Travel & Vacation", "Software & Tech", "Medical", "Insurance", "Bills & Utilities", "Fees & Charges", "Business Services", "Other Expense", "Loan Payment"
            - **`category_l1`: "Transfer"**
                - `category_l2`: "Credit Card Payment", "Internal Transfer", "ATM Withdrawal"
        * Use the `persona_type` to inform your categorization. For example, a "Full-Time Rideshare Driver" might have income from "UBER" or "LYFT" that should be categorized as "Gig Income", not "Payroll".
    4.  **Update Batch:** Call `update_transactions_with_ai_categories` **once** with all your categorizations for the batch.
    5.  **Report Summary:** Use the `summary` from the update tool to create a data-driven and visually appealing markdown report. Include the number of transactions categorized and a breakdown of the top categories. Example: '**Transaction Batch Processed**\n\nSuccessfully categorized **150** debit transactions. Top categories assigned:\n- Shopping: 45 transactions\n- Food & Dining: 32 transactions\n- Bills & Utilities: 25 transactions'
    """,
)

# --- A Loop to run the Categorizer ---
# This loop simply runs the categorizer agent repeatedly. The logic of what
# type of transaction to process is now handled by the agent's instructions.
individual_transaction_categorization_workflow = LoopAgent(
    name="individual_transaction_categorization_workflow",
    description="This is a worker agent that processes batches of transactions of a specific type (credit or debit).",
    sub_agents=[individual_transaction_categorizer_agent],
    max_iterations=20 # High iteration count to process all transactions of a given type.
)

# --- Controller Agent to Manage Credit/Debit Runs ---
txn_categorization_controller = LlmAgent(
    name="txn_categorization_controller",
    model="gemini-2.5-flash",
    tools=[AgentTool(agent=individual_transaction_categorization_workflow)],
    instruction="""
    You are a workflow orchestrator. Your sole responsibility is to categorize all remaining credit and debit transactions by running the `individual_transaction_categorization_workflow` agent in two distinct phases.

    **You MUST follow these steps in this exact order:**

    1.  **Credit Phase:** First, you MUST call the `individual_transaction_categorization_workflow` tool with the following input: "Process all credit transactions."
    2.  **Debit Phase:** After the first tool call is fully complete, you MUST call the `individual_transaction_categorization_workflow` tool a second time with the following input: "Process all debit transactions."

    Do not stop until both phases are complete. Report back a simple confirmation message when the entire process is finished.
    """
)

# --- Root Orchestrator Agent ---
financial_transaction_categorizer = Agent(
    name="financial_transaction_categorizer",
    model="gemini-2.5-flash",
    tools=[
        get_data_quality_report,
        reset_all_transaction_categorizations,
        run_custom_query,
        review_and_resolve_rule_conflicts,
        prepare_transaction_data,
        apply_rules_based_categorization,
        create_new_categorization_rule,
        harmonize_recurring_transaction_categories,
        learn_new_categorization_rules,
        apply_fallback_categorization,
        AgentTool(agent=recurring_transaction_identification_workflow),
        AgentTool(agent=merchant_categorization_workflow),
        AgentTool(agent=pattern_categorization_workflow),
        # OPTIMIZATION: The root agent now calls the controller, not the loop directly.
        AgentTool(agent=txn_categorization_controller),
    ],
    instruction=f"""
    You are an elite financial transaction data analyst 🤖. Your purpose is to help users categorize their financial transactions using a powerful and efficient workflow.

    **Primary Workflow:**
    1.  **Initial Greeting**: Unless the user asks a specific question, greet them with this exact message:
        "Hello! I'm ready to help you categorize your financial transactions using a powerful and efficient workflow. 🚀

        Please choose an option to begin:

        1.  📊 **Audit Data Quality**: Get a high-level overview and identify issues in your data.
        2.  ⚙️ **Run Full Categorization**: prepare and categorize your data using rules and AI.
        3.  🔁 **Analyze Recurring Transactions**: Identify recurring transactions like subscriptions and bills.
        4.  🔎 **Conduct Custom Research**: Analyze transactional data using natural language.
        5.  ➕ **Create Custom Rule**: Create a custom transaction categorizaton rule.
        6.  🔄 **Reset All Categorizations**: Clear all data cleansing and category assignments."

    2.  **Executing User's Choice**:
        - If the user chooses **1 (Audit)**, call `get_data_quality_report` and present the results.
        - If the user chooses **2 (Run Full Categorization)**, you must execute the main categorization workflow. You MUST follow this order and report on the outcome of each step before proceeding to the next.
            - **Step 1: Rule Conflict Review:** Call `review_and_resolve_rule_conflicts`.
            - **Step 2: Data Cleansing:** Call `prepare_transaction_data`.
            - **Step 3: Rules Application:** Call `apply_rules_based_categorization`.
            - **Step 4: AI Merchant Categorization:** Call the `merchant_categorization_workflow` agent.
            - **Step 5: AI Pattern Categorization:** Call the `pattern_categorization_workflow` agent.
            - **Step 6: AI Transaction-Level Categorization:** Call the `txn_categorization_controller` agent. This controller will manage the final categorization, processing all credits first, then all debits.
            - **Step 7: Learn New Rules:** Call `learn_new_categorization_rules` to learn from the AI's work.
            - **Step 8: Apply Fallback Categories:** Call `apply_fallback_categorization` to ensure all transactions are categorized.
            - After the final step, provide a concluding summary.
        - If the user chooses **3 (Analyze Recurring Transactions)**, you must execute the dedicated recurring analysis workflow. You MUST follow this order and report on the outcome of each step.
            - **Step 1: AI Recurring Identification:** Call the `recurring_transaction_identification_workflow` agent.
            - **Step 2: Harmonize Recurring:** Call `harmonize_recurring_transaction_categories`.
            - After the final step, provide a concluding summary.
        - If the user chooses **6 (Reset)**, call `reset_all_transaction_categorizations`. You must first call with `confirm=False`, show the user the warning, and only proceed if they explicitly confirm.

    3.  **Handling Follow-up Questions**: After completing a task, ask the user what they would like to do next.

    **Custom Rule Creation:**
    - If the user asks to create a new rule (e.g., "set up a rule for medical expenses"), you must guide them through the process.
    - **Analyze the Request:** Determine the key information from the user's request.
    - **Gather Information:** Ask clarifying questions to get all the required parameters for the `create_new_categorization_rule` tool: `identifier`, `rule_type` ('MERCHANT' or 'PATTERN'), `category_l1`, `category_l2`, `transaction_type` ('Debit', 'Credit', or 'All'), and `persona_type` (optional, for specific personas). Also ask if it should be a recurring rule.
    - **Propose and Confirm:** Propose the complete rule to the user in a clear format. Example: "Great! I'm ready to create a rule. Does this look correct?\\n\\n- **Match On**: 'MEDICAL'\\n- **Rule Type**: PATTERN\\n- **Set Category To**: Expense / Medical\\n- **For Transaction Type**: All\\n- **For Persona Type**: Global (applies to all)\\n- **Mark as Recurring**: No"
    - **Execute:** Once the user confirms, call the `create_new_categorization_rule` tool with the confirmed parameters.

    **Ad-hoc Queries:**
    - If the user asks a specific question about their data (e.g., "how many transactions from Starbucks?"), use the `run_custom_query` tool to answer them.
    - To do this effectively, you must understand the available data schemas. The following tables are available within the BigQuery dataset `{PROJECT_ID}.{DATASET_ID}`:

        - **`{TABLE_ID}` (transactions)**: Contains the financial transaction data.
          - **Key Columns**:
            - `transaction_id` (STRING): Unique identifier for each transaction.
            - `transaction_date` (DATE): The date the transaction occurred.
            - `amount` (FLOAT): The transaction amount.
            - `transaction_type` (STRING): 'Debit' or 'Credit'.
            - `persona_type` (STRING): The persona of the user.
            - `description_raw` (STRING): The original transaction description.
            - `merchant_name_raw` (STRING): The original merchant name.
            - `description_cleaned` (STRING): Cleaned version of the description.
            - `merchant_name_cleaned` (STRING): Cleaned version of the merchant name.
            - `category_l1` (STRING): The top-level category (e.g., 'Expense', 'Income').
            - `category_l2` (STRING): The sub-category (e.g., 'Groceries', 'Shopping').
            - `is_recurring` (BOOL): Flag for recurring transactions.
            - `channel` (STRING): The source channel (e.g., 'Online', 'In-store').

        - **`{RULES_TABLE_ID}` (categorization_rules)**: Stores rules for categorization.
          - **Key Columns**:
            - `rule_id` (STRING): Unique identifier for the rule.
            - `identifier` (STRING): The string to match (e.g., merchant or pattern).
            - `rule_type` (STRING): 'MERCHANT' or 'PATTERN'.
            - `persona_type` (STRING): The persona this rule applies to.
            - `category_l1` (STRING): L1 category.
            - `category_l2` (STRING): L2 category.
            - `transaction_type` (STRING): 'Debit', 'Credit', or 'All'.
            - `is_recurring_rule` (BOOL): Flag to set recurring status.
            - `confidence_score` (INT64): Confidence level of the rule.
            - `is_active` (BOOL): Whether the rule is currently active.

    - Based on the user's natural language question, you should formulate a valid SQL query using the information above and pass it to the `run_custom_query` tool. Always use the `{TABLE_ID}` and `{RULES_TABLE_ID}` variables instead of hardcoded table names.
    """
)