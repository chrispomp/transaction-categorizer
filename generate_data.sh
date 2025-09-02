# Generates high-fidelity, narratively cohesive synthetic data for testing
# AI/ML models for credit scoring based on transaction history.
# Version 6.0 - Added raw and cleaned merchant name fields.

import os
import json
import uuid
import random
import logging
import asyncio
import time
import re
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Optional, Tuple

# Faker for realistic data generation
from faker import Faker
# NumPy for statistical modeling
import numpy as np

# Using Vertex AI SDK for GCP integration
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import bigquery
from google.api_core import exceptions
from google.api_core import retry_async

# --- Configuration ---
PROJECT_ID = os.getenv("PROJECT_ID", "fsi-banking-agentspace")
LOCATION = os.getenv("LOCATION", "us-central1")
DATASET_ID = os.getenv("DATASET_ID", "equifax_txns")
TABLE_ID = "transactions"

# --- Generation Parameters ---
NUM_CONSUMERS_TO_GENERATE = 5
MIN_VARIABLE_TRANSACTIONS_PER_MONTH = 35
MAX_VARIABLE_TRANSACTIONS_PER_MONTH = 50
TRANSACTION_HISTORY_MONTHS = 12
CONCURRENT_CONSUMER_JOBS = 5

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Faker ---
fake = Faker()

# --- I. CANONICAL DATA STRUCTURES ---

INSTITUTION_NAMES = ["Capital One", "Chase", "Ally Bank", "Bank of America"]
ACCOUNT_TYPES = ["Credit Card", "Checking Account", "Savings Account"]
CHANNELS = ["ATM", "Point-of-Sale", "Card-Not-Present", "Wire Transfer", "ACH", "Check", "P2P", "Internal Transfer"]

EXPENSE_TAXONOMY = [
    {"category_l1": "Expense", "category_l2": "Groceries", "tier": "Premium", "merchant": "Whole Foods Market", "channel": "Point-of-Sale"},
    {"category_l1": "Expense", "category_l2": "Groceries", "tier": "Mid-Range", "merchant": "Trader Joe's", "channel": "Point-of-Sale"},
    {"category_l1": "Expense", "category_l2": "Food & Dining", "tier": "Premium", "merchant": "Blue Bottle Coffee", "channel": "Point-of-Sale"},
    {"category_l1": "Expense", "category_l2": "Food & Dining", "tier": "Mid-Range", "merchant": "Chipotle", "channel": "Point-of-Sale"},
    {"category_l1": "Expense", "category_l2": "Shopping", "tier": "Mid-Range", "merchant": "Amazon.com", "channel": "Card-Not-Present"},
    {"category_l1": "Expense", "category_l2": "Shopping", "tier": "Premium", "merchant": "Lululemon", "channel": "Card-Not-Present"},
    {"category_l1": "Expense", "category_l2": "Entertainment", "tier": "Mid-Range", "merchant": "Netflix.com", "channel": "Card-Not-Present"},
    {"category_l1": "Expense", "category_l2": "Health & Wellness", "tier": "Mid-Range", "merchant": "CVS Pharmacy", "channel": "Point-of-Sale"},
    {"category_l1": "Expense", "category_l2": "Auto & Transport", "tier": "Mid-Range", "merchant": "Uber", "channel": "Card-Not-Present"},
    {"category_l1": "Expense", "category_l2": "Travel & Vacation", "tier": "Mid-Range", "merchant": "Expedia", "channel": "Card-Not-Present"},
    {"category_l1": "Expense", "category_l2": "Auto & Transport", "tier": "Mid-Range", "merchant": "Shell Gas Station", "channel": "Point-of-Sale"},
    {"category_l1": "Expense", "category_l2": "Software & Tech", "tier": "Premium", "merchant": "ADOBE INC", "channel": "Card-Not-Present"},
    {"category_l1": "Expense", "category_l2": "Shopping", "tier": "Mid-Range", "merchant": "Staples", "channel": "Point-of-Sale"},
    {"category_l1": "Expense", "category_l2": "Medical", "tier": "Premium", "merchant": "City Hospital", "channel": "ACH"},
    {"category_l1": "Expense", "category_l2": "Insurance", "tier": "Mid-Range", "merchant": "GEICO", "channel": "ACH"},
    {"category_l1": "Expense", "category_l2": "Bills & Utilities", "tier": "Mid-Range", "merchant": "T-MOBILE", "channel": "ACH"},
]

INCOME_CATEGORIES = [
    {"category_l2": "Income", "description_template": "{merchant} DEPOSIT PMT_{id}"},
    {"category_l2": "Interest Income", "description_template": "{bank} INTEREST PAYMENT"},
    {"category_l2": "Refund", "description_template": "REFUND FROM {merchant}"},
    {"category_l2": "Other Income", "description_template": "MISC CREDIT {origin}"},
]

MERCHANT_TO_CHANNEL_MAP = {item['merchant']: item['channel'] for item in EXPENSE_TAXONOMY}

LIFE_EVENT_IMPACT_MATRIX = {
    "Unexpected Major Car Repair": {
        "category": "Negative Financial Shock", "magnitude_range": (-2500, -1000), "duration": 2,
        "primary_signature": {"category_l2": "Auto & Transport", "merchant_options": ["Firestone Auto Care", "Pep Boys", "Local Mechanic LLC"]},
        "secondary_effects_prompt": "Reflect a period of reduced discretionary spending (less Food & Dining, Shopping, Entertainment) for the next 2 months to recover from the car repair cost."
    },
    "Significant Medical Bill": {
        "category": "Negative Financial Shock", "magnitude_range": (-3000, -500), "duration": 3,
        "primary_signature": {"category_l2": "Medical", "merchant_options": ["Local Hospital Billing", "Specialist Co-Pay", "Out-of-Network Dr."]},
        "secondary_effects_prompt": "Significantly reduce non-essential spending for the next 3 months to accommodate this large, unexpected medical cost."
    },
}

PERSONAS = [
    {
        "persona_name": "Full-Time Rideshare Driver",
        "description": "Earns primary income from Uber and Lyft. Co-mingles funds, using the same account for business (gas, maintenance) and personal (food, shopping) expenses.",
        "income_merchants": ["UBER", "LYFT", "DOORDASH"], "p2p_income_channels": [],
        "business_expense_categories": ["Auto & Transport"],
        "spending_tier_affinities": {"Budget": 0.5, "Mid-Range": 0.4, "Premium": 0.1},
        "recurring_expenses": [
            {"merchant_name": "GEICO INSURANCE", "day_of_month": 1, "amount_mean": -180.50, "amount_std": 10.0, "category_l1": "Expense", "category_l2": "Insurance"},
            {"merchant_name": "T-MOBILE", "day_of_month": 15, "amount_mean": -95.00, "amount_std": 5.0, "category_l1": "Expense", "category_l2": "Bills & Utilities"},
        ]
    },
    {
        "persona_name": "Freelance Creative",
        "description": "A graphic designer who receives client payments via Stripe (for corporate clients) and P2P apps (for individual clients). Pays for business software and personal living costs from one account.",
        "income_merchants": ["STRIPE"], "p2p_income_channels": ["Zelle", "Cash App", "PayPal"],
        "business_expense_categories": ["Software & Tech", "Shopping", "Bills & Utilities"],
        "spending_tier_affinities": {"Budget": 0.1, "Mid-Range": 0.4, "Premium": 0.5},
        "recurring_expenses": [
            {"merchant_name": "ADOBE INC", "day_of_month": 5, "amount_mean": -59.99, "amount_std": 0, "category_l1": "Expense", "category_l2": "Software & Tech"},
        ]
    },
]

# --- II. HYBRID GENERATION & STATISTICAL MODELING ---

AMOUNT_DISTRIBUTIONS = {
    "Groceries": {"log_mean": 3.8, "log_std": 0.6}, "Food & Dining": {"log_mean": 2.8, "log_std": 0.8},
    "Shopping": {"log_mean": 4.2, "log_std": 1.0}, "Entertainment": {"log_mean": 3.2, "log_std": 0.7},
    "Health & Wellness": {"log_mean": 3.5, "log_std": 0.8}, "Medical": {"log_mean": 4.5, "log_std": 1.1},
    "Auto & Transport": {"log_mean": 3.4, "log_std": 0.9}, "Travel & Vacation": {"log_mean": 5.5, "log_std": 0.8},
    "Income": {"log_mean": 6.5, "log_std": 0.5}, "Interest Income": {"log_mean": 2.5, "log_std": 0.4},
    "Refund": {"log_mean": 4.0, "log_std": 0.9}, "Other Income": {"log_mean": 5.0, "log_std": 1.2},
    "Peer to Peer Transfer": {"log_mean": 4.8, "log_std": 1.0},
    "Default": {"log_mean": 3.0, "log_std": 1.0}
}

WEEKDAY_HOUR_WEIGHTS = [1, 1, 1, 1, 1, 2, 5, 8, 9, 7, 5, 6, 10, 10, 8, 6, 7, 9, 9, 8, 6, 4, 3, 2]
WEEKEND_HOUR_WEIGHTS = [2, 2, 1, 1, 1, 2, 3, 4, 6, 8, 10, 10, 9, 8, 7, 7, 8, 9, 9, 8, 6, 5, 4, 3]

def generate_realistic_amount(category_l2: str) -> float:
    params = AMOUNT_DISTRIBUTIONS.get(category_l2, AMOUNT_DISTRIBUTIONS["Default"])
    amount = np.random.lognormal(mean=params['log_mean'], sigma=params['log_std'])
    return round(amount, 2)

def generate_realistic_timestamp(base_date: datetime) -> datetime:
    is_weekday = base_date.weekday() < 5
    weights = WEEKDAY_HOUR_WEIGHTS if is_weekday else WEEKEND_HOUR_WEIGHTS
    hour = random.choices(range(24), weights=weights, k=1)[0]
    minute, second = random.randint(0, 59), random.randint(0, 59)
    return base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)

def clean_description(raw_desc: str) -> str:
    if not isinstance(raw_desc, str):
        return ""
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', raw_desc).lower()
    return re.sub(r'\s+', ' ', cleaned).strip()

# --- III. PROMPT ENGINEERING & SCHEMA DEFINITION ---

TRANSACTION_SCHEMA_FOR_LLM = {
    "type": "array", "items": {
        "type": "object", "properties": {
            "description_raw": {"type": "string", "description": "A realistic, varied, raw transaction description."},
            "merchant_name_raw": {"type": "string", "description": "The raw merchant name as extracted from the description (e.g., 'SQ *BLUE BOTTLE COFFEE #B12')."},
            "merchant_name_cleaned": {"type": "string", "description": "The cleaned, canonical merchant name (e.g., 'Blue Bottle Coffee')."},
            "category_l2": {"type": "string", "description": "The detailed sub-category of the transaction."}
        }, "required": ["description_raw", "merchant_name_raw", "merchant_name_cleaned", "category_l2"]
    }
}

def build_monthly_prompt(profile: Dict, month_date: datetime, transactions_this_month: int, active_event: Optional[Dict], seasonal_context: Optional[str]) -> str:
    month_name = month_date.strftime("%B %Y")
    persona = profile['persona']
    narrative_block = "This was a typical month."
    if active_event:
        narrative_block = f"CRITICAL NARRATIVE EVENT: This month, the consumer is dealing with '{active_event['name']}'. {active_event['details']['secondary_effects_prompt']}"
    elif seasonal_context:
        narrative_block = f"SEASONAL CONTEXT: {seasonal_context}"

    income_merchant_example = random.choice(persona.get("income_merchants", ["DEPOSIT"]))
    
    few_shot_examples = f"""
    **High-Quality Output Examples (for style guidance only):**
    ```json
    [
        {{"description_raw": "SQ *BLUE BOTTLE COFFEE #B12", "merchant_name_raw": "SQ *BLUE BOTTLE COFFEE", "merchant_name_cleaned": "Blue Bottle Coffee", "category_l2": "Food & Dining"}},
        {{"description_raw": "POS DEBIT TRADER JOE'S #552 PHOENIX AZ", "merchant_name_raw": "TRADER JOE'S #552", "merchant_name_cleaned": "Trader Joe's", "category_l2": "Groceries"}},
        {{"description_raw": "TST* The Corner Bistro", "merchant_name_raw": "TST* The Corner Bistro", "merchant_name_cleaned": "The Corner Bistro", "category_l2": "Food & Dining"}},
        {{"description_raw": "AMAZON.COM*A12B34CD5 AMZN.COM/BILL WA", "merchant_name_raw": "AMAZON.COM*A12B34CD5", "merchant_name_cleaned": "Amazon.com", "category_l2": "Shopping"}},
        {{"description_raw": "UBER TRIP 6J7K8L HELP.UBER.COM", "merchant_name_raw": "UBER TRIP", "merchant_name_cleaned": "Uber", "category_l2": "Auto & Transport"}},
        {{"description_raw": "{income_merchant_example} DEPOSIT PMT_1234", "merchant_name_raw": "{income_merchant_example} DEPOSIT", "merchant_name_cleaned": "{income_merchant_example}", "category_l2": "Income"}}
    ]
    ```
    """
    
    return f"""
    Generate a flat JSON array of exactly {transactions_this_month} realistic, variable bank transactions for '{profile["consumer_name"]}' for **{month_name}**.
    - Persona: '{persona["persona_name"]}' ({persona["description"]})
    - **Monthly Narrative:** {narrative_block}
    **CRITICAL INSTRUCTIONS:**
    1.  For each transaction, provide a `description_raw`, `merchant_name_raw`, `merchant_name_cleaned`, and `category_l2`.
    2.  **The `merchant_name_raw` MUST be consistent with the `description_raw`.**
    3.  **The `merchant_name_cleaned` MUST be the canonical, recognizable name of the business.**
    4.  For income, use `category_l2`: "Income", "Refund", or "Other Income".
    5.  Do NOT generate predictable monthly bills; they are handled separately.
    6.  The entire output MUST be ONLY the raw JSON array, conforming strictly to the provided schema. Ensure all strings are properly escaped.
    {few_shot_examples}
    **Your Task:** Generate the JSON array for **{month_name}**.
    """

# --- IV. CLIENT & BQ SETUP ---

try:
    if not PROJECT_ID or "your-gcp-project-id" in PROJECT_ID:
        raise ValueError("PROJECT_ID is not set correctly.")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-flash")
    bq_client = bigquery.Client(project=PROJECT_ID)
    logging.info(f"Initialized Vertex AI and BigQuery for project '{PROJECT_ID}'")
except Exception as e:
    logging.error(f"Failed to initialize Google Cloud clients: {e}. Ensure you are authenticated ('gcloud auth application-default login').")
    exit()

TRANSACTIONS_SCHEMA = [
    bigquery.SchemaField("transaction_id", "STRING", mode="REQUIRED", description="Primary Key. A unique identifier for the transaction."),
    bigquery.SchemaField("account_id", "STRING", mode="REQUIRED", description="Identifier for a specific bank account this transaction belongs to."),
    bigquery.SchemaField("consumer_name", "STRING", mode="NULLABLE", description="The first and last name of the synthetic consumer."),
    bigquery.SchemaField("persona_type", "STRING", mode="NULLABLE", description="The generated persona profile for the consumer (e.g., 'Freelance Creative')."),
    bigquery.SchemaField("institution_name", "STRING", mode="NULLABLE", description="The name of the financial institution (e.g., 'Chase')."),
    bigquery.SchemaField("account_type", "STRING", mode="NULLABLE", description="The type of account (e.g., 'Checking Account', 'Credit Card')."),
    bigquery.SchemaField("transaction_date", "TIMESTAMP", mode="NULLABLE", description="The exact date and time the transaction was posted."),
    bigquery.SchemaField("transaction_type", "STRING", mode="NULLABLE", description="The type of transaction, either 'Debit' or 'Credit'."),
    bigquery.SchemaField("amount", "FLOAT", mode="NULLABLE", description="The value of the transaction. Negative for debits, positive for credits."),
    bigquery.SchemaField("is_recurring", "BOOLEAN", mode="NULLABLE", description="A boolean flag (True/False) to indicate if it's a predictable, recurring payment."),
    bigquery.SchemaField("description_raw", "STRING", mode="NULLABLE", description="The original, unaltered transaction description from the bank statement."),
    bigquery.SchemaField("description_cleaned", "STRING", mode="NULLABLE", description="A standardized and cleaned version of the raw description (lowercase, no special characters)."),
    bigquery.SchemaField("merchant_name_raw", "STRING", mode="NULLABLE", description="The raw, potentially messy merchant name as it might appear on a statement."),
    bigquery.SchemaField("merchant_name_cleaned", "STRING", mode="NULLABLE", description="The cleaned, canonical name of the merchant for analytics."),
    bigquery.SchemaField("category_l1", "STRING", mode="NULLABLE", description="The high-level category: 'Income', 'Expense', or 'Transfer'."),
    bigquery.SchemaField("category_l2", "STRING", mode="NULLABLE", description="The detailed sub-category (e.g., 'Groceries', 'Software & Tech', 'Interest Income')."),
    bigquery.SchemaField("channel", "STRING", mode="NULLABLE", description="The method or channel of the transaction (e.g., 'Point-of-Sale', 'ACH')."),
    bigquery.SchemaField("categorization_update_timestamp", "TIMESTAMP", mode="NULLABLE", description="The timestamp when the transaction's categorization was last updated."),
    bigquery.SchemaField("categorization_method", "STRING", mode="NULLABLE", description="The method for which the transaction's categorization was updated.")
]

@retry_async.AsyncRetry(predicate=retry_async.if_exception_type(exceptions.Aborted, exceptions.DeadlineExceeded, exceptions.ServiceUnavailable, exceptions.TooManyRequests), initial=1.0, maximum=16.0, multiplier=2.0)
async def generate_data_with_gemini(prompt: str) -> List[Dict[str, Any]]:
    logging.info(f"Sending prompt to Gemini (first 120 chars): {prompt[:120].strip().replace('/n', '')}...")
    try:
        generation_config = GenerationConfig(response_mime_type="application/json", response_schema=TRANSACTION_SCHEMA_FOR_LLM, temperature=1.0, max_output_tokens=8192)
        response = await model.generate_content_async(prompt, generation_config=generation_config)
        
        # Clean the response before parsing
        text_response = response.text.strip()
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]

        return json.loads(text_response)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from Vertex AI API: {e}")
        logging.error(f"Raw response: {response.text}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred with the Vertex AI API: {e}")
        return []

def setup_bigquery_table():
    full_dataset_id = f"{PROJECT_ID}.{DATASET_ID}"
    dataset = bigquery.Dataset(full_dataset_id)
    dataset.location = LOCATION
    try:
        bq_client.create_dataset(dataset, exists_ok=True)
        logging.info(f"Dataset '{full_dataset_id}' is ready.")
    except Exception as e:
        logging.error(f"Failed to create or verify dataset: {e}")
        exit()
    table_ref = bq_client.dataset(DATASET_ID).table(TABLE_ID)
    try:
        bq_client.delete_table(table_ref, not_found_ok=True)
        logging.info(f"Ensured old table '{TABLE_ID}' is removed.")
    except Exception as e:
        logging.error(f"Error during table cleanup: {e}")
    table = bigquery.Table(table_ref, schema=TRANSACTIONS_SCHEMA)
    try:
        bq_client.create_table(table)
        logging.info(f"Sent request to create table '{TABLE_ID}'.")
        time.sleep(5)
    except Exception as e:
        logging.error(f"Failed to send create table request: {e}")
        exit()
    max_retries = 10
    retry_delay = 5
    for i in range(max_retries):
        try:
            bq_client.get_table(table_ref)
            logging.info(f"✅ Table '{TABLE_ID}' successfully verified and is ready.")
            return
        except exceptions.NotFound:
            if i < max_retries - 1:
                logging.warning(f"Table not found yet, retrying in {retry_delay}s... (Attempt {i+2}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logging.error("Fatal: Failed to verify table creation after multiple retries. Aborting.")
                exit()

def upload_to_bigquery(data: List[Dict[str, Any]]):
    if not data:
        logging.warning("No data to upload.")
        return
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    job_config = bigquery.LoadJobConfig(
        schema=TRANSACTIONS_SCHEMA,
        write_disposition="WRITE_APPEND",
    )
    try:
        logging.info(f"Starting BigQuery Load Job to insert {len(data)} rows into {table_id}...")
        load_job = bq_client.load_table_from_json(data, table_id, job_config=job_config)
        load_job.result()
        logging.info(f"✅ Success! Load Job complete. Loaded {len(data)} rows.")
    except Exception as e:
        logging.error(f"An unexpected exception occurred during the BigQuery Load Job: {e}")
        if 'load_job' in locals() and load_job.errors:
            for error in load_job.errors:
                logging.error(f"  - BQ Error: {error['message']}")
        raise

# --- V. PROGRAMMATIC & HYBRID TRANSACTION LOGIC ---

def generate_life_events(history_months: int) -> List[Dict[str, Any]]:
    events = []
    if history_months <= 1: return events
    num_events = random.randint(1, min(2, history_months - 1))
    used_months = set()
    for _ in range(num_events):
        event_name = random.choice(list(LIFE_EVENT_IMPACT_MATRIX.keys()))
        details = LIFE_EVENT_IMPACT_MATRIX[event_name]
        potential_start = random.randint(1, history_months - 1)
        if not any(m in used_months for m in range(potential_start, potential_start + details['duration'])):
            for m in range(potential_start, potential_start + details['duration']): used_months.add(m)
            events.append({"name": event_name, "details": details, "magnitude": round(random.uniform(*details['magnitude_range']), 2), "start_month_ago": potential_start, "end_month_ago": potential_start - details['duration']})
    return events

def inject_recurring_transactions(profile: Dict, history_months: int) -> List[Dict[str, Any]]:
    persona = profile['persona']
    accounts = profile['accounts']
    checking_account = accounts['Checking Account']
    txns = []
    for bill in persona.get('recurring_expenses', []):
        for i in range(history_months):
            base_date = datetime.now(timezone.utc) - relativedelta(months=i)
            day = min(bill['day_of_month'], (base_date.replace(day=28) + timedelta(days=4)).day)
            date = base_date.replace(day=day, hour=random.randint(9, 17), minute=random.randint(0, 59))
            amount = round(random.gauss(bill['amount_mean'], bill['amount_std']), 2)
            raw_desc = f"ACH DEBIT - {bill['merchant_name']}"
            txns.append({
                "transaction_id": f"TXN-{uuid.uuid4()}", "account_id": checking_account['account_id'],
                "consumer_name": profile['consumer_name'], "persona_type": persona['persona_name'],
                "institution_name": checking_account['institution_name'], "account_type": "Checking Account",
                "transaction_date": date.isoformat(), "transaction_type": "Debit", "amount": amount, "is_recurring": True,
                "description_raw": raw_desc, "description_cleaned": clean_description(raw_desc),
                "merchant_name_raw": bill['merchant_name'], "merchant_name_cleaned": bill['merchant_name'], 
                "category_l1": bill['category_l1'], "category_l2": bill['category_l2'], "channel": "ACH",
                "categorization_update_timestamp": datetime.now(timezone.utc).isoformat(),
            })
    return txns

def inject_programmatic_event_transactions(profile: Dict, life_events: List[Dict]) -> List[Dict[str, Any]]:
    txns = []
    accounts = profile['accounts']
    for event in life_events:
        if not (sig := event['details'].get('primary_signature')): continue
        amount = event['magnitude']
        is_credit = amount > 0
        account_type = "Checking Account" if is_credit else random.choice(["Checking Account", "Credit Card"])
        account = accounts.get(account_type, accounts['Checking Account'])
        channel = "ACH" if is_credit else "Card-Not-Present"
        date = (datetime.now(timezone.utc) - relativedelta(months=event['start_month_ago'])).replace(day=random.randint(5, 25), hour=random.randint(10, 16))
        merchant = random.choice(sig['merchant_options'])
        raw_desc = f"EVENT: {event['name'].upper()}"
        txns.append({
            "transaction_id": f"TXN-{uuid.uuid4()}", "account_id": account['account_id'],
            "consumer_name": profile['consumer_name'], "persona_type": profile['persona']['persona_name'],
            "institution_name": account['institution_name'], "account_type": account_type, "transaction_date": date.isoformat(),
            "transaction_type": "Credit" if is_credit else "Debit", "amount": amount, "is_recurring": False,
            "description_raw": raw_desc, "description_cleaned": clean_description(raw_desc), 
            "merchant_name_raw": merchant, "merchant_name_cleaned": merchant,
            "category_l1": "Income" if is_credit else "Expense", "category_l2": sig['category_l2'], "channel": channel,
            "categorization_update_timestamp": datetime.now(timezone.utc).isoformat(),
        })
    return txns

async def generate_cohesive_txns_for_consumer(profile: Dict, history_months: int, min_txns: int, max_txns: int) -> List[Dict[str, Any]]:
    logging.info(f"Generating full history for '{profile['consumer_name']}'...")
    life_events = generate_life_events(history_months)
    final_txns = inject_recurring_transactions(profile, history_months)
    final_txns.extend(inject_programmatic_event_transactions(profile, life_events))
    
    cat_lookup = {item['category_l2']: "Expense" for item in EXPENSE_TAXONOMY}
    for inc_cat in INCOME_CATEGORIES:
        cat_lookup[inc_cat['category_l2']] = "Income"
    cat_lookup.update({"Peer to Peer Transfer": "Transfer"})

    for i in range(history_months):
        month_date = datetime.now(timezone.utc) - relativedelta(months=i)
        active_event = next((e for e in life_events if e['end_month_ago'] < i <= e['start_month_ago']), None)
        seasonal = "Holiday season spending." if month_date.month in [11, 12] else "Summer vacation spending." if month_date.month in [6, 7, 8] else None
        
        prompt = build_monthly_prompt(profile, month_date, random.randint(min_txns, max_txns), active_event, seasonal)
        monthly_txns_from_llm = await generate_data_with_gemini(prompt)
        
        for txn in monthly_txns_from_llm:
            try:
                # Get initial data from the LLM response
                cat_l2 = txn['category_l2']
                merchant_raw = txn['merchant_name_raw']
                merchant_cleaned = txn['merchant_name_cleaned']
                
                # Programmatically determine financial details
                amount = generate_realistic_amount(cat_l2)
                is_credit = cat_lookup.get(cat_l2) == "Income" or cat_l2 == "Peer to Peer Transfer"
                
                if is_credit:
                    account_type = "Checking Account"
                    transaction_type = "Credit"
                    final_amount = abs(amount)
                    channel = "ACH"
                    if cat_l2 == "Peer to Peer Transfer":
                        p2p_name = fake.name()
                        merchant_raw = p2p_name
                        merchant_cleaned = p2p_name
                        channel = "P2P"
                else: # Is Expense
                    account_type = random.choice([acc for acc in profile['accounts'].keys() if acc != "Savings Account"])
                    transaction_type = "Debit"
                    final_amount = -abs(amount)
                    # Use the cleaned merchant name for reliable channel lookup
                    channel = MERCHANT_TO_CHANNEL_MAP.get(merchant_cleaned, "Card-Not-Present")
                
                account = profile['accounts'][account_type]
                
                # Consolidate all fields and add to the original dictionary from the LLM
                txn.update({
                    "transaction_id": f"TXN-{uuid.uuid4()}",
                    "account_id": account['account_id'],
                    "consumer_name": profile['consumer_name'],
                    "persona_type": profile['persona']['persona_name'],
                    "institution_name": account['institution_name'],
                    "account_type": account_type,
                    "transaction_date": generate_realistic_timestamp(month_date.replace(day=random.randint(1, 28))).isoformat(),
                    "transaction_type": transaction_type,
                    "amount": final_amount,
                    "is_recurring": False,
                    "description_cleaned": clean_description(txn.get('description_raw')),
                    "merchant_name_raw": merchant_raw,
                    "merchant_name_cleaned": merchant_cleaned,
                    "category_l1": cat_lookup.get(cat_l2, "Expense"),
                    "channel": channel,
                    "categorization_update_timestamp": datetime.now(timezone.utc).isoformat(),
                })
                final_txns.append(txn)
            except (ValueError, TypeError, KeyError) as e:
                logging.warning(f"Could not process transaction: {e}. Raw txn: {txn}")
    return final_txns


# --- VI. MAIN ORCHESTRATION ---
async def generate_gig_worker_transactions_main(num_consumers: int, min_txns_monthly: int, max_txns_monthly: int, history_months: int, concurrent_jobs: int):
    logging.info(f"--- Starting High-Fidelity Synthetic Data Generation for {num_consumers} Consumers ---")
    setup_bigquery_table()
    
    consumer_profiles = []
    for _ in range(num_consumers):
        consumer_name = fake.name()
        institutions_for_consumer = random.sample(INSTITUTION_NAMES, k=min(len(INSTITUTION_NAMES), 2))
        profile = {
            "consumer_name": consumer_name, 
            "persona": random.choice(PERSONAS),
            "accounts": {
                "Checking Account": {"account_id": f"ACC-{str(uuid.uuid4())[:12].upper()}", "institution_name": institutions_for_consumer[0]},
                "Credit Card": {"account_id": f"ACC-{str(uuid.uuid4())[:12].upper()}", "institution_name": institutions_for_consumer[1]},
            }
        }
        consumer_profiles.append(profile)

    semaphore = asyncio.Semaphore(concurrent_jobs)
    async def process_and_upload(profile):
        async with semaphore:
            transactions = await generate_cohesive_txns_for_consumer(profile, history_months, min_txns_monthly, max_txns_monthly)
            if transactions:
                upload_to_bigquery(transactions)
                return len(transactions)
            return 0

    tasks = [process_and_upload(profile) for profile in consumer_profiles]
    results = await asyncio.gather(*tasks)
    
    total_generated = sum(results)
    if not total_generated:
        logging.error("No transaction data was generated. Aborting.")
        return

    logging.info(f"Generated a grand total of {total_generated} transactions.")
    logging.info("--- ✅ High-Fidelity Data Generation and Upload Complete! ---")


if __name__ == "__main__":
    try:
        asyncio.run(generate_gig_worker_transactions_main(
            num_consumers=NUM_CONSUMERS_TO_GENERATE,
            min_txns_monthly=MIN_VARIABLE_TRANSACTIONS_PER_MONTH,
            max_txns_monthly=MAX_VARIABLE_TRANSACTIONS_PER_MONTH,
            history_months=TRANSACTION_HISTORY_MONTHS,
            concurrent_jobs=CONCURRENT_CONSUMER_JOBS
        ))
    except (RuntimeError, Exception) as e:
        logging.error(f"The script failed to complete due to an error: {e}")