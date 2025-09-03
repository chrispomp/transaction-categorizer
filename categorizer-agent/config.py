# config.py

from __future__ import annotations
import os
import json
import logging
from dotenv import load_dotenv

import pandas as pd
import vertexai
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError

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
    RULES_TABLE_NAME = os.environ["BIGQUERY_RULES_TABLE"]
    TEMP_TABLE_NAME = os.environ["BIGQUERY_TEMP_TABLE"]
    LOCATION = os.getenv("GCP_LOCATION", "us-central1")

    TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
    RULES_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{RULES_TABLE_NAME}"
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
    "Income": ["Gig Income", "Payroll", "Other Income", "Refund", "Interest Income"],
    "Expense": [
        "Groceries", "Pharmacy", "Office Supplies", "Food & Dining", "Shopping", "Entertainment",
        "Health & Wellness", "Auto & Transport", "Travel & Vacation",
        "Software & Tech", "Medical", "Insurance", "Bills & Utilities",
        "Fees & Charges", "Business Services", "Other Expense", "Loan Payment"
    ],
    "Transfer": ["Credit Card Payment", "Internal Transfer", "ATM Withdrawal"]
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
        logger.error(f"Failed to decode or parse LLM JSON: {e}\\nInput: {json_string}")
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