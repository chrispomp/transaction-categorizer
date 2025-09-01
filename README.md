  export GOOGLE_GENAI_USE_VERTEXAI=TRUE
  export GCP_PROJECT_ID="fsi-banking-agentspace"
  export BIGQUERY_DATASET="equifax_txns"
  export BIGQUERY_TABLE="transactions"
  export BIGQUERY_RULES_TABLE="categorization_rules"
  export BIGQUERY_TEMP_TABLE="temp_updates"

  source .venv/bin/activate

  adk web