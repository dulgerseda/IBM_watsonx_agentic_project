"""
Analyst Tool for IBM watsonx Orchestrate.
Analyses persona survey responses and generates structured insights for the Visa product team.
"""

import json
import requests
import time
from ibm_watsonx_orchestrate.agent_builder.tools import tool
from ibm_watsonx_orchestrate.agent_builder.connections import ConnectionType
from ibm_watsonx_orchestrate.run import connections

MY_APP_ID = "watsonx_credentials"

ANALYST_SYSTEM_PROMPT = """You are a Visa Switzerland consumer insights analyst. 
Analyse survey responses from 5 Swiss personas and produce a structured JSON insight report.

Segments (population weight — fee ceiling):
- Felix/Student: 8% — CHF 5
- Lukas/Professional: 12% — CHF 25-30
- Anna/Mittelstand: 28% — CHF 12
- Sandra/Family: 22% — CHF 10
- Peter/Retired: 18% — CHF 5

Rules:
- Anna (28%) matters most for market sizing
- MAYBE=45%, YES=85%, NO=5% adoption probability
- Mass market sweet spot: CHF 8 max unless justified
- Peter is NOT a priority acquisition target
- 3+ NO segments = price not viable for mass market

Produce a JSON object with these fields only:
{
  "insight_id": "insight_<survey_run_id>",
  "generated_at": "<ISO timestamp>",
  "survey_run_id": "<echoed from input>",
  "personas_analysed": <count>,
  "segment_responses": {
    "<persona_name>": {
      "verdict": "YES | NO | MAYBE",
      "key_reasoning": "<one sentence>",
      "fee_ceiling": "<CHF X>"
    }
  },
  "cross_segment_insight": "<3-5 sentence synthesis>",
  "pricing_sweet_spot": "CHF X — one sentence explanation",
  "top_features": ["feature 1", "feature 2", "feature 3"],
  "main_barriers": ["barrier 1", "barrier 2"],
  "visa_recommendation": "<2-3 sentence actionable recommendation for Visa product team>",
  "confidence_level": "HIGH | MEDIUM | LOW"
}

Return ONLY the JSON object. No explanation, no markdown, no extra text."""

_iam_token_cache = None
_iam_token_expiry = 0


def _get_iam_token(credentials) -> str:
    global _iam_token_cache, _iam_token_expiry
    if not _iam_token_cache or time.time() >= _iam_token_expiry:
        resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": credentials["WATSONX_API_KEY"],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        _iam_token_cache = data["access_token"]
        _iam_token_expiry = time.time() + data.get("expires_in", 3600) - 60
    return _iam_token_cache


def _call_watsonx(prompt: str, credentials) -> str:
    token = _get_iam_token(credentials)
    url = credentials["WATSONX_URL"].rstrip("/")
    project_id = credentials["WATSONX_PROJECT_ID"]
    model_id = credentials.get("WATSONX_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

    full_prompt = f"<s>[INST] <<SYS>>\n{ANALYST_SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"

    payload = {
        "model_id": model_id,
        "project_id": project_id,
        "input": full_prompt,
        "parameters": {
            "max_new_tokens": 4000,
            "temperature": 0.2,
            "repetition_penalty": 1.1,
            "stop_sequences": ["</s>", "[/INST]", "<|eom_id|>"],
        },
    }

    for attempt in range(5):
        resp = requests.post(
            f"{url}/ml/v1/text/generation?version=2023-05-29",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if resp.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()["results"][0]["generated_text"].strip()
    resp.raise_for_status()


@tool(expected_credentials=[
        {"app_id": MY_APP_ID, "type": ConnectionType.KEY_VALUE}
    ])
def analyse_persona_responses(
    question_text: str,
    question_category: str,
    responses_json: str,
) -> str:
    """
    Analyses survey responses from Swiss consumer personas and generates structured Visa insights.

    Args:
        question_text: The survey question that was asked to all personas.
        question_category: The category of the question, e.g. price_sensitivity, feature_preference, behavior_switching, trade_off, willingness_to_pay.
        responses_json: A JSON string mapping persona names to their answers. Example: {"student": "answer...", "family": "answer...", "mittelstand": "answer...", "professional": "answer...", "retired": "answer..."}.

    Returns:
        A structured JSON insight report with verdicts, recommendations, and pricing analysis for the Visa product team.
    """
    conn = connections.key_value(MY_APP_ID)
    try:
        responses = json.loads(responses_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid responses_json format: {str(e)}"

    PERSONA_METADATA = {
        "student":      {"agent_label": "Campus Felix — Student Segment",            "population_weight_pct": 8,  "fee_ceiling": "CHF 5"},
        "professional": {"agent_label": "Mobile-First Lukas — Professional Segment", "population_weight_pct": 12, "fee_ceiling": "CHF 25-30"},
        "mittelstand":  {"agent_label": "Average-Earner Anna — Mittelstand Segment", "population_weight_pct": 28, "fee_ceiling": "CHF 12"},
        "family":       {"agent_label": "Value-Smart Sandra — Family Segment",       "population_weight_pct": 22, "fee_ceiling": "CHF 10"},
        "retired":      {"agent_label": "Trust-First Peter — Retired Segment",       "population_weight_pct": 18, "fee_ceiling": "CHF 5"},
    }

    response_objects = []
    for persona_key, meta in PERSONA_METADATA.items():
        answer = responses.get(persona_key, "")
        if answer:
            response_objects.append({
                "agent_label":           meta["agent_label"],
                "population_weight_pct": meta["population_weight_pct"],
                "fee_ceiling":           meta["fee_ceiling"],
                "verbatim_response":     answer[:400],
            })

    bundle = {
        "question_text":     question_text,
        "question_category": question_category,
        "responses":         response_objects,
    }

    prompt = f"Analyse the following Swiss consumer survey response bundle:\n\n{json.dumps(bundle, indent=2, ensure_ascii=False)}"

    try:
        raw = _call_watsonx(prompt, conn)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1:
            return f"Error: No JSON found in response: {raw[:300]}"
        return raw[start:end]
    except Exception as e:
        return f"Error during analysis: {str(e)}"