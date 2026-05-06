"""
Insight Report Tool for IBM watsonx Orchestrate.
Enriches raw analyst insights into category-level structured reports per persona.
"""

import os
import json
import requests
import time
from ibm_watsonx_orchestrate.agent_builder.tools import tool
from pathlib import Path


from dotenv import load_dotenv
load_dotenv("/Users/sedadulger/Visual studio codes/final-branch/.env")


SYSTEM_PROMPT = """You are a senior consumer insights analyst for Visa Switzerland.
You receive raw survey responses and verdicts for ONE persona and must produce a structured JSON enrichment.

Return ONLY a valid JSON object with this exact structure — no markdown, no explanation:
{
  "key_metrics": {
    "max_monthly_fee_chf": "CHF X",
    "top_feature": "...",
    "switching_intent": "high | medium | low",
    "twint_loyalty": "...",
    "welcome_bonus_effectiveness": "..."
  },
  "category_insights": {
    "price_sensitivity": {
      "finding": "One clear analytical sentence about this persona's price sensitivity.",
      "key_quote": "Most revealing quote from their response.",
      "tags": ["tag1", "tag2", "tag3"]
    },
    "feature_preference": {
      "finding": "...",
      "key_quote": "...",
      "tags": ["...", "..."]
    },
    "behavior_switching": {
      "finding": "...",
      "key_quote": "...",
      "tags": ["...", "..."]
    },
    "trade_off": {
      "finding": "...",
      "key_quote": "...",
      "tags": ["...", "..."]
    },
    "willingness_to_pay": {
      "finding": "...",
      "key_quote": "...",
      "tags": ["...", "..."]
    }
  },
  "visa_recommendation": {
    "viable_product": "...",
    "avoid": ["...", "..."],
    "acquisition_trigger": "..."
  }
}"""

PERSONA_METADATA = {
    "Felix":  {"segment": "Student",      "fee_ceiling": "CHF 5",     "population_pct": 8},
    "Lukas":  {"segment": "Professional", "fee_ceiling": "CHF 25-30", "population_pct": 12},
    "Anna":   {"segment": "Mittelstand",  "fee_ceiling": "CHF 12",    "population_pct": 28},
    "Sandra": {"segment": "Family",       "fee_ceiling": "CHF 10",    "population_pct": 22},
    "Peter":  {"segment": "Retired",      "fee_ceiling": "CHF 5",     "population_pct": 18},
}

_iam_token_cache = None
_iam_token_expiry = 0


def _get_iam_token() -> str:
    global _iam_token_cache, _iam_token_expiry
    if not _iam_token_cache or time.time() >= _iam_token_expiry:
        resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                  "apikey": os.environ["WATSONX_API_KEY"]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        _iam_token_cache = data["access_token"]
        _iam_token_expiry = time.time() + data.get("expires_in", 3600) - 60
    return _iam_token_cache


def _call_watsonx(prompt: str) -> str:
    token = _get_iam_token()
    url = os.environ["WATSONX_URL"].rstrip("/")
    project_id = os.environ["WATSONX_PROJECT_ID"]
    model_id = os.environ.get("WATSONX_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

    full_prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"

    payload = {
        "model_id": model_id,
        "project_id": project_id,
        "input": full_prompt,
        "parameters": {
            "max_new_tokens": 2000,
            "temperature": 0.3,
            "repetition_penalty": 1.1,
            "stop_sequences": ["</s>", "[/INST]"],
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


@tool
def generate_insight_report(persona_name: str, persona_responses_json: str, raw_insights_json: str) -> str:
    """
    Generates an enriched category-level insight report for a single Swiss consumer persona.

    Args:
        persona_name: The persona name. Must be one of: Felix, Lukas, Anna, Sandra, Peter.
        persona_responses_json: A JSON string containing the persona's survey responses as a list of objects with fields: question, answer, category.
        raw_insights_json: A JSON string containing the raw analyst insights from the analyse_persona_responses tool.

    Returns:
        A structured JSON insight report with key_metrics, category_insights, and visa_recommendation for the given persona.
    """
    valid_personas = {"Felix", "Lukas", "Anna", "Sandra", "Peter"}
    if persona_name not in valid_personas:
        return f"Error: Invalid persona '{persona_name}'. Must be one of: {sorted(valid_personas)}"

    try:
        persona_responses = json.loads(persona_responses_json)
        raw_insights = json.loads(raw_insights_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input: {str(e)}"

    meta = PERSONA_METADATA[persona_name]

    responses_text = ""
    for r in persona_responses[:5]:
        responses_text += f"\n[{r.get('category', '')}] Q: {r.get('question', '')}\nA: {r.get('answer', '')}\n"

    prompt = (
        f"Persona: {persona_name}\n"
        f"Segment: {meta['segment']}\n"
        f"Fee ceiling: {meta['fee_ceiling']}\n"
        f"Population: {meta['population_pct']}% of Swiss adults\n\n"
        f"Survey responses:\n{responses_text}\n\n"
        f"Raw verdicts:\n{json.dumps(raw_insights, indent=2)}\n\n"
        f"Produce the enriched insight JSON for {persona_name} only."
    )

    try:
        raw = _call_watsonx(prompt)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1:
            return f"Error: No JSON found in response."
        return raw[start:end]
    except Exception as e:
        return f"Error generating insight report: {str(e)}"
