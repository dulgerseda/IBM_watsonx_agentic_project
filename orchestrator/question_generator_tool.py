"""
Question Generator Tool for IBM watsonx Orchestrate.
Generates survey questions based on a given topic.
"""

import os
import json
import requests
import time
from ibm_watsonx_orchestrate.agent_builder.tools import tool
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("/Users/sedadulger/Visual studio codes/final-branch/.env")

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

    payload = {
        "model_id": model_id,
        "project_id": project_id,
        "input": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
            "stop_sequences": ["</s>", "[/INST]"],
        },
    }

    for attempt in range(5):
        resp = requests.post(
            f"{url}/ml/v1/text/generation?version=2023-05-29",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if resp.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()["results"][0]["generated_text"].strip()
    resp.raise_for_status()


def _build_prompt(topic: str, num_questions: int) -> str:
    return f"""<s>[INST] <<SYS>>
You are an expert market research consultant designing a consumer survey for Visa card products in Switzerland.
Generate exactly {num_questions} survey questions about the topic below.
Return ONLY a valid JSON array with fields: id, category, text.
Categories: price_sensitivity, feature_preference, behavior_switching, trade_off, willingness_to_pay.
Questions must be in second person, applicable to all Swiss consumer segments.
<</SYS>>

TOPIC: {topic}

Return ONLY the JSON array. [/INST]"""


def _parse_questions(raw: str, num_questions: int) -> list:
    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found")
    json_str = raw[start:end]
    try:
        questions = json.loads(json_str)
    except json.JSONDecodeError:
        last = json_str.rfind("}")
        json_str = json_str[:last + 1] + "]"
        questions = json.loads(json_str)
    return [{"id": q.get("id", f"Q{i}"), "category": q.get("category", "general"),
             "text": q.get("text", "")} for i, q in enumerate(questions, 1)][:num_questions]

@tool
def generate_survey_questions(topic: str, num_questions: int = 5) -> str:
    """
    Generates survey questions for Visa card research based on a given topic.

    Args:
        topic: The research topic to generate questions about. Example: 'Visa card pricing in Switzerland'.
        num_questions: Number of questions to generate. Defaults to 5.

    Returns:
        A JSON string containing the generated survey questions with id, category, and text fields.
    """
    try:
        prompt = _build_prompt(topic, num_questions)
        raw = _call_watsonx(prompt)
        questions = _parse_questions(raw, num_questions)
        return json.dumps(questions, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error generating questions: {str(e)}"
