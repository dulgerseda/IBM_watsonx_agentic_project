"""
Analyst Agent
PROJECT_WATSONX — Multi-Agent Synthetic Researcher

Reads persona response JSONs and produces structured insight reports
for the Visa product team. Called by the Orchestrator after all
persona agents have completed their survey runs.

Usage (standalone):
    python3 analyst/analyst_agent.py --response responses/family_20260426.json
    python3 analyst/analyst_agent.py --all
    python3 analyst/analyst_agent.py --all --provider mock

Usage (from Orchestrator):
    from analyst.analyst_agent import run_analyst
    insights = run_analyst(response_files, provider=args.provider)
"""

import os
import sys
import json
import argparse
import requests
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent.parent
RESPONSES_DIR = BASE_DIR / "responses"
INSIGHTS_DIR  = BASE_DIR / "insights"
INSIGHTS_DIR.mkdir(exist_ok=True)

# ── Token tracking ────────────────────────────────────────────────────────────
_token_log = []

# ── Load .env ─────────────────────────────────────────────────────────────────

def load_env():
    env_path = BASE_DIR.parent / ".env"
    if not env_path.exists():
        env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

load_env()

# ── Persona metadata ──────────────────────────────────────────────────────────

PERSONA_METADATA = {
    "student": {
        "persona_id":            "persona_1_student",
        "agent_label":           "Campus Felix — Student Segment",
        "segment_type":          "Student",
        "population_weight_pct": 8,
        "fee_ceiling":           "CHF 5",
    },
    "professional": {
        "persona_id":            "persona_2_professional",
        "agent_label":           "Mobile-First Lukas — Professional Segment",
        "segment_type":          "Professional",
        "population_weight_pct": 12,
        "fee_ceiling":           "CHF 25-30",
    },
    "mittelstand": {
        "persona_id":            "persona_3_mittelstand",
        "agent_label":           "Average-Earner Anna — Mittelstand Segment",
        "segment_type":          "Mittelstand",
        "population_weight_pct": 28,
        "fee_ceiling":           "CHF 12",
    },
    "family": {
        "persona_id":            "persona_4_family",
        "agent_label":           "Value-Smart Sandra — Family Segment",
        "segment_type":          "Family",
        "population_weight_pct": 22,
        "fee_ceiling":           "CHF 10",
    },
    "retired": {
        "persona_id":            "persona_5_retired",
        "agent_label":           "Trust-First Peter — Retired Segment",
        "segment_type":          "Retired",
        "population_weight_pct": 18,
        "fee_ceiling":           "CHF 5",
    },
}

# ── System prompt ─────────────────────────────────────────────────────────────

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

Return ONLY the JSON object. No explanation, no markdown, no extra text.
IMPORTANT: You MUST always return a complete valid JSON object. Never return an empty response."""

# ── IAM Token ─────────────────────────────────────────────────────────────────

_iam_token_cache  = None
_iam_token_expiry = 0

def get_iam_token() -> str:
    global _iam_token_cache, _iam_token_expiry
    if not _iam_token_cache or time.time() >= _iam_token_expiry:
        resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": os.environ["WATSONX_API_KEY"],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        _iam_token_cache  = data["access_token"]
        _iam_token_expiry = time.time() + data.get("expires_in", 3600) - 60
    return _iam_token_cache

# ── LLM Call ──────────────────────────────────────────────────────────────────

def call_watsonx(prompt: str) -> str:
    token      = get_iam_token()
    url        = os.environ["WATSONX_URL"].rstrip("/")
    project_id = os.environ["WATSONX_PROJECT_ID"]
    model_id   = os.environ.get("WATSONX_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

    full_prompt = f"<s>[INST] <<SYS>>\n{ANALYST_SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"

    payload = {
        "model_id":   model_id,
        "project_id": project_id,
        "input":      full_prompt,
        "parameters": {
            "max_new_tokens":     4000,
            "temperature":        0.2,
            "repetition_penalty": 1.1,
            "stop_sequences":     ["</s>", "[/INST]", "<|eom_id|>"],
        },
    }

    for attempt in range(5):
        resp = requests.post(
            f"{url}/ml/v1/text/generation?version=2023-05-29",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=120,
        )
        if resp.status_code == 429:
            wait = 3 * (attempt + 1)
            print(f"  Rate limit, {wait}s bekleniyor...")
            time.sleep(wait)
            continue
        print("STATUS:", resp.status_code)
        resp.raise_for_status()
        raw_response = resp.json()
        # ── Token tracking ────────────────────────────────────────────────
        r = raw_response["results"][0]
        _token_log.append({
            "input_tokens":  r.get("input_token_count", 0),
            "output_tokens": r.get("generated_token_count", 0),
        })
        # ─────────────────────────────────────────────────────────────────
        return r["generated_text"].strip()

    resp.raise_for_status()


def call_mock(prompt: str) -> str:
    return json.dumps({
        "insight_id":        "insight_mock_run",
        "generated_at":      datetime.now().isoformat(),
        "survey_run_id":     "mock_run",
        "personas_analysed": 5,
        "segment_responses": {
            "Felix":   {"verdict": "NO",    "key_reasoning": "Fee above CHF 5 ceiling", "fee_ceiling": "CHF 5"},
            "Lukas":   {"verdict": "YES",   "key_reasoning": "ROI positive at his spending level", "fee_ceiling": "CHF 25-30"},
            "Anna":    {"verdict": "MAYBE", "key_reasoning": "Needs free trial to verify value", "fee_ceiling": "CHF 12"},
            "Sandra":  {"verdict": "MAYBE", "key_reasoning": "Joint decision required with partner", "fee_ceiling": "CHF 10"},
            "Peter":   {"verdict": "NO",    "key_reasoning": "Fee above ceiling, trust barrier", "fee_ceiling": "CHF 5"},
        },
        "cross_segment_insight": "[MOCK] Sandra and Anna both prioritize grocery cashback.",
        "pricing_sweet_spot": "CHF 8 — balances mass market adoption with revenue generation",
        "top_features": ["grocery cashback", "purchase protection", "simple fee structure"],
        "main_barriers": ["fee anxiety", "joint decision required for family segment"],
        "visa_recommendation": "[MOCK] Launch at CHF 8 with 30-day free trial targeting Mittelstand and Family segments.",
        "confidence_level": "MEDIUM",
    })

# ── Format input bundle ───────────────────────────────────────────────────────

def format_input_bundle(survey_run_id: str, question: dict, responses: dict) -> str:
    response_objects = []
    missing_personas = []

    for persona_key, meta in PERSONA_METADATA.items():
        answer = responses.get(persona_key)
        if not answer:
            missing_personas.append(persona_key)
            continue
        response_objects.append({
            "persona_id":            meta["persona_id"],
            "agent_label":           meta["agent_label"],
            "segment_type":          meta["segment_type"],
            "population_weight_pct": meta["population_weight_pct"],
            "fee_ceiling":           meta["fee_ceiling"],
            "verbatim_response":     answer[:400],
        })

    bundle = {
        "survey_run_id":    survey_run_id,
        "run_mode":         "synthetic",
        "question_id":      question.get("question_id", question.get("id", "Q?")),
        "question_text":    question.get("text", ""),
        "question_type":    question.get("category", "general"),
        "responses":        response_objects,
        "missing_personas": missing_personas,
    }

    return (
        f"Analyse the following Swiss consumer survey response bundle and produce a structured insight JSON.\n\n"
        f"Response bundle:\n{json.dumps(bundle, indent=2, ensure_ascii=False)}"
    )

# ── Parse LLM response ────────────────────────────────────────────────────────

def parse_insight(raw: str) -> dict:
    raw   = raw.strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found:\n{raw[:500]}")
    # Parantez sayarak doğru sonu bul
    depth = 0
    end   = start
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(raw[start:end])

# ── Core analyst function ─────────────────────────────────────────────────────

def analyse_question(
    survey_run_id: str,
    question:      dict,
    responses:     dict,
    provider:      str = "watsonx",
) -> dict:
    prompt = format_input_bundle(survey_run_id, question, responses)
    if provider == "mock":
        raw = call_mock(prompt)
    else:
        raw = call_watsonx(prompt)
    return parse_insight(raw)

# ── Token summary helper ──────────────────────────────────────────────────────

def get_token_usage() -> dict:
    total_in  = sum(r["input_tokens"]  for r in _token_log)
    total_out = sum(r["output_tokens"] for r in _token_log)
    return {"input_tokens": total_in, "output_tokens": total_out, "total_tokens": total_in + total_out}

# ── Run analyst on response files ─────────────────────────────────────────────

def run_analyst(response_files: dict, provider: str = "watsonx") -> dict:
    if not response_files:
        print("  ✗ No response files provided to analyst")
        return {}

    persona_data = {}
    for persona_key, file_path in response_files.items():
        with open(file_path, encoding="utf-8") as f:
            persona_data[persona_key] = json.load(f)

    first_persona = next(iter(persona_data.values()))
    questions     = first_persona.get("results", [])
    survey_run_id = f"visa_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    all_insights  = {}

    print(f"\n  Analyst Agent — analysing {len(questions)} questions across {len(persona_data)} personas")

    for item in questions:
        q_id   = item.get("question_id", item.get("id", "Q?"))
        q_text = item.get("question", item.get("text", ""))
        q_cat  = item.get("category", "general")

        print(f"\n  → Analysing {q_id}...")

        responses = {}
        for persona_key, data in persona_data.items():
            for result in data.get("results", []):
                if result.get("question_id") == q_id:
                    responses[persona_key] = result.get("answer", "")
                    break

        question = {"question_id": q_id, "category": q_cat, "text": q_text}

        try:
            insight = analyse_question(survey_run_id, question, responses, provider)
            all_insights[q_id] = insight
            print(f"  ✓ {q_id} insight generated — confidence: {insight.get('confidence_level', 'N/A')}")
        except Exception as e:
            print(f"  ✗ Failed to analyse {q_id}: {e}")

    output = {
        "survey_run_id":   survey_run_id,
        "generated_at":    datetime.now().isoformat(),
        "provider":        provider,
        "total_questions": len(questions),
        "insights":        all_insights,
    }

    out_path = INSIGHTS_DIR / f"insights_{survey_run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ All insights saved → insights/insights_{survey_run_id}.json")

    return all_insights

# ── CLI ───────────────────────────────────────────────────────────────────────

def get_latest_response(persona: str) -> Path | None:
    files = sorted(RESPONSES_DIR.glob(f"{persona}_*.json"), reverse=True)
    return files[0] if files else None


def main():
    parser = argparse.ArgumentParser(description="Analyst Agent")
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--response", type=str)
    parser.add_argument("--provider", choices=["watsonx", "mock"], default="watsonx")
    args = parser.parse_args()

    if not args.all and not args.response:
        parser.error("Specify --all or --response <file>")

    if args.all:
        response_files = {}
        for persona in PERSONA_METADATA.keys():
            file = get_latest_response(persona)
            if file:
                response_files[persona] = file
        if not response_files:
            print("✗ No response files found.")
            sys.exit(1)
        run_analyst(response_files, args.provider)

    elif args.response:
        file = Path(args.response)
        if not file.exists():
            print(f"✗ File not found: {file}")
            sys.exit(1)
        persona = file.stem.split("_")[0]
        run_analyst({persona: file}, args.provider)


if __name__ == "__main__":
    main()
