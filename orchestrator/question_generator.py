"""
Question Generator Agent
PROJECT_WATSONX — Multi-Agent Synthetic Researcher

Dynamically generates survey questions based on a given topic.
Questions are the same for all personas — only the topic drives generation.
Called by the Orchestrator before running persona agents.

Usage (standalone):
    python3 orchestrator/question_generator.py --topic "Visa card pricing sensitivity"
    python3 orchestrator/question_generator.py --topic "Mobile payment adoption" --num 5

Usage (from Orchestrator):
    from orchestrator.question_generator import generate_questions
    questions, tokens = generate_questions(topic="Visa card adoption", num_questions=5)
"""

import os
import sys
import json
import argparse
import requests
import time
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "config.json"
SURVEY_DIR  = BASE_DIR / "survey"
SURVEY_DIR.mkdir(exist_ok=True)

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

# ── Load config ───────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

# ── IAM Token ─────────────────────────────────────────────────────────────────

_iam_token_cache = None
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

def call_watsonx(prompt: str, config: dict) -> str:
    token      = get_iam_token()
    url        = os.environ["WATSONX_URL"].rstrip("/")
    project_id = os.environ["WATSONX_PROJECT_ID"]
    model_id   = os.environ.get(
        "WATSONX_MODEL_ID",
        config.get("question_generator", {}).get("model_id", "meta-llama/llama-3-3-70b-instruct")
    )
    max_tokens  = config.get("question_generator", {}).get("max_new_tokens", 500)
    temperature = config.get("question_generator", {}).get("temperature", 0.7)

    payload = {
        "model_id":   model_id,
        "project_id": project_id,
        "input":      prompt,
        "parameters": {
            "max_new_tokens":     max_tokens,
            "temperature":        temperature,
            "repetition_penalty": 1.1,
            "stop_sequences":     ["</s>", "[/INST]"],
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
            timeout=60,
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


def call_mock(prompt: str, config: dict) -> str:
    return json.dumps([
        {"id": "Q1", "category": "price_sensitivity",  "text": "[MOCK] Would you pay CHF 5/month for a card with cashback on groceries?"},
        {"id": "Q2", "category": "feature_preference",  "text": "[MOCK] Which feature matters most: cashback, travel insurance, or no foreign fees?"},
        {"id": "Q3", "category": "behavior_switching",  "text": "[MOCK] What would make you switch from Twint to a Visa card?"},
        {"id": "Q4", "category": "trade_off",           "text": "[MOCK] Is CHF 50 welcome bonus worth a CHF 72 yearly fee to you?"},
        {"id": "Q5", "category": "willingness_to_pay",  "text": "[MOCK] What is the maximum monthly fee you would accept for a Visa card?"},
    ])

# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_generation_prompt(topic: str, num_questions: int) -> str:
    return f"""<s>[INST] <<SYS>>
You are an expert market research consultant designing a consumer survey for Visa card products in Switzerland.
Your task is to generate survey questions that will be asked to synthetic Swiss consumer personas to understand their payment behavior and attitudes.

Rules:
- Questions must be relevant to the topic provided
- Questions must be practical and realistic for Swiss consumers
- Questions should cover different angles: price sensitivity, features, behavior, trade-offs, willingness to pay
- Each question should be specific enough to reveal meaningful differences between consumer segments
- Do NOT use generic questions — make them specific to Switzerland and the topic
- Return ONLY a valid JSON array, no explanation, no markdown, no extra text
- Questions must be relevant to ALL Swiss consumer segments — students, families, professionals, retirees and median earners
- Avoid questions about interest rates, cash advances, or credit features — Swiss consumers pay in full
- Focus on: monthly fees, cashback on groceries, Twint vs card switching, feature preferences, willingness to pay
- Questions should reveal differences between segments, not confuse them
- Each question must be answerable by ALL 5 personas without confusion
- Never include persona names or character names in parentheses in any question
- All questions must be written in second person (you/your) and applicable to any Swiss consumer
- Never use vague terms like "additional rewards" — be specific

JSON format:
[
  {{
    "id": "Q1",
    "category": "price_sensitivity",
    "text": "question text here"
  }},
  ...
]

Categories to use (pick the most relevant for each question):
- price_sensitivity
- feature_preference
- behavior_switching
- trade_off
- willingness_to_pay
<</SYS>>

Generate exactly {num_questions} survey questions about the following topic:

TOPIC: {topic}

Return ONLY the JSON array with {num_questions} questions. [/INST]"""

# ── Parse LLM Response ────────────────────────────────────────────────────────

def parse_questions(raw: str, num_questions: int) -> list:
    raw   = raw.strip()
    start = raw.find("[")
    end   = raw.rfind("]") + 1

    if start == -1 or end == 0:
        raise ValueError(f"No JSON array found in LLM response:\n{raw}")

    json_str = raw[start:end]

    try:
        questions = json.loads(json_str)
    except json.JSONDecodeError:
        last = max(json_str.rfind("},"), json_str.rfind("}"))
        if last == -1:
            raise ValueError(f"Could not parse questions JSON:\n{json_str[:200]}")
        json_str = json_str[:last + 1] + "]"
        questions = json.loads(json_str)

    validated = []
    for i, q in enumerate(questions, 1):
        if not isinstance(q, dict):
            continue
        validated.append({
            "id":       q.get("id", f"Q{i}"),
            "category": q.get("category", "general"),
            "text":     q.get("text", q.get("question", "")),
        })

    if not validated:
        raise ValueError("No valid questions parsed from LLM response")

    return validated[:num_questions]

# ── Main Generator ────────────────────────────────────────────────────────────

def generate_questions(
    topic:         str,
    num_questions: int  = None,
    provider:      str  = "watsonx",
    save:          bool = True,
) -> list:
    config = load_config()

    if num_questions is None:
        num_questions = config.get("survey_config", {}).get("num_questions", 5)

    print(f"\n  Topic        : {topic}")
    print(f"  Questions    : {num_questions}")
    print(f"  Provider     : {provider}")

    prompt = build_generation_prompt(topic, num_questions)

    print(f"  Generating...")
    if provider == "mock":
        raw = call_mock(prompt, config)
    else:
        raw = call_watsonx(prompt, config)

    questions = parse_questions(raw, num_questions)
    print(f"  ✓ Generated {len(questions)} questions")

    for q in questions:
        print(f"\n  [{q['id']}] {q['category']}")
        print(f"  Q: {q['text']}")

    if save:
        survey_data = {
            "survey_title":    f"Survey — {topic}",
            "generated_topic": topic,
            "generated_at":    datetime.now().isoformat(),
            "provider":        provider,
            "total_questions": len(questions),
            "questions":       questions,
        }
        out_path = SURVEY_DIR / f"survey_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(survey_data, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Saved → survey/survey_questions.json")

    return questions

# ── Token summary helper ──────────────────────────────────────────────────────

def get_token_usage() -> dict:
    total_in  = sum(r["input_tokens"]  for r in _token_log)
    total_out = sum(r["output_tokens"] for r in _token_log)
    return {"input_tokens": total_in, "output_tokens": total_out, "total_tokens": total_in + total_out}

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Question Generator Agent")
    parser.add_argument("--topic",    type=str, required=True)
    parser.add_argument("--num",      type=int, default=None)
    parser.add_argument("--provider", choices=["watsonx", "mock"], default="watsonx")
    parser.add_argument("--no-save",  action="store_true")
    args = parser.parse_args()

    questions = generate_questions(
        topic         = args.topic,
        num_questions = args.num,
        provider      = args.provider,
        save          = not args.no_save,
    )
    print(f"\n✓ Done — {len(questions)} questions generated.")


if __name__ == "__main__":
    main()
