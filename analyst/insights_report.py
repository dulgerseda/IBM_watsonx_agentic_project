"""
Insights Enricher
PROJECT_WATSONX — Multi-Agent Synthetic Researcher

Reads raw insights JSON and produces a richer structured JSON
with category_insights, persona behaviors, key metrics, and recommendations.

Usage:
    python3 analyst/insights_report.py
    python3 analyst/insights_report.py --input insights/insights_visa_survey_20260429.json
    python3 analyst/insights_report.py --provider mock
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

BASE_DIR     = Path(__file__).parent.parent
INSIGHTS_DIR = BASE_DIR / "insights"
INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Token tracking ────────────────────────────────────────────────────────────
_token_log = []

# ── Load .env ─────────────────────────────────────────────────────────────────

def load_env():
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
    "Felix":  {"segment": "Student",      "fee_ceiling": "CHF 5",     "population_pct": 8,  "key": "student"},
    "Lukas":  {"segment": "Professional", "fee_ceiling": "CHF 25-30", "population_pct": 12, "key": "professional"},
    "Anna":   {"segment": "Mittelstand",  "fee_ceiling": "CHF 12",    "population_pct": 28, "key": "mittelstand"},
    "Sandra": {"segment": "Family",       "fee_ceiling": "CHF 10",    "population_pct": 22, "key": "family"},
    "Peter":  {"segment": "Retired",      "fee_ceiling": "CHF 5",     "population_pct": 18, "key": "retired"},
}

# ── System prompt ─────────────────────────────────────────────────────────────

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

    full_prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"

    payload = {
        "model_id":   model_id,
        "project_id": project_id,
        "input":      full_prompt,
        "parameters": {
            "max_new_tokens":     2000,
            "temperature":        0.3,
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


# ── Mock templates (unchanged) ────────────────────────────────────────────────

MOCK_TEMPLATES = {
    "Felix":  {"key_metrics": {"max_monthly_fee_chf": "CHF 5",     "top_feature": "No annual fee", "switching_intent": "low",    "twint_loyalty": "high",   "welcome_bonus_effectiveness": "medium"}, "category_insights": {"price_sensitivity": {"finding": "Fee above CHF 5 is immediate barrier.", "key_quote": "I don't pay fees when Twint is free.", "tags": ["fee-free preferred"]}, "feature_preference": {"finding": "Simplicity over features.", "key_quote": "Just something that works.", "tags": ["simplicity first"]}, "behavior_switching": {"finding": "Low — Twint covers all needs.", "key_quote": "I'd only switch for real benefit.", "tags": ["Twint loyal"]}, "trade_off": {"finding": "Rejects bonus if annual fee exceeds CHF 60.", "key_quote": "Fee stays forever, bonus disappears.", "tags": ["long-term cost focus"]}, "willingness_to_pay": {"finding": "Max CHF 5/month.", "key_quote": "Maybe CHF 5 if I get something back.", "tags": ["CHF 5 ceiling"]}}, "visa_recommendation": {"viable_product": "Free student Visa card with 1% cashback", "avoid": ["annual fees above CHF 60"], "acquisition_trigger": "Student bank partnership"}},
    "Lukas":  {"key_metrics": {"max_monthly_fee_chf": "CHF 25-30", "top_feature": "No foreign fees + travel insurance", "switching_intent": "high",   "twint_loyalty": "medium", "welcome_bonus_effectiveness": "low"},    "category_insights": {"price_sensitivity": {"finding": "Value-driven not price-sensitive.", "key_quote": "The math clearly favors zero foreign fees.", "tags": ["ROI focused"]}, "feature_preference": {"finding": "No foreign fees + travel insurance wins.", "key_quote": "Foreign fees have continuous value.", "tags": ["travel-first"]}, "behavior_switching": {"finding": "Will switch if rewards are frictionless.", "key_quote": "If I see monthly net gains I would switch.", "tags": ["high switching readiness"]}, "trade_off": {"finding": "Accepts net gain instantly.", "key_quote": "That is a clear net gain.", "tags": ["rejects one-time bonuses"]}, "willingness_to_pay": {"finding": "CHF 25-30 ceiling.", "key_quote": "Up to CHF 25-30 if value justifies it.", "tags": ["premium tier"]}}, "visa_recommendation": {"viable_product": "Premium Visa — CHF 20-25/month with travel insurance", "avoid": ["points systems"], "acquisition_trigger": "Automatic value delivery"}},
    "Anna":   {"key_metrics": {"max_monthly_fee_chf": "CHF 12",    "top_feature": "Grocery cashback", "switching_intent": "medium", "twint_loyalty": "high",   "welcome_bonus_effectiveness": "medium"}, "category_insights": {"price_sensitivity": {"finding": "Needs clear monthly savings before committing.", "key_quote": "I need to calculate if cashback covers the fee.", "tags": ["budget-first"]}, "feature_preference": {"finding": "Grocery cashback is primary motivator.", "key_quote": "I spend a lot at Migros. Cashback adds up.", "tags": ["grocery cashback"]}, "behavior_switching": {"finding": "Would switch if trial proves value.", "key_quote": "I'd want to try it for a month first.", "tags": ["trial-first"]}, "trade_off": {"finding": "Accepts CHF 8 if cashback consistently exceeds fee.", "key_quote": "If I save CHF 15 and pay CHF 8 that makes sense.", "tags": ["positive ROI required"]}, "willingness_to_pay": {"finding": "CHF 8-12 if grocery cashback delivers.", "key_quote": "Up to CHF 12 if cashback really works.", "tags": ["CHF 8-12 range"]}}, "visa_recommendation": {"viable_product": "Mass-market Visa — CHF 8/month with 1% grocery cashback", "avoid": ["travel perks"], "acquisition_trigger": "30-day free trial"}},
    "Sandra": {"key_metrics": {"max_monthly_fee_chf": "CHF 10",    "top_feature": "Family purchase protection + grocery cashback", "switching_intent": "medium", "twint_loyalty": "medium", "welcome_bonus_effectiveness": "high"},   "category_insights": {"price_sensitivity": {"finding": "Joint household decision required.", "key_quote": "I need to discuss with my partner first.", "tags": ["joint decision"]}, "feature_preference": {"finding": "Purchase protection equally valued as cashback.", "key_quote": "When buying appliances protection matters.", "tags": ["purchase protection"]}, "behavior_switching": {"finding": "Would switch for family-wide card.", "key_quote": "If we could share the cashback that would be worth it.", "tags": ["family plan preferred"]}, "trade_off": {"finding": "Needs both cashback and protection for CHF 8.", "key_quote": "The cashback alone might not be enough.", "tags": ["dual feature requirement"]}, "willingness_to_pay": {"finding": "CHF 8-10 if family value is clear.", "key_quote": "Ten francs is reasonable if we save more.", "tags": ["CHF 10 ceiling"]}}, "visa_recommendation": {"viable_product": "Family Visa — CHF 8/month with cashback + protection", "avoid": ["individual-only positioning"], "acquisition_trigger": "Family plan with shared cashback pool"}},
    "Peter":  {"key_metrics": {"max_monthly_fee_chf": "CHF 5",     "top_feature": "Simple interface + fraud protection", "switching_intent": "low",    "twint_loyalty": "low",    "welcome_bonus_effectiveness": "low"},    "category_insights": {"price_sensitivity": {"finding": "Fixed income — any fee above CHF 5 rejected.", "key_quote": "I live on a pension. Every franc counts.", "tags": ["CHF 5 hard ceiling"]}, "feature_preference": {"finding": "Security and simplicity override all features.", "key_quote": "I need to know my money is safe.", "tags": ["security first"]}, "behavior_switching": {"finding": "Very low — habits deeply ingrained.", "key_quote": "I've used the same bank for 30 years.", "tags": ["habit-driven"]}, "trade_off": {"finding": "No trade-off above CHF 5.", "key_quote": "A bonus means nothing if I don't trust the product.", "tags": ["trust barrier dominant"]}, "willingness_to_pay": {"finding": "Max CHF 5 via trusted bank.", "key_quote": "If my bank offered it simply I might consider five francs.", "tags": ["CHF 5 ceiling"]}}, "visa_recommendation": {"viable_product": "Basic Visa debit — no fee or max CHF 5 via bank partner", "avoid": ["premium positioning"], "acquisition_trigger": "Bank partnership model"}},
}


def call_mock(persona_name: str) -> str:
    template = MOCK_TEMPLATES.get(persona_name, MOCK_TEMPLATES["Anna"])
    return json.dumps(template)

# ── Build prompt ──────────────────────────────────────────────────────────────

def build_prompt(persona_name: str, persona_responses: list, raw_insights: dict) -> str:
    meta = PERSONA_METADATA.get(persona_name, {})
    responses_text = ""
    for r in persona_responses[:5]:
        q   = r.get("question", "")
        ans = r.get("answer", "")
        cat = r.get("category", "")
        responses_text += f"\n[{cat}] Q: {q}\nA: {ans}\n"

    verdicts = {}
    for q_id, insight in raw_insights.items():
        seg = insight.get("segment_responses", {}).get(persona_name, {})
        if seg:
            verdicts[q_id] = seg

    return (
        f"Persona: {persona_name}\n"
        f"Segment: {meta.get('segment', '')}\n"
        f"Fee ceiling: {meta.get('fee_ceiling', '')}\n"
        f"Population: {meta.get('population_pct', '')}% of Swiss adults\n\n"
        f"Survey responses:\n{responses_text}\n\n"
        f"Raw verdicts:\n{json.dumps(verdicts, indent=2)}\n\n"
        f"Produce the enriched insight JSON for {persona_name} only."
    )

# ── Token summary helper ──────────────────────────────────────────────────────

def get_token_usage() -> dict:
    total_in  = sum(r["input_tokens"]  for r in _token_log)
    total_out = sum(r["output_tokens"] for r in _token_log)
    return {"input_tokens": total_in, "output_tokens": total_out, "total_tokens": total_in + total_out}

# ── Main enricher ─────────────────────────────────────────────────────────────

def enrich_insights(insights_path: Path, provider: str = "watsonx") -> dict:
    with open(insights_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_insights  = data.get("insights", {})
    survey_run_id = data.get("survey_run_id", "unknown")
    responses_dir = BASE_DIR / "responses"

    enriched = {
        "survey_run_id": survey_run_id,
        "generated_at":  datetime.now().isoformat(),
        "provider":      provider,
        "personas":      {}
    }

    for persona_name, meta in PERSONA_METADATA.items():
        print(f"\n  → Enriching {persona_name} ({meta['segment']})...")

        persona_key       = meta["key"]
        files             = sorted(responses_dir.glob(f"{persona_key}_*.json"), reverse=True)
        persona_responses = []
        if files:
            with open(files[0], encoding="utf-8") as f:
                resp_data         = json.load(f)
                persona_responses = resp_data.get("results", [])

        try:
            if provider == "mock":
                raw = call_mock(persona_name)
            else:
                prompt = build_prompt(persona_name, persona_responses, raw_insights)
                raw    = call_watsonx(prompt)

            raw    = raw.strip()
            start  = raw.find("{")
            end    = raw.rfind("}") + 1
            parsed = json.loads(raw[start:end])

            enriched["personas"][persona_name] = parsed
            print(f"  ✓ {persona_name} enriched")

        except Exception as e:
            print(f"  ✗ Failed to enrich {persona_name}: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = INSIGHTS_DIR / f"insight_report_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Saved → insights/insight_report_{timestamp}.json")
    return enriched

# ── CLI ───────────────────────────────────────────────────────────────────────

def get_latest_insights() -> Path | None:
    files = sorted(INSIGHTS_DIR.glob("insights_*.json"), reverse=True)
    return files[0] if files else None


def main():
    parser = argparse.ArgumentParser(description="Enrich insights JSON")
    parser.add_argument("--input",    type=str, default=None)
    parser.add_argument("--provider", choices=["watsonx", "mock"], default="watsonx")
    args = parser.parse_args()

    insights_path = Path(args.input) if args.input else get_latest_insights()
    if not insights_path:
        print("✗ No insights file found.")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  INSIGHTS ENRICHER")
    print(f"  Input    : {insights_path.name}")
    print(f"  Provider : {args.provider}")
    print(f"{'═' * 60}")

    enrich_insights(insights_path, args.provider)
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
