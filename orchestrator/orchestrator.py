"""
Orchestrator Agent
PROJECT_WATSONX — Multi-Agent Synthetic Researcher

The Orchestrator coordinates the full pipeline:
    1. Runs all persona survey agents (via run_survey.py)
    2. Runs the evaluation suite (consistency, confidence, differentiation)
    3. Calls the Analyst Agent to generate insight JSONs
    4. Saves a consolidated final report

Usage:
    python3 orchestrator/orchestrator.py --provider watsonx
    python3 orchestrator/orchestrator.py --provider mock
    python3 orchestrator/orchestrator.py --provider watsonx --personas student family
"""

import os
import sys
import json
import argparse
import subprocess
from langfuse import Langfuse
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from question_generator import generate_questions
import time

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent.parent
SCRIPTS_DIR    = BASE_DIR / "scripts"
EVAL_DIR       = BASE_DIR / "evaluation" / "scripts"
INSIGHTS_DIR   = BASE_DIR / "insights"
RESPONSES_DIR  = BASE_DIR / "responses"
REPORT_DIR     = BASE_DIR / "orchestrator" / "reports"

INSIGHTS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

ALL_PERSONAS = ["student", "professional", "mittelstand", "family", "retired"]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def get_latest_response(persona: str) -> Path | None:
    """Get the most recent response file for a given persona."""
    files = sorted(RESPONSES_DIR.glob(f"{persona}_*.json"), reverse=True)
    return files[0] if files else None


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved → {path.relative_to(BASE_DIR)}")

# ── Step 1: Run Persona Survey Agents ─────────────────────────────────────────

def run_surveys(personas: list, provider: str) -> dict:
    """
    Runs run_survey.py for each persona as a subprocess.
    Returns a dict of persona → response file path.
    """
    section("STEP 1 — RUNNING PERSONA SURVEY AGENTS")
    response_files = {}

    for persona in personas:
        print(f"\n  → Running persona: {persona.upper()}")
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "run_survey.py"),
             "--persona", persona,
             "--provider", provider],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ✗ Error running {persona}")
            continue

        response_file = get_latest_response(persona)
        if response_file:
            response_files[persona] = response_file
            print(f"  ✓ {persona} response saved → {response_file.name}")
        else:
            print(f"  ✗ No response file found for {persona}")

    return response_files

# ── Step 2: Run Evaluation Suite ──────────────────────────────────────────────

def run_evaluation() -> dict:
    """
    Runs the full evaluation suite (consistency, confidence, differentiation).
    Returns the evaluation summary.
    """
    section("STEP 2 — RUNNING EVALUATION SUITE")

    result = subprocess.run(
        [sys.executable, str(EVAL_DIR / "run_evaluation.py")],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print("  ✗ Evaluation failed")
        return {}

    results_dir = BASE_DIR / "evaluation" / "results"
    summaries   = sorted(results_dir.glob("evaluation_summary_*.json"), reverse=True)

    if summaries:
        summary = load_json(summaries[0])
        print(f"  ✓ Evaluation complete → {summaries[0].name}")
        return summary

    return {}

# ── Step 3a: Run Analyst Agent ────────────────────────────────────────────────

def run_analyst(response_files: dict, provider: str = "watsonx") -> dict:
    section("STEP 3a — RUNNING ANALYST AGENT")

    insights       = {}
    analyst_script = BASE_DIR / "analyst" / "analyst_agent.py"

    if analyst_script.exists():
        sys.path.insert(0, str(BASE_DIR))
        from analyst.analyst_agent import run_analyst as _run_analyst
        insights = _run_analyst(response_files, provider=provider)
        print(f"  ✓ Analyst complete — {len(insights)} questions analysed")
    else:
        print("  Analyst Agent not found — skipping insight generation")
        print("  Expected path:", analyst_script)

    return insights

# ── Step 3b: Run Insight Report ───────────────────────────────────────────────

def run_insight_report(provider: str) -> dict:
    section("STEP 3b — GENERATING INSIGHT REPORT PER CATEGORY")

    insight_script = BASE_DIR / "analyst" / "insights_report.py"

    if not insight_script.exists():
        print(f"  ✗ insights_report.py not found — skipping")
        print(f"  Expected path: {insight_script}")
        return {}

    insights_files = sorted(INSIGHTS_DIR.glob("insights_*.json"), reverse=True)
    if not insights_files:
        print("  ✗ No insights file found — run analyst first")
        return {}

    latest_insights = insights_files[0]
    print(f"  Using: {latest_insights.name}")

    sys.path.insert(0, str(BASE_DIR / "analyst"))
    from insights_report import enrich_insights

    try:
        enriched = enrich_insights(latest_insights, provider=provider)
        print(f"  ✓ Insight report complete — {len(enriched.get('personas', {}))} personas enriched")
        return enriched
    except Exception as e:
        print(f"  ✗ Insight report failed: {e}")
        return {}


# ── Step 4: Generate Consolidated Report ──────────────────────────────────────

def generate_report(
    personas:        list,
    provider:        str,
    response_files:  dict,
    eval_summary:    dict,
    insights:        dict,
    insight_report:  dict,
    elapsed:         float = 0.0,
    token_breakdown: dict  = None,
) -> Path:
    """
    Combines all outputs into a single consolidated report JSON.
    """
    section("STEP 4 — GENERATING CONSOLIDATED REPORT")

    # Collect flagged personas from all categories
    flagged = eval_summary.get("flagged_for_review", {})
    all_flagged = set(
        flagged.get("low_confidence", []) +
        flagged.get("low_consistency", []) +
        flagged.get("low_differentiation", [])
    )

    # Collect per-persona summary
    persona_summaries = {}
    for persona in personas:
        response_file = response_files.get(persona)
        persona_summaries[persona] = {
            "response_file":     str(response_file.name) if response_file else None,
            "consistency_score": eval_summary.get("consistency", {}).get(persona),
            "confidence_score":  eval_summary.get("confidence",  {}).get(persona),
            "needs_review":      persona in all_flagged,
            "insight_generated": len(insights) > 0,
        }

    report = {
        "timestamp": TIMESTAMP,
        "bias_indicators": {
             "differentiation_score": eval_summary.get("differentiation", 0.0),
             "persona_fee_ceilings_aligned": True,
             "flagged_personas": eval_summary.get("flagged_for_review", {}).get("low_confidence", []),
             "mitigation": "RAG-grounded with real Swiss demographic data (SNB 2024, BFS 2024, ZHAW 2025)",
        },

        "provider": provider,
        "performance": {
            "duration_seconds": elapsed,
            "duration_minutes": round(elapsed / 60, 1),
        },
        
        "token_usage": token_breakdown or {},
        "personas_run":          personas,
        "differentiation_score": eval_summary.get("differentiation"),
        "flagged_for_review":    eval_summary.get("flagged_for_review", {}),
        "persona_summaries":     persona_summaries,
        "pipeline_steps": {
            "surveys_completed":   list(response_files.keys()),
            "evaluation_complete": bool(eval_summary),
            "insights_generated":  list(insights.keys()),
        },
    }

    report_path = REPORT_DIR / f"final_report_{TIMESTAMP}.json"
    save_json(report, report_path)

    # Print summary to terminal
    print(f"\n  {'─' * 50}")
    print(f"  PIPELINE SUMMARY")
    print(f"  {'─' * 50}")
    print(f"  Personas run       : {', '.join(personas)}")
    print(f"  Duration           : {round(elapsed / 60, 1)} min ({elapsed}s)")
    print(f"  Differentiation    : {eval_summary.get('differentiation', 'N/A')}")
    print(f"  Flagged for review : {list(all_flagged) or 'none'}")
    for persona, s in persona_summaries.items():
        consistency = s['consistency_score']
        confidence  = s['confidence_score']
        review_flag = '⚠' if s['needs_review'] else '✓'
        print(f"  {review_flag} {persona:<14} consistency={consistency}  confidence={confidence}")
    print(f"  {'─' * 50}")

    return report_path

# ── Main Orchestrator ─────────────────────────────────────────────────────────

def main():

    # Load .env first
    from evaluation_utils import load_env
    load_env()
    
    # Then init Langfuse
    global langfuse
    langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST")
    )

    parser = argparse.ArgumentParser(description="Orchestrator — runs the full multi-agent pipeline")
    parser.add_argument(
        "--topic",
        type=str,
        default="Visa card adoption and payment behavior in Switzerland",
        help="Survey topic for question generation"
    )
    parser.add_argument(
        "--provider",
        choices=["watsonx", "mock"],
        default="watsonx",
        help="LLM provider to use (default: watsonx)"
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        choices=ALL_PERSONAS,
        default=ALL_PERSONAS,
        help="Which personas to run (default: all)"
    )
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  ORCHESTRATOR — Multi-Agent Synthetic Researcher")
    print(f"  Provider : {args.provider}")
    print(f"  Topic    : {args.topic}")
    print(f"  Personas : {', '.join(args.personas)}")
    print(f"  Time     : {TIMESTAMP}")
    print(f"{'═' * 60}")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline_start = time.time()
    section("STEP 0 — GENERATING SURVEY QUESTIONS")
    print(f"\n  Topic: {args.topic}")

    trace = langfuse.trace(
        name="orchestrator_pipeline",
        metadata={"provider": args.provider, "personas": args.personas, "topic": args.topic}
    )

    questions = generate_questions(
        topic    = args.topic,
        provider = args.provider,
        save     = True,
    )
    print(f"  ✓ {len(questions)} questions ready for all personas")

    response_files = run_surveys(args.personas, args.provider)

    if not response_files:
        print("\n✗ No responses generated — stopping pipeline.")
        sys.exit(1)

    eval_summary   = run_evaluation()
    insights       = run_analyst(response_files, provider=args.provider)
    insight_report = run_insight_report(provider=args.provider)

    elapsed = round(time.time() - pipeline_start, 1)

    langfuse.score(trace_id=trace.id, name="pipeline_duration_seconds", value=elapsed)
    langfuse.score(trace_id=trace.id, name="personas_completed", value=len(response_files))

    # ── Collect token breakdown from all steps ────────────────────────────
    from question_generator import get_token_usage as qg_tokens
    from analyst.analyst_agent import get_token_usage as analyst_tokens
    sys.path.insert(0, str(BASE_DIR / "analyst"))
    from insights_report import get_token_usage as insight_tokens

    token_breakdown = {
        "question_generation": qg_tokens(),
        "analyst":             analyst_tokens(),
        "insights_report":     insight_tokens(),
    }
    # Survey tokens come from response files (saved by run_survey.py)
    survey_input = survey_output = 0
    for rf in response_files.values():
        try:
            d = load_json(rf)
            tu = d.get("token_usage", {})
            survey_input  += tu.get("input_tokens", 0)
            survey_output += tu.get("output_tokens", 0)
        except Exception:
            pass
    token_breakdown["survey"] = {
        "input_tokens":  survey_input,
        "output_tokens": survey_output,
        "total_tokens":  survey_input + survey_output,
    }
    # Grand total
    total_in  = sum(v.get("input_tokens", 0)  for v in token_breakdown.values())
    total_out = sum(v.get("output_tokens", 0) for v in token_breakdown.values())
    token_breakdown["total"] = {
        "input_tokens":  total_in,
        "output_tokens": total_out,
        "total_tokens":  total_in + total_out,
    }

    report_path    = generate_report(
        personas        = args.personas,
        provider        = args.provider,
        response_files  = response_files,
        eval_summary    = eval_summary,
        insights        = insights,
        insight_report  = insight_report,
        elapsed         = elapsed,
        token_breakdown = token_breakdown,
    )

    langfuse.score(
        trace_id=trace.id,
        name="differentiation_score",
        value=eval_summary.get("differentiation") or 0.0
    )
    langfuse.flush()
    print("✓ Langfuse trace completed")

    print(f"\n✓ Pipeline complete → {report_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
