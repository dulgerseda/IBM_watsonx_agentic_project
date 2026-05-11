"""
Run Evaluation
Orchestrates all evaluation scripts and saves results separately.
Usage: python evaluation/scripts/run_evaluation.py
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from langfuse import Langfuse

# Add evaluation dir to path so we can import sibling scripts
sys.path.insert(0, str(Path(__file__).parent))

from evaluation_utils import load_env, save_report
from consistency_score import score_consistency
from confidence_score import score_confidence
from differentiation_score import score_differentiation

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent.parent.parent
RESPONSES_DIR = BASE_DIR / "responses"
RESULTS_DIR   = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

PERSONAS = ["student", "professional", "family", "retired", "mittelstand"]

CONSISTENCY_THRESHOLD     = 0.6
CONFIDENCE_THRESHOLD      = 0.6
DIFFERENTIATION_THRESHOLD = 0.6

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_latest_response(persona: str) -> Path | None:
    files = sorted(RESPONSES_DIR.glob(f"{persona}_*.json"), reverse=True)
    return files[0] if files else None

def section(title: str):
    print(f"\n── {title} {'─' * (50 - len(title))}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_env()

    # ── Init Langfuse ─────────────────────────────────────────────────────────
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"]
    )
    trace = langfuse.trace(
        name="evaluation_pipeline",
    )
    print("✓ Langfuse trace started")

    # ── 1. Consistency ────────────────────────────────────────────────────────
    section("CONSISTENCY")
    consistency_report = {}

    for persona in PERSONAS:
        file = get_latest_response(persona)
        if not file:
            print(f"  ⚠ No response file found for {persona} — skipping")
            continue
        print(f"\n  {persona}...")
        result = score_consistency(file)
        consistency_report[persona] = result
        score = result["consistency_score"]
        flag  = " ⚠ FLAGGED" if score < CONSISTENCY_THRESHOLD else ""
        print(f"  ✓ {persona} consistency: {score}{flag}")

        # Log to Langfuse
        langfuse.score(
            trace_id=trace.id,
            name=f"consistency_{persona}",
            value=score,
            comment="flagged" if score < CONSISTENCY_THRESHOLD else "ok"
        )

    save_report(consistency_report, RESULTS_DIR, f"consistency_report_{TIMESTAMP}")

    # ── 2. Confidence ─────────────────────────────────────────────────────────
    section("CONFIDENCE")
    confidence_report = {}

    for persona in PERSONAS:
        file = get_latest_response(persona)
        if not file:
            print(f"  ⚠ No response file found for {persona} — skipping")
            continue
        print(f"\n  {persona}...")
        result = score_confidence(file)
        confidence_report[persona] = result
        print(f"  ✓ {persona} confidence: {result['confidence_score']}")
        if result["needs_review"]:
            print(f"  ⚠ needs_review: True")

        # Log to Langfuse
        langfuse.score(
            trace_id=trace.id,
            name=f"confidence_{persona}",
            value=result["confidence_score"],
            comment="needs_review" if result["needs_review"] else "ok"
        )

    save_report(confidence_report, RESULTS_DIR, f"confidence_report_{TIMESTAMP}")

    # ── 3. Differentiation ────────────────────────────────────────────────────
    section("DIFFERENTIATION")

    response_files = []
    for persona in PERSONAS:
        file = get_latest_response(persona)
        if file:
            response_files.append(file)

    diff_result = score_differentiation(response_files)

    if diff_result:
        print(f"\n  Personas compared: {diff_result['personas_compared']}")
        diff_score = diff_result.get("differentiation_score", 0.0)
        diff_flag  = " ⚠ FLAGGED" if diff_score < DIFFERENTIATION_THRESHOLD else ""
        print(f"  ✓ differentiation: {diff_score}{diff_flag}")

        # Log to Langfuse
        langfuse.score(
            trace_id=trace.id,
            name="differentiation",
            value=diff_score,
            comment="flagged" if diff_score < DIFFERENTIATION_THRESHOLD else "ok"
        )
    else:
        print("  ✗ Differentiation could not be computed (need at least 2 personas)")
        diff_result = {"differentiation_score": None, "personas_compared": []}

    save_report(diff_result, RESULTS_DIR, f"differentiation_report_{TIMESTAMP}")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    section("SUMMARY")

    flagged_confidence      = [p for p, r in confidence_report.items() if r.get("needs_review", False)]
    flagged_consistency     = [p for p, r in consistency_report.items() if r.get("consistency_score", 1.0) < CONSISTENCY_THRESHOLD]
    flagged_differentiation = [] if (diff_result.get("differentiation_score") or 1.0) >= DIFFERENTIATION_THRESHOLD else ["all"]

    combined_quality = {}
    for persona in consistency_report:
        c_score    = consistency_report[persona].get("consistency_score", 0.0)
        conf_score = confidence_report.get(persona, {}).get("confidence_score", 0.0)
        diff_score = diff_result.get("differentiation_score") or 0.0
        combined_quality[persona] = round((c_score + conf_score + diff_score) / 3, 3)

    summary = {
        "timestamp": TIMESTAMP,
        "consistency": {p: r["consistency_score"] for p, r in consistency_report.items()},
        "confidence":  {p: r["confidence_score"]  for p, r in confidence_report.items()},
        "differentiation": diff_result.get("differentiation_score"),
        "differentiation_note": (
            "Score based on first 5 questions only (token efficiency). "
            "Full-question differentiation not yet implemented."
        ),
        "combined_quality_score": combined_quality,
        "flagged_for_review": {
            "low_confidence":      flagged_confidence,
            "low_consistency":     flagged_consistency,
            "low_differentiation": flagged_differentiation,
        },
    }

    save_report(summary, RESULTS_DIR, f"evaluation_summary_{TIMESTAMP}")

    # Log summary to Langfuse
    langfuse.score(
        trace_id=trace.id,
        name="evaluation_summary",
        value=1.0,
        comment=str(combined_quality)
    )

    all_flagged = set(flagged_confidence + flagged_consistency)
    if all_flagged or flagged_differentiation:
        print(f"\n  ⚠ Flagged for human review:")
        if flagged_confidence:
            print(f"    Low confidence:      {flagged_confidence}")
        if flagged_consistency:
            print(f"    Low consistency:     {flagged_consistency}")
        if flagged_differentiation:
            print(f"    Low differentiation: overall score below threshold")
    else:
        print(f"\n  ✓ No personas flagged for review")

    print(f"\n  Combined quality scores:")
    for persona, score in combined_quality.items():
        print(f"    {persona}: {score}")

    print(f"\n  ✓ All reports saved → evaluation/results/")

    # ── Flush Langfuse ────────────────────────────────────────────────────────
    langfuse.flush()
    print("✓ Langfuse trace completed — check your dashboard at cloud.langfuse.com")


if __name__ == "__main__":
    main()