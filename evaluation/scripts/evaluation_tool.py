"""
Evaluation Tool for IBM watsonx Orchestrate.
Runs consistency, confidence and differentiation evaluation on persona responses
and logs results to Langfuse.
"""

import os
import sys
import json
from pathlib import Path
from ibm_watsonx_orchestrate.agent_builder.tools import tool
from pathlib import Path

def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv("/Users/sedadulger/Visual studio codes/final-branch/.env")
    except Exception:
        pass
_load_env()        

BASE_DIR = Path(__file__).parent.parent
RESPONSES_DIR = BASE_DIR / "responses"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PERSONAS = ["student", "professional", "family", "retired", "mittelstand"]
CONSISTENCY_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 0.6
DIFFERENTIATION_THRESHOLD = 0.6


def _get_latest_response(persona: str):
    files = sorted(RESPONSES_DIR.glob(f"{persona}_*.json"), reverse=True)
    return files[0] if files else None


@tool
def run_evaluation() -> str:
    """
    Runs the full evaluation pipeline on the latest persona responses.
    Scores consistency, confidence and differentiation for all 5 Swiss consumer personas.
    Logs all scores to Langfuse dashboard and returns a summary report.

    Returns:
        A JSON string with consistency scores, confidence scores, differentiation score,
        combined quality scores, and flagged personas for human review.
    """
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))

    try:
        from consistency_score import score_consistency
        from confidence_score import score_confidence
        from differentiation_score import score_differentiation
    except ImportError as e:
        return f"Error importing evaluation scripts: {str(e)}"

    # ── Load env for Langfuse ─────────────────────────────────────────────────
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    # ── Init Langfuse ─────────────────────────────────────────────────────────
    try:
        from langfuse import Langfuse
        langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
        trace = langfuse.trace(name="evaluation_pipeline_orchestrate")
        langfuse_enabled = True
    except Exception:
        langfuse_enabled = False
        langfuse = None
        trace = None

    consistency_report = {}
    confidence_report = {}
    response_files = []

    # ── Consistency & Confidence ──────────────────────────────────────────────
    for persona in PERSONAS:
        file = _get_latest_response(persona)
        if not file:
            continue

        response_files.append(file)

        try:
            c_result = score_consistency(file)
            consistency_report[persona] = c_result
            if langfuse_enabled:
                langfuse.score(trace_id=trace.id, name=f"consistency_{persona}",
                               value=c_result["consistency_score"])
        except Exception as e:
            consistency_report[persona] = {"consistency_score": 0.0, "error": str(e)}

        try:
            conf_result = score_confidence(file)
            confidence_report[persona] = conf_result
            if langfuse_enabled:
                langfuse.score(trace_id=trace.id, name=f"confidence_{persona}",
                               value=conf_result["confidence_score"])
        except Exception as e:
            confidence_report[persona] = {"confidence_score": 0.0, "error": str(e)}

    # ── Differentiation ───────────────────────────────────────────────────────
    try:
        diff_result = score_differentiation(response_files) if len(response_files) >= 2 else {}
        diff_score = diff_result.get("differentiation_score", 0.0) if diff_result else 0.0
        if langfuse_enabled and diff_result:
            langfuse.score(trace_id=trace.id, name="differentiation", value=diff_score)
    except Exception as e:
        diff_result = {"differentiation_score": 0.0, "error": str(e)}
        diff_score = 0.0

    # ── Combined quality ──────────────────────────────────────────────────────
    combined_quality = {}
    for persona in consistency_report:
        c = consistency_report[persona].get("consistency_score", 0.0)
        conf = confidence_report.get(persona, {}).get("confidence_score", 0.0)
        combined_quality[persona] = round((c + conf + diff_score) / 3, 3)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "consistency":  {p: r.get("consistency_score", 0.0) for p, r in consistency_report.items()},
        "confidence":   {p: r.get("confidence_score", 0.0)  for p, r in confidence_report.items()},
        "differentiation": diff_score,
        "combined_quality_score": combined_quality,
        "flagged_for_review": {
            "low_confidence":  [p for p, r in confidence_report.items()  if r.get("confidence_score", 1.0)  < CONFIDENCE_THRESHOLD],
            "low_consistency": [p for p, r in consistency_report.items() if r.get("consistency_score", 1.0) < CONSISTENCY_THRESHOLD],
            "low_differentiation": [] if diff_score >= DIFFERENTIATION_THRESHOLD else ["all"],
        },
        "langfuse_logged": langfuse_enabled,
    }

    if langfuse_enabled:
        langfuse.flush()

    return json.dumps(summary, indent=2, ensure_ascii=False)
