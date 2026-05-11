#!/bin/bash

# ── Multi-Agent Synthetic Researcher — Step-by-Step Pipeline
# PROJECT_WATSONX — Group 29 HSLU
#
# Usage:
#   bash run.sh
#   bash run.sh --topic "Visa card pricing sensitivity in Switzerland"
#   bash run.sh --personas student family
#   bash run.sh --topic "Mobile payment adoption" --personas student professional

set -e

TOPIC="Visa card adoption and payment behavior in Switzerland"
PERSONAS="student professional mittelstand family retired"

while [[ $# -gt 0 ]]; do
  case $1 in
    --topic)
      TOPIC="$2"
      shift 2
      ;;
    --personas)
      shift
      PERSONAS=""
      while [[ $# -gt 0 && ! "$1" == --* ]]; do
        PERSONAS="$PERSONAS $1"
        shift
      done
      ;;
    *)
      shift
      ;;
  esac
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Multi-Agent Synthetic Researcher — Group 29"
echo "  Topic    : $TOPIC"
echo "  Personas : $PERSONAS"
echo "════════════════════════════════════════════════════════════"

# ── Step 0: Generate survey questions ─────────────────────────
echo ""
echo "  STEP 0 — Generating survey questions..."
python3 orchestrator/question_generator.py \
  --topic "$TOPIC" \
  --provider watsonx

# ── Step 1: Run persona survey agents ─────────────────────────
echo ""
echo "  STEP 1 — Running persona survey agents..."
for PERSONA in $PERSONAS; do
  echo "  → $PERSONA"
  python3 scripts/run_survey.py --persona "$PERSONA" --provider watsonx
done

# ── Step 2: Run evaluation suite ──────────────────────────────
echo ""
echo "  STEP 2 — Running evaluation suite..."
python3 evaluation/scripts/run_evaluation.py

# ── Step 3a: Run analyst agent ─────────────────────────────────
echo ""
echo "  STEP 3a — Running analyst agent..."
python3 analyst/analyst_agent.py --all --provider watsonx

# ── Step 3b: Generate insight report ──────────────────────────
echo ""
echo "  STEP 3b — Generating insight report..."
python3 analyst/insights_report.py --provider watsonx

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✓ Pipeline complete"
echo "════════════════════════════════════════════════════════════"
echo ""
