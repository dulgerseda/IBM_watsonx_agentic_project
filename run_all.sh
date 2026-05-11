#!/bin/bash

# PROJECT_WATSONX — Group 29 HSLU
# ────────────────────────────────────────────────────────────
# OPTION 1 — Run via Orchestrator (recommended)
# ────────────────────────────────────────────────────────────
python3 orchestrator/orchestrator.py --provider watsonx

# ────────────────────────────────────────────────────────────
# OPTION 2 — Run step by step
# ────────────────────────────────────────────────────────────
# python3 orchestrator/question_generator.py --topic "Visa card adoption and payment behavior in Switzerland" --provider watsonx
# python3 scripts/run_survey.py --all --provider watsonx
# python3 evaluation/scripts/run_evaluation.py
# python3 analyst/analyst_agent.py --all --provider watsonx
# python3 analyst/insights_report.py --provider watsonx