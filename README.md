# Multi-Agent Synthetic Researcher
**PROJECT WATSONX — Group 29 HSLU**

An AI-powered multi-agent system that simulates Swiss consumer feedback for Visa card product research. Five synthetic personas grounded in real Swiss demographic data answer survey questions, producing structured insights for the Visa product team.

---

## System Pipeline

```
Orchestrator
      ↓
Question Generator → generates questions from a topic
      ↓
Persona Agents (5) → same questions, all personas, RAG-grounded
      ↓
Evaluation Suite → consistency, confidence, differentiation
      ↓
Analyst Agent → reads all persona responses per question → structured insight JSON with Visa recommendations
      ↓
Insight Report → category-level enriched insights per persona (price sensitivity, feature preference, switching behavior, trade-offs, willingness to pay)
```

---

## Personas

| Persona | Name | Age | Net Income | Fee Ceiling |
|---------|------|-----|------------|-------------|
| Student | Campus Felix | 22 | CHF 800–2,000/mo | CHF 5/mo |
| Professional | Mobile-First Lukas | 32 | CHF 8,500–11,200/mo | CHF 25–30/mo |
| Mittelstand | Average-Earner Anna | 36 | CHF 5,600/mo | CHF 12/mo |
| Family | Value-Smart Sandra | 44 | CHF 7,500–9,800/mo (household) | CHF 10/mo |
| Retired | Trust-First Peter | 65 | CHF 1,682/mo | CHF 5/mo |

---

## Quick Start

**1. Install dependencies:**
```bash
pip install requests chromadb
```

**2. Create `.env` in project root:**
```
# Calling LLM models 

WATSONX_URL=https://eu-de.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id
WATSONX_API_KEY=your_api_key
WATSONX_MODEL_ID=meta-llama/llama-3-3-70b-instruct

# watsonX Orchestrate (ADK)

ORCHESTRATE_URL=https://api-eu-de.watson-orchestrate.cloud.ibm.com/instances/ba0aa69
ORCHESTRATE_API_KEY=YzzXPUGMbzIzmhKW9mE-92LpzQSma9Rd7JVfFXn0WbHB

# Connect to Langfuse

LANGFUSE_SECRET_KEY=sk-lf-199f881f-8b8d-44f3-b923-7931956f1d1a
LANGFUSE_PUBLIC_KEY=pk-lf-af60f384-33c7-45d7-94ca-b06391557ca0
LANGFUSE_HOST=https://cloud.langfuse.com
```

**3. Run the full pipeline:**
```bash
python3 orchestrator/orchestrator.py --provider watsonx --topic "Visa card pricing in Switzerland"
```

**4. Test without API calls:**
```bash
python3 orchestrator/orchestrator.py --provider mock --topic "Mobile payment adoption"
```

**5. Run with specific personas only (faster for testing):**
```bash
python3 orchestrator/orchestrator.py --provider watsonx --topic "Visa card pricing in Switzerland" --personas family mittelstand
```

**6. Run survey or evaluation separately:**
```bash
python3 scripts/run_survey.py --all --provider watsonx
python3 evaluation/scripts/run_evaluation.py
```

**Expected runtime:** ~5-8 minutes for 2 personas, ~15-25 minutes for all 5 personas. WatsonX API response time is the bottleneck, not local hardware.

---

## Configuration

Edit `config.json` to change number of questions, personas, or model settings without touching any code.

---

## Data Sources

The RAG knowledge base uses 57 chunks from publicly available Swiss data. No sensitive or Visa-internal data is used.

| Source | Publisher | Year |
|--------|-----------|------|
| Payment Methods Survey | Swiss National Bank (SNB) | 2024 |
| Swiss Payment Monitor | ZHAW / University of St.Gallen | 2025 |
| Population by Age | BFS | 2024 |
| Household Budget Survey | BFS | 2022–2023 |

---

## Project Structure

```
project-watsonx-group29/
├── config.json
├── .env                      ← credentials, gitignored
├── orchestrator/             ← agents that coordinate
│   ├── orchestrator.py
│   └── question_generator.py
├── rag/                      ← knowledge layer, self-contained
│   ├── chunks.py
│   ├── retriever.py
│   ├── update_chunks_docling.py    ← gets the key points from the sources automatically
│   └── chunk_suggestions.md        ← summary of generated info from the sources for chunks
├── scripts/
│   └── run_survey.py
├── agents/                   ← keeps generated agents for watsonx orchestrate
│   └── generate_persona_yamls.py 
├── evaluation/
│   ├── results/              ← evaluation outputs
│   └── scripts/
│       ├── run_evaluation.py
│       ├── consistency_score.py
│       ├── confidence_score.py
│       ├── differentiation_score.py
│       └── evaluation_utils.py
├── analyst/
│   ├── analyst_agent.py      ← generates insight JSONs from persona responses
│   └── insights_report.py    ← category-level enriched report per persona
├── personas/                 ← persona specifications
├── prompts/                  ← LLM prompts
├── survey/                   ← keeps the generated survey questions
├── responses/                ← raw persona outputs
├── sources/                  ← source pdfs for demographic data 
└── insights/                 ← analyst outputs
    ├── insights_visa_survey_{timestamp}.json   ← analyst agent output
    └── insight_report_{timestamp}.json         ← insight report output
```

---

*Group 29 — HSLU MSc Agentic AI Course*
