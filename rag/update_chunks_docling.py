"""
RAG Knowledge Base Updater — Docling
Yearly update script: extracts text from source PDFs and Excel files.
Usage:
    python3 rag/update_chunks_docling.py
Output:
    Saves cleaned extracted content to rag/chunk_suggestions.md for manual review.
    After review, manually add relevant facts to rag/chunks.py
"""

from pathlib import Path
from docling.document_converter import DocumentConverter
from datetime import datetime

SOURCES_DIR = Path(__file__).parent.parent / "sources"
OUTPUT_FILE = Path(__file__).parent / "chunk_suggestions.md"

SOURCES = [
    "paytrans_survey_report_2024.en.pdf",
    "Report_Swiss_Payment_Monitor_2025-1_final_en.pdf",
    "household_budget_data.xlsx",
    "population_by_age_region.xlsx",
]

NOISE_PATTERNS = [
    "<!-- image -->",
    "List of figures",
    "List of tables",
    "Table of contents",
    "Figure ",
    "Appendix",
    "-----",
    "____",
]

def is_useful_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if len(line) < 30:
        return False
    if any(pattern in line for pattern in NOISE_PATTERNS):
        return False
    if line.startswith("|"):
        return False
    if "�" in line:
        return False
    return True

def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = [line.strip() for line in lines if is_useful_line(line)]
    return "\n\n".join(cleaned)

def extract_source(filename: str) -> str:
    path = SOURCES_DIR / filename
    if not path.exists():
        print(f"⚠ File not found: {filename}")
        return ""
    print(f"Extracting: {filename}...")
    converter = DocumentConverter()
    result    = converter.convert(str(path))
    raw       = result.document.export_to_markdown()
    return clean_text(raw)

def main():
    lines = []
    lines.append(f"# Chunk Suggestions — Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("Review this file and add relevant facts to `rag/chunks.py`.\n")
    lines.append("---\n")

    for source in SOURCES:
        text = extract_source(source)
        if text:
            lines.append(f"\n## {source}\n")
            lines.append(text)
            lines.append("\n---\n")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Saved → rag/chunk_suggestions.md")
    print("Review the file and add relevant facts to rag/chunks.py")

if __name__ == "__main__":
    main()