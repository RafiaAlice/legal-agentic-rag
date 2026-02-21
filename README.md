# Multi-Agent Legal RAG (Jurisdiction-Aware)

Agentic Retrieval-Augmented Generation system for **U.S. legal Q&A** with:
- **Router agent** (predicts domain + jurisdiction)
- **Retriever agent** (ChromaDB dense retrieval with metadata filters)
- **Citation agent** (keeps only chunks with citations / case identifiers)
- **Synthesis agent** (answers with citations)
- Optional **web fallback** when the local DB has no relevant sources

> ⚠️ **Disclaimer:** This project is for educational/research use and does not constitute legal advice.

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Set your Gemini key
Create a `.env` file (or export an env var):
```bash
export GEMINI_API_KEY="YOUR_KEY"
```

### 3) Build the vector DB
```bash
python -m legal_rag.cli build-db --rebuild
```

### 4) Chat
```bash
python -m legal_rag.cli chat
```

Example questions:
- What are the penalties for making false statements to federal agents?
- What is the Sherman Antitrust Act about?
- What are my rights under 42 U.S.C. § 1983?

## What it fetches (demo-scale)
- A few U.S. Code sections via Cornell LII (HTML)
- Recent Federal Register entries via official API
- Recent SCOTUS opinions via CourtListener API

## Repo structure
- `legal_rag/` — core package (config, ingestion, vector DB, agents, CLI)
- `requirements.txt` — Python dependencies

## Notes
- Chunking defaults to **800 tokens** with **200 overlap** to preserve legal context across chunk boundaries.
- This repo uses **Gemini** (not OpenAI GPT) via `google-generativeai`.
