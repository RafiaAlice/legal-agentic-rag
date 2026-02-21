
# Multi-Agent Legal RAG with Jurisdiction-Aware Retrieval

A modular Retrieval-Augmented Generation (RAG) system for U.S. federal legal question answering.

## Architecture
- Router Agent (domain + jurisdiction prediction)
- Retrieval Agent (ChromaDB + MiniLM embeddings)
- Citation Verification Agent
- Synthesis Agent (Gemini 2.0 Flash)

## Features
- Sliding-window chunking (800 + 200 overlap)
- Jurisdiction-aware filtering
- Citation verification
- Web-based fallback retrieval
