from __future__ import annotations

import argparse
import sys

from .agents import LegalAgentSystem
from .config import Config, ensure_dirs
from .ingest import LegalDataIngester
from .vectordb import LegalVectorDB

def build_db(cfg: Config) -> None:
    ensure_dirs(cfg)
    ingester = LegalDataIngester(cfg)
    db = LegalVectorDB(cfg)

    if cfg.REBUILD_DB or db.count() == 0:
        docs = []
        docs.extend(ingester.fetch_us_code_sample())
        docs.extend(ingester.fetch_federal_register_updates())
        docs.extend(ingester.fetch_supreme_court_opinions())
        chunked = ingester.chunk_documents(docs)
        db.add_documents(chunked)
        print(f"✓ Built DB with {db.count()} chunks")
    else:
        print(f"✓ Using existing DB with {db.count()} chunks")

def interactive_qa(cfg: Config) -> None:
    ensure_dirs(cfg)
    db = LegalVectorDB(cfg)
    if db.count() == 0:
        print("Vector DB is empty. Run: python -m legal_rag.cli build-db --rebuild")
        return

    system = LegalAgentSystem(cfg, db)

    print("\nLegal Agentic RAG (type 'exit' to quit)\n")
    while True:
        q = input("Q> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        out = system.query(q)
        print("\nA:")
        print(out["answer"])
        if out["citations"]:
            print("\nCitations:")
            for i, c in enumerate(out["citations"], 1):
                print(f"  {i}. {c}")
        print()

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent Legal RAG (Jurisdiction-Aware)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-db", help="Fetch sources, chunk, embed, and persist ChromaDB")
    p_build.add_argument("--rebuild", action="store_true", help="Force rebuild (ignores existing DB)")

    sub.add_parser("chat", help="Interactive Q&A loop using existing DB")

    args = parser.parse_args(argv)

    cfg = Config()
    if args.cmd == "build-db":
        if args.rebuild:
            # override env-driven value
            object.__setattr__(cfg, "REBUILD_DB", True)  # type: ignore
        build_db(cfg)
    elif args.cmd == "chat":
        interactive_qa(cfg)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
