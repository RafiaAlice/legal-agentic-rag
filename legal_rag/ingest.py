from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import Config

@dataclass
class LegalDataIngester:
    """Fetches and chunks a small set of U.S. legal sources.

    Notes:
      - This is a *demo-scale* ingester (a few statutes + a few recent docs).
      - For production, you’d expand coverage and use official bulk data sources.
    """

    cfg: Config

    def __post_init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.CHUNK_SIZE,
            chunk_overlap=self.cfg.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def fetch_us_code_sample(self) -> List[Dict[str, Any]]:
        """Fetch a few U.S. Code sections from Cornell LII (HTML)."""
        sample_urls = [
            "https://www.law.cornell.edu/uscode/text/18/1001",  # False statements
            "https://www.law.cornell.edu/uscode/text/15/1",     # Sherman Act
            "https://www.law.cornell.edu/uscode/text/42/1983",  # Civil Rights
        ]

        docs: List[Dict[str, Any]] = []
        headers = {"User-Agent": "Legal-Agentic-RAG/0.1 (+github)"}

        for url in sample_urls:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.content, "html.parser")
                title_elem = soup.find("h1")
                title = title_elem.get_text(strip=True) if title_elem else "Unknown"

                content = ""
                candidates = [
                    soup.find("div", {"id": "tab-source-content"}),
                    soup.find("div", {"id": "content"}),
                    soup.find("main"),
                    soup.find("article"),
                ]
                for cand in candidates:
                    if cand:
                        content = cand.get_text(separator="\n", strip=True)
                        if content:
                            break
                if not content:
                    content = soup.get_text(separator="\n", strip=True)

                parts = url.rstrip("/").split("/")
                citation = url
                if len(parts) >= 2:
                    title_num, section_num = parts[-2], parts[-1]
                    citation = f"{title_num} U.S.C. § {section_num}"

                docs.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": "Cornell LII (US Code)",
                            "citation": citation,
                            "title": title,
                            "url": url,
                            "jurisdiction": "Federal",
                            "type": "Statute",
                            "date_accessed": datetime.utcnow().isoformat(),
                        },
                    }
                )
            except Exception:
                continue

        return docs

    def fetch_federal_register_updates(self, per_page: int = 20) -> List[Dict[str, Any]]:
        """Fetch recent Federal Register entries via the official API."""
        docs: List[Dict[str, Any]] = []
        url = f"https://www.federalregister.gov/api/v1/documents.json?per_page={per_page}&order=newest"
        headers = {"User-Agent": "Legal-Agentic-RAG/0.1 (+github)"}

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return docs

            data = resp.json()
            for doc in data.get("results", [])[:10]:
                content = f"{doc.get('title','')}\n\n{doc.get('abstract','')}"
                docs.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": "Federal Register",
                            "title": doc.get("title", ""),
                            "document_number": doc.get("document_number", ""),
                            "publication_date": doc.get("publication_date", ""),
                            "url": doc.get("html_url", ""),
                            "jurisdiction": "Federal",
                            "type": "Regulation Update",
                        },
                    }
                )
        except Exception:
            return docs

        return docs

    def fetch_supreme_court_opinions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent SCOTUS opinions via CourtListener (public API)."""
        docs: List[Dict[str, Any]] = []
        url = "https://www.courtlistener.com/api/rest/v3/opinions/?court=scotus&order_by=-date_filed"
        headers = {"User-Agent": "Legal-Agentic-RAG/0.1 (+github)"}

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return docs

            data = resp.json()
            for opinion in data.get("results", [])[:n]:
                resource_uri = opinion.get("resource_uri")
                if not resource_uri:
                    continue

                full_url = f"https://www.courtlistener.com{resource_uri}"
                op_resp = requests.get(full_url, headers=headers, timeout=15)
                if op_resp.status_code != 200:
                    continue

                op_data = op_resp.json()
                text = op_data.get("plain_text") or op_data.get("html", "") or ""
                docs.append(
                    {
                        "content": text[:8000],
                        "metadata": {
                            "source": "CourtListener (SCOTUS)",
                            "case_name": opinion.get("case_name", ""),
                            "citation": opinion.get("case_name", ""),
                            "date_filed": opinion.get("date_filed", ""),
                            "url": f"https://www.courtlistener.com{opinion.get('absolute_url','')}",
                            "jurisdiction": "Federal",
                            "court": "SCOTUS",
                            "type": "Case",
                        },
                    }
                )
        except Exception:
            return docs

        return docs

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split full documents into overlapping chunks and propagate metadata."""
        chunked: List[Dict[str, Any]] = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                meta = dict(doc["metadata"])
                meta.update({"chunk_id": i, "total_chunks": len(chunks)})
                chunked.append({"content": chunk, "metadata": meta})
        return chunked
