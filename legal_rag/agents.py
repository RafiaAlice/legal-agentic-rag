from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from langgraph.graph import END, StateGraph

from .config import Config
from .vectordb import LegalVectorDB

class AgentState(TypedDict):
    query: str
    legal_domain: str
    jurisdiction: str
    retrieved_docs: List[Dict[str, Any]]
    verified_docs: List[Dict[str, Any]]
    final_answer: str
    citations: List[str]
    suggestions: List[str]

@dataclass
class LegalAgentSystem:
    """Agentic RAG system (router → retriever → verifier → synthesizer)."""

    cfg: Config
    vector_db: LegalVectorDB

    def __post_init__(self) -> None:
        if not self.cfg.GEMINI_API_KEY:
            raise RuntimeError("Missing GEMINI_API_KEY / GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=self.cfg.GEMINI_API_KEY)

        # Initialize Gemini model
        self.llm = genai.GenerativeModel(self.cfg.LLM_MODEL)

        # Build workflow
        self.workflow = self._build_workflow()

    # ---------------- Web fallback helpers ----------------

    def _simple_web_search(self, question: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Minimal DuckDuckGo HTML scrape (best-effort). Returns {title,url}."""
        q = quote_plus(question + " law site:.gov OR site:.edu OR site:.org")
        url = f"https://duckduckgo.com/html/?q={q}"
        headers = {"User-Agent": "Legal-Agentic-RAG/0.1 (+github)"}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code >= 400:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.select("a.result__a") or soup.select("a[href]")
            results: List[Dict[str, str]] = []
            for a in links:
                href = a.get("href")
                title = a.get_text(" ", strip=True)
                if not href or not title or href.startswith("#"):
                    continue
                results.append({"title": title, "url": href})
                if len(results) >= max_results:
                    break
            return results
        except Exception:
            return []

    def _normalize_url(self, raw_url: str) -> str:
        url = raw_url
        if url.startswith("//"):
            url = "https:" + url
        if url.startswith("/"):
            url = "https://duckduckgo.com" + url
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
        except Exception:
            pass
        return url

    def _fetch_web_pages(self, entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
        pages: List[Dict[str, str]] = []
        headers = {"User-Agent": "Legal-Agentic-RAG/0.1 (+github)"}
        for e in entries:
            url = self._normalize_url(e["url"])
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(" ", strip=True)
                if not text:
                    continue
                pages.append({"title": e["title"], "url": url, "content": text[:8000]})
            except Exception:
                continue
        return pages

    def _answer_via_web(self, question: str) -> tuple[str, List[str]]:
        results = self._simple_web_search(question, max_results=3)
        if not results:
            return (
                "No relevant sources were found in the local knowledge base, and web fallback returned no usable results. "
                "For legal matters, consult an up-to-date legal database or a licensed attorney.",
                [],
            )

        pages = self._fetch_web_pages(results)
        if not pages:
            return (
                "Web fallback could not extract enough information to answer reliably. "
                "Please consult an up-to-date legal database or a licensed attorney.",
                [],
            )

        context = "\n\n---\n\n".join(
            [f"Title: {p['title']}\nURL: {p['url']}\nContent:\n{p['content']}" for p in pages]
        )

        prompt = f"""You are a legal research assistant.

Question: {question}

Web Sources (snippets):
{context}

Rules:
- Base your answer ONLY on the snippets above.
- Cite statutes/cases if present in snippets.
- Be explicit about uncertainty and limitations.

Return a structured answer with short sections."""

        try:
            resp = self.llm.generate_content(prompt)
            answer = getattr(resp, "text", str(resp))

            pattern = (
                r"\b\d+\s+U\.S\.C\.\s+§\s*\d+[A-Za-z0-9]*"
                r"|\bK\.S\.A\.\s*\d+-\d+[A-Za-z0-9]*"
                r"|\b[A-Z][A-Za-z]+ v\. [A-Z][A-Za-z]+"
            )
            citations = list(dict.fromkeys(re.findall(pattern, answer)))
            return answer, citations
        except Exception as e:
            return f"Error generating web-based answer: {e}", []

    # ---------------- Workflow ----------------

    def _build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self.router_agent)
        workflow.add_node("retriever", self.retrieval_agent)
        workflow.add_node("verifier", self.citation_agent)
        workflow.add_node("synthesizer", self.synthesis_agent)

        workflow.set_entry_point("router")
        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "verifier")
        workflow.add_edge("verifier", "synthesizer")
        workflow.add_edge("synthesizer", END)
        return workflow.compile()

    def router_agent(self, state: AgentState) -> AgentState:
        query = state["query"]
        prompt = f"""Analyze the legal query and determine:
1) legal domain (criminal, civil rights, immigration, administrative, constitutional, etc.)
2) jurisdiction (federal, state, or both)

Query: {query}

Respond ONLY in JSON:
{{"legal_domain": "...", "jurisdiction": "..."}}"""
        try:
            resp = self.llm.generate_content(prompt)
            text = (resp.text or "").strip().replace("```json", "").replace("```", "")
            result = json.loads(text)
            state["legal_domain"] = result.get("legal_domain", "general")
            state["jurisdiction"] = result.get("jurisdiction", "federal")
        except Exception:
            state["legal_domain"] = "general"
            state["jurisdiction"] = "federal"
        return state

    def retrieval_agent(self, state: AgentState) -> AgentState:
        query = state["query"]
        jurisdiction = state["jurisdiction"]

        filters: Optional[Dict[str, Any]] = None
        if jurisdiction.lower() != "both":
            filters = {"jurisdiction": {"$eq": jurisdiction.capitalize()}}

        state["retrieved_docs"] = self.vector_db.search(
            query=query,
            n_results=self.cfg.TOP_K,
            filters=filters,
        )
        return state

    def citation_agent(self, state: AgentState) -> AgentState:
        verified: List[Dict[str, Any]] = []
        for doc in state["retrieved_docs"]:
            meta = doc.get("metadata", {})
            if meta.get("citation") or meta.get("document_number") or meta.get("case_name"):
                verified.append(doc)
        state["verified_docs"] = verified
        return state

    def synthesis_agent(self, state: AgentState) -> AgentState:
        query = state["query"]
        docs = state["verified_docs"]

        if not docs:
            answer, citations = self._answer_via_web(query)
            state["final_answer"] = answer
            state["citations"] = citations
            return state

        context = "\n\n---\n\n".join(
            [
                (
                    f"Source: {d['metadata'].get('source','Unknown')}\n"
                    f"Citation/Title: {d['metadata'].get('citation', d['metadata'].get('title','N/A'))}\n"
                    f"Content: {d['content'][:700]}..."
                )
                for d in docs[:3]
            ]
        )

        prompt = f"""You are an expert legal assistant. Answer ONLY using the sources below.

Question: {query}

Sources:
{context}

Provide:
- A clear answer in plain language
- Citations (as provided)
- Practical next steps
- Limitations/disclaimer

Return a structured response."""

        try:
            resp = self.llm.generate_content(prompt)
            state["final_answer"] = resp.text
            state["citations"] = [
                d["metadata"].get("citation", d["metadata"].get("title", "Unknown"))
                for d in docs
            ]
        except Exception as e:
            state["final_answer"] = f"Error generating response: {e}"
            state["citations"] = []

        return state

    def query(self, question: str) -> Dict[str, Any]:
        initial_state: AgentState = {
            "query": question,
            "legal_domain": "",
            "jurisdiction": "",
            "retrieved_docs": [],
            "verified_docs": [],
            "final_answer": "",
            "citations": [],
            "suggestions": [],
        }
        final_state = self.workflow.invoke(initial_state)
        return {
            "question": question,
            "answer": final_state["final_answer"],
            "citations": final_state["citations"],
            "domain": final_state["legal_domain"],
            "jurisdiction": final_state["jurisdiction"],
        }
