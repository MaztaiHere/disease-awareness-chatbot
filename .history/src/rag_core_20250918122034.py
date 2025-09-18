# src/rag_core.py
import os
import re
import json
import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Directories
PROCESSED_DATA_DIR = Path("data/processed")
VECTOR_DB_DIR = Path("vector_db")

logging.basicConfig(level=logging.INFO)


class MedicalRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store_client = None
        self.chains = {}

        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Build chains for each domain
        for domain in ["outbreak", "symptom", "misinformation"]:
            self.build_vector_store(domain)

    # -------------------------------------------------------------------------
    # Utility: load processed chunks
    def _load_chunks(self, domain):
        file_path = PROCESSED_DATA_DIR / f"{domain}_chunks.json"
        if not file_path.exists():
            logging.error("Processed chunks file not found: %s", file_path)
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------------------------------------------------
    # Build vector store and chain
    def build_vector_store(self, domain):
        logging.info("Building vector store for %s...", domain)
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

        chunks = self._load_chunks(domain)
        if not chunks:
            logging.warning("No chunks found for domain: %s", domain)
            return

        # Create Chroma DB
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(VECTOR_DB_DIR / domain),
            collection_name=f"medical_rag_{domain}",
        )
        vector_store.persist()

        # Keep a client ref for debug retrieval
        self.vector_store_client = vector_store

        # Retriever + chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 8})
        self.chains[domain] = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
        )
        logging.info("Built vector store and chain for %s.", domain)

    # -------------------------------------------------------------------------
    # Debug retrieval helpers
    def _get_collection(self, domain):
        name = f"medical_rag_{domain}"
        try:
            return Chroma(
                persist_directory=str(VECTOR_DB_DIR / domain),
                collection_name=name,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            logging.warning("Could not get collection '%s': %s", name, e)
            return None

    def debug_retrieve(self, domain, query_text, n_results=5, where: dict = None):
        """
        Query chroma collection directly and return the top results for debugging.
        where: optional dict for metadata filtering (e.g. {"disease": "malaria", "year": 2024})
        """
        collection = self._get_collection(domain)
        if not collection:
            logging.error("Collection for domain '%s' not found.", domain)
            return []

        try:
            if where:
                resp = collection._collection.query(
                    query_texts=[query_text], n_results=n_results, where=where
                )
            else:
                resp = collection._collection.query(
                    query_texts=[query_text], n_results=n_results
                )
        except Exception as e:
            logging.error("Chroma query failed: %s", e)
            return []

        docs = []
        docs_list = resp.get("documents", [[]])[0]
        metas_list = resp.get("metadatas", [[]])[0]

        for d, m in zip(docs_list, metas_list):
            docs.append({"page_content": d, "metadata": m})
        return docs

    # -------------------------------------------------------------------------
    # Extraction of diseases and years
    def _extract_diseases_and_years(self, text):
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
        years = [int(y) for y in years]
        disease_keywords = [
            "malaria",
            "dengue",
            "rabies",
            "cholera",
            "diarrheal",
            "typhoid",
        ]
        diseases_found = []
        lower = text.lower()
        for kw in disease_keywords:
            if kw in lower:
                diseases_found.append(kw)
        return list(set(diseases_found)), list(set(years))

    # -------------------------------------------------------------------------
    # Main query function
    def query(self, domain, english_query, user_language="en"):
        trace = {"domain": domain, "query": english_query, "language": user_language}

        if domain not in self.chains:
            return "Domain not supported.", trace

        # Debug: show base retrieval
        try:
            debug_docs = self.debug_retrieve(domain, english_query, n_results=8)
            trace["debug_retrieved"] = [
                {
                    "metadata": d.get("metadata", {}),
                    "snippet": d.get("page_content", "")[:200],
                }
                for d in debug_docs
            ]
        except Exception as e:
            trace["debug_retrieved_error"] = str(e)

        # Expand multi-disease/year queries
        diseases, years = self._extract_diseases_and_years(english_query)
        aggregated_docs = []
        if diseases or years:
            combos = []
            if diseases and years:
                combos = [f"{d} {y}" for d in diseases for y in years]
            elif diseases:
                combos = diseases
            elif years:
                combos = [str(y) for y in years]

            for subq in combos:
                docs = self.debug_retrieve(domain, subq, n_results=5)
                aggregated_docs.extend(docs)

            seen = set()
            unique_docs = []
            for d in aggregated_docs:
                key = (
                    d.get("metadata", {}).get("source"),
                    d.get("page_content", "")[:200],
                )
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(d)

            trace["expanded_retrieval_count"] = len(unique_docs)
            trace["expanded_retrieval_examples"] = [
                {
                    "meta": d.get("metadata", {}),
                    "snippet": d.get("page_content", "")[:200],
                }
                for d in unique_docs[:8]
            ]

        # Finally, run the RetrievalQA chain
        response = self.chains[domain].invoke({"query": english_query})

        return response["result"], trace
