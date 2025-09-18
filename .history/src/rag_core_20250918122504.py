import torch
torch.set_num_threads(1)  # Prevents segfaults on macOS with Streamlit + transformers

import os
import json
import logging
import requests
import chromadb
from pathlib import Path
from functools import lru_cache
from typing import Optional

# LangChain / Vector / LLM Imports
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

# Hugging Face translation pipeline
from langdetect import detect
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = str(BASE_DIR / "data" / "processed")
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
PERSIST_DIRECTORY = str(BASE_DIR / "vector_db")
LLM_MODEL_ID = "mistral-7b-instruct"  # logical name only

# Local LLM config (GGUF download)
MODEL_DIR = str(BASE_DIR / "models")
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_URL = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# mBART Supported Language Codes
MBART_LANG_CODES = {
    "en": "en_XX", "hi": "hi_IN", "ta": "ta_IN", "te": "te_IN", "ml": "ml_IN",
    "bn": "bn_IN", "gu": "gu_IN", "mr": "mr_IN", "kn": "kn_IN",
    # Add other languages mBART supports here if needed
    "es": "es_XX", "fr": "fr_XX", "de": "de_DE", "it": "it_IT", "ja": "ja_XX"
}

# Cached mBART pipeline (lazy load once)
@lru_cache(maxsize=1)
def get_mbart_pipeline():
    return pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")

class MedicalRAG:
    """
    Retrieval-Augmented Generation system for medical/public health.
    - Local LLM (via llama-cpp GGUF)
    - Chroma vector DB with HuggingFace embeddings
    - Translation handled exclusively with mBART
    """

    def __init__(self):
        logging.info("Initializing MedicalRAG system...")
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        self._ensure_model_downloaded()

        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.llm = self._initialize_llm()
        self.mbart_pipeline: Optional[pipeline] = None
        self.chains = self._initialize_chains()
        logging.info("MedicalRAG system initialized.")

    def _ensure_model_downloaded(self):
        if os.path.exists(MODEL_PATH):
            logging.info("Local GGUF model already present; skipping download.")
            return

        logging.info("Downloading GGUF model...")
        try:
            with requests.get(MODEL_URL, stream=True, timeout=600) as r: # Increased timeout
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded / total * 100
                                if downloaded % (1024 * 1024 * 50) < 8192: # Log every 50MB
                                    logging.info(f"Downloaded {pct:.1f}%")
            logging.info("Model downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            logging.error("Please download manually: %s", MODEL_PATH)

    def _initialize_llm(self):
        try:
            if not os.path.exists(MODEL_PATH):
                logging.warning("GGUF model not found at %s; skipping LLM init.", MODEL_PATH)
                return None

            llm = LlamaCpp(
                model_path=MODEL_PATH, n_ctx=4096, n_threads=8, n_batch=512,
                temperature=0.1, top_p=0.9, repeat_penalty=1.15, max_tokens=512, verbose=False,
            )
            logging.info("Local LLM initialized via LlamaCpp.")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize local LLM: {e}")
            return None

    def _get_mbart_pipeline(self):
        if self.mbart_pipeline is not None:
            return self.mbart_pipeline
        try:
            logging.info("Initializing mBART pipeline...")
            self.mbart_pipeline = get_mbart_pipeline()
            return self.mbart_pipeline
        except Exception as e:
            logging.error("Failed to init mBART pipeline: %s", e)
            return None

    def translate_text(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        if not text: return ""
        if not source_lang:
            try:
                source_lang = detect(text)
            except Exception:
                source_lang = "en"
        
        if source_lang == target_lang: return text

        mbart = self._get_mbart_pipeline()
        if not mbart: return text

        src_code = MBART_LANG_CODES.get(source_lang, "en_XX")
        tgt_code = MBART_LANG_CODES.get(target_lang, "en_XX")

        try:
            out = mbart(text, src_lang=src_code, tgt_lang=tgt_code)
            return out[0]["translation_text"]
        except Exception as e:
            logging.warning("mBART translation failed: %s", e)
            return text

    def build_vector_store(self, domain):
        collection_name = f"medical_rag_{domain}"
        json_path = os.path.join(PROCESSED_DATA_DIR, f"{domain}_chunks.json")
        if not os.path.exists(json_path):
            logging.error("Processed data file not found: %s", json_path)
            return
        try:
            collection = self.vector_store_client.get_collection(name=collection_name)
            if collection.count() > 0:
                logging.info("Vector store for '%s' already exists.", domain)
                return
        except Exception:
            logging.info("No collection found; creating for '%s'.", domain)

        with open(json_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        Chroma.from_texts(
            texts=[d["page_content"] for d in docs],
            embedding=self.embedding_function,
            metadatas=[d.get("metadata", {}) for d in docs],
            collection_name=collection_name,
            client=self.vector_store_client,
        )
        logging.info("Vector store for '%s' built successfully.", domain)

    def _initialize_chains(self):
        chains = {}
        prompt_template = """
        Use ONLY the provided context to answer the question.
        Write the answer in clear, complete sentences.
        If the context does not contain an answer, state: "Based on the available data, I cannot provide an answer."

        Context: {context}
        Question: {question}
        Answer:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        for domain in ["outbreak", "symptom", "misinformation"]:
            try:
                vector_store = Chroma(
                    collection_name=f"medical_rag_{domain}",
                    embedding_function=self.embedding_function,
                    client=self.vector_store_client,
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                chains[domain] = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT},
                )
                logging.info("QA chain for '%s' initialized.", domain)
            except Exception as e:
                logging.warning("Could not init chain for '%s': %s", domain, e)
        return chains

    def query(self, user_query: str, domain: str, debug: bool = True):
        trace = {"input_query": user_query, "domain": domain}

        if domain not in self.chains:
            return {"result": "Error", "source_documents": [], "trace": trace}

        try:
            original_lang = detect(user_query)
            trace["detected_lang"] = original_lang
            
            if original_lang not in MBART_LANG_CODES:
                english_query = user_query
                original_lang = "en"
                trace["translation_status"] = "Unsupported language, treated as English"
            elif original_lang == "en":
                english_query = user_query
                trace["translation_status"] = "Already English"
            else:
                english_query = self.translate_text(user_query, "en", source_lang=original_lang)
                trace["translation_status"] = f"Translated from {original_lang} to en"

        except Exception as e:
            logging.warning(f"Language detection failed: {e}. Assuming English.")
            original_lang = "en"
            english_query = user_query
            trace["detected_lang"] = "en (detection failed)"
            trace["translation_status"] = "Assumed English"
        
        trace["english_query"] = english_query

        try:
            response = self.chains[domain].invoke({"query": english_query})
            trace["llm_raw_output"] = response.get("result", "")

            if original_lang != "en":
                translated = self.translate_text(response["result"], original_lang, source_lang="en")
                response["result"] = translated
                trace["translated_result"] = translated
            else:
                trace["translated_result"] = response.get("result", "")

            if debug:
                response["trace"] = trace
                
            return response
        except Exception as e:
            logging.error("Error during RAG query: %s", e)
            trace["error"] = str(e)
            return {"result": "Sorry, an error occurred.", "source_documents": [], "trace": trace}

if __name__ == "__main__":
    rag_system = MedicalRAG()
    for domain in ["outbreak", "symptom", "misinformation"]:
        rag_system.build_vector_store(domain)

    print("\nInteractive Medical RAG (type 'exit' to quit)\n")
    while True:
        q = input("Query: ")
        if q.lower().strip() == "exit":
            break
        res = rag_system.query(q, domain="symptom")
        print("\nAnswer:", res["result"])
        if "trace" in res:
            print("\nTrace:")
            for k, v in res["trace"].items():
                print(f"- {k}: {v}")
        print("\nSources:")
        if res.get("source_documents"):
            for s in res["source_documents"]:
                print("Source:", s.metadata, "\nContent:", s.page_content[:200], "...\n")