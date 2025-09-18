import torch
torch.set_num_threads(1)

import os
import json
import logging
import requests
import chromadb
from pathlib import Path
from typing import Optional

# LangChain / Vector / LLM Imports
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

# --- NEW: Use Google Translate for reliability ---
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration (remains the same) ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = str(BASE_DIR / "data" / "processed")
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
PERSIST_DIRECTORY = str(BASE_DIR / "vector_db")
LLM_MODEL_ID = "mistral-7b-instruct"
MODEL_DIR = str(BASE_DIR / "models")
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_URL = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

class MedicalRAG:
    def __init__(self):
        logging.info("Initializing MedicalRAG system...")
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._ensure_model_downloaded()

        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.llm = self._initialize_llm()
        
        # --- NEW: Initialize the Google Translator ---
        self.translator = Translator()
        
        self.chains = self._initialize_chains()
        logging.info("MedicalRAG system initialized.")

    def _ensure_model_downloaded(self):
        if os.path.exists(MODEL_PATH):
            logging.info("Local GGUF model already present; skipping download.")
            return
        logging.info("Downloading GGUF model...")
        try:
            with requests.get(MODEL_URL, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logging.info("Model downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")

    def _initialize_llm(self):
        try:
            if not os.path.exists(MODEL_PATH):
                logging.warning("GGUF model not found; skipping LLM init.")
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

    # --- NEW: Simpler and more reliable translation function ---
    def translate_text(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        if not text or source_lang == target_lang:
            return text
        try:
            translated = self.translator.translate(text, dest=target_lang, src=source_lang)
            return translated.text
        except Exception as e:
            logging.error(f"Google Translate failed: {e}. Returning original text.")
            return text

    def build_vector_store(self, domain):
        # This function remains the same
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
        # This function remains the same
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
                    llm=self.llm, chain_type="stuff", retriever=retriever,
                    return_source_documents=True, chain_type_kwargs={"prompt": PROMPT},
                )
                logging.info("QA chain for '%s' initialized.", domain)
            except Exception as e:
                logging.warning("Could not init chain for '%s': %s", domain, e)
        return chains

    # --- NEW: Updated query function to use Google Translate ---
    def query(self, user_query: str, domain: str, debug: bool = True):
        trace = {"input_query": user_query, "domain": domain}
        if domain not in self.chains:
            return {"result": "Error", "source_documents": [], "trace": trace}

        try:
            detected = self.translator.detect(user_query)
            original_lang = detected.lang
            trace["detected_lang"] = f"{original_lang} (confidence: {detected.confidence})"
        except Exception:
            original_lang = "en"
            trace["detected_lang"] = "en (detection failed)"

        if original_lang != "en":
            english_query = self.translate_text(user_query, "en", source_lang=original_lang)
        else:
            english_query = user_query
        trace["english_query"] = english_query

        try:
            response = self.chains[domain].invoke({"query": english_query})
            if original_lang != "en" and "result" in response:
                translated_result = self.translate_text(response["result"], original_lang, source_lang="en")
                response["result"] = translated_result
            
            if debug: response["trace"] = trace
            return response
        except Exception as e:
            logging.error("Error during RAG query: %s", e)
            return {"result": "Sorry, an error occurred.", "source_documents": [], "trace": trace}

if __name__ == "__main__":
    rag_system = MedicalRAG()
    for domain in ["outbreak", "symptom", "misinformation"]:
        rag_system.build_vector_store(domain)