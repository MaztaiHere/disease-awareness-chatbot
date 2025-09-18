import torch
torch.set_num_threads(1)  # Prevents segfaults on macOS with Streamlit + transformers

import os
import json
import logging
import requests
import chromadb
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any

# LangChain / Vector / LLM Imports
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

# Improved language detection
from lingua import Language, LanguageDetectorBuilder
from transformers import pipeline, MBart50TokenizerFast, MBartForConditionalGeneration

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

# mBART Supported Language Codes - Expanded with more languages
MBART_LANG_CODES = {
    "en": "en_XX", "hi": "hi_IN", "ta": "ta_IN", "te": "te_IN", "ml": "ml_IN",
    "bn": "bn_IN", "gu": "gu_IN", "mr": "mr_IN", "kn": "kn_IN", "pa": "pa_IN",
    "es": "es_XX", "fr": "fr_XX", "de": "de_DE", "it": "it_IT", "ja": "ja_XX",
    "ru": "ru_RU", "zh": "zh_CN", "ar": "ar_AR", "pt": "pt_XX", "tr": "tr_TR",
    "vi": "vi_VN", "ko": "ko_KR", "nl": "nl_XX", "uk": "uk_UA", "pl": "pl_PL"
}

# Language name mapping for UI
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
    "bn": "Bengali", "gu": "Gujarati", "mr": "Marathi", "kn": "Kannada", "pa": "Punjabi",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "ja": "Japanese",
    "ru": "Russian", "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese", "tr": "Turkish",
    "vi": "Vietnamese", "ko": "Korean", "nl": "Dutch", "uk": "Ukrainian", "pl": "Polish"
}

# Initialize language detector with common languages
language_detector = LanguageDetectorBuilder.from_all_languages().build()

class MedicalRAG:
    """
    Retrieval-Augmented Generation system for medical/public health.
    - Local LLM (via llama-cpp GGUF)
    - Chroma vector DB with HuggingFace embeddings
    - Translation handled with mBART
    """

    def __init__(self):
        logging.info("Initializing MedicalRAG system...")
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        self._ensure_model_downloaded()

        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.llm = self._initialize_llm()
        self.mbart_model = None
        self.mbart_tokenizer = None
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
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded / total * 100
                                if downloaded % (1024 * 1024 * 50) < 8192:
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

    def _initialize_mbart(self):
        """Initialize mBART model and tokenizer for translation"""
        if self.mbart_model is not None and self.mbart_tokenizer is not None:
            return
            
        try:
            logging.info("Loading mBART model and tokenizer...")
            self.mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            self.mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            logging.info("mBART model and tokenizer loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load mBART model: {e}")
            self.mbart_model = None
            self.mbart_tokenizer = None

    def detect_language(self, text: str) -> str:
        """Improved language detection using lingua"""
        if not text or len(text.strip()) < 3:
            return "en"  # Default to English for very short texts
            
        try:
            # Use lingua for more reliable detection
            detected_language = language_detector.detect_language_of(text)
            if detected_language:
                # Map lingua language to ISO code
                language_map = {
                    Language.ENGLISH: "en",
                    Language.HINDI: "hi",
                    Language.TAMIL: "ta",
                    Language.TELUGU: "te",
                    Language.MALAYALAM: "ml",
                    Language.BENGALI: "bn",
                    Language.GUJARATI: "gu",
                    Language.MARATHI: "mr",
                    Language.KANNADA: "kn",
                    Language.PUNJABI: "pa",
                    Language.SPANISH: "es",
                    Language.FRENCH: "fr",
                    Language.GERMAN: "de",
                    Language.ITALIAN: "it",
                    Language.JAPANESE: "ja",
                    Language.RUSSIAN: "ru",
                    Language.CHINESE: "zh",
                    Language.ARABIC: "ar",
                    Language.PORTUGUESE: "pt",
                    Language.TURKISH: "tr",
                    Language.VIETNAMESE: "vi",
                    Language.KOREAN: "ko",
                    Language.DUTCH: "nl",
                    Language.UKRAINIAN: "uk",
                    Language.POLISH: "pl"
                }
                return language_map.get(detected_language, "en")
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
            
        return "en"  # Fallback to English

    def translate_text(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Translate text using mBART model"""
        if not text or target_lang == source_lang:
            return text
            
        # Initialize mBART if not already done
        self._initialize_mbart()
        
        if self.mbart_model is None or self.mbart_tokenizer is None:
            logging.error("mBART not available for translation")
            return text
            
        # Determine source language if not provided
        if source_lang is None:
            source_lang = self.detect_language(text)
            
        # Check if both languages are supported
        if source_lang not in MBART_LANG_CODES or target_lang not in MBART_LANG_CODES:
            logging.warning(f"Unsupported language pair: {source_lang} -> {target_lang}")
            return text
            
        try:
            # Set source and target language codes
            src_code = MBART_LANG_CODES[source_lang]
            tgt_code = MBART_LANG_CODES[target_lang]
            
            # Tokenize input text
            self.mbart_tokenizer.src_lang = src_code
            encoded_input = self.mbart_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate translation
            generated_tokens = self.mbart_model.generate(
                **encoded_input,
                forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id[tgt_code],
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode the translation
            translation = self.mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation
            
        except Exception as e:
            logging.error(f"Translation failed: {e}")
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

    def query(self, user_query: str, domain: str, target_lang: Optional[str] = None, debug: bool = True):
        trace = {"input_query": user_query, "domain": domain}

        if domain not in self.chains:
            return {"result": "Error", "source_documents": [], "trace": trace}

        try:
            # Detect language of the query
            original_lang = self.detect_language(user_query)
            trace["detected_lang"] = original_lang
            
            # If target language is not specified, use the detected language
            if target_lang is None:
                target_lang = original_lang
                
            # Translate to English if needed for processing
            if original_lang != "en":
                english_query = self.translate_text(user_query, "en", source_lang=original_lang)
                trace["translation_status"] = f"Translated from {original_lang} to en"
            else:
                english_query = user_query
                trace["translation_status"] = "Already English"
            
            trace["english_query"] = english_query

            # Process the query
            response = self.chains[domain].invoke({"query": english_query})
            trace["llm_raw_output"] = response.get("result", "")

            # Translate response back to target language if needed
            if target_lang != "en":
                translated = self.translate_text(response["result"], target_lang, source_lang="en")
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
        lang = input("Target language (en, hi, es, etc.) or press Enter for auto-detection: ").strip()
        if not lang:
            lang = None
        res = rag_system.query(q, domain="symptom", target_lang=lang)
        print("\nAnswer:", res["result"])
        if "trace" in res:
            print("\nTrace:")
            for k, v in res["trace"].items():
                print(f"- {k}: {v}")
        print("\nSources:")
        if res.get("source_documents"):
            for s in res["source_documents"]:
                print("Source:", s.metadata, "\nContent:", s.page_content[:200], "...\n")