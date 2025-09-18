import torch
torch.set_num_threads(1)  # Prevents segfaults on macOS with Streamlit + transformers

import os
import json
import logging
import requests
import chromadb
import re
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any

# LangChain / Vector / LLM Imports
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

# Language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results

# Transformers for translation
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = str(BASE_DIR / "data" / "processed")
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
PERSIST_DIRECTORY = str(BASE_DIR / "vector_db")
LLM_MODEL_ID = "mistral-7b-instruct"

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

# Medical symptom keywords in different languages for accurate detection
MEDICAL_KEYWORDS = {
    "ml": ["വേദന", "പനി", "ഛർദ്ദി", "തലവേദന", "അസുഖം", "രോഗം", "ലക്ഷണം", "പുറം", "മുഖം", "ഛർദി", "വാന்தി"],
    "ta": ["வலி", "காய்ச்சல்", "வாந்தி", "தலைவலி", "நோய்", "அசௌகரியம்", "அறிகுறி", "முதுகு", "முகம்", "வாந்திபோடு"],
    "hi": ["दर्द", "बुखार", "उल्टी", "सिरदर्द", "बीमारी", "तकलीफ", "लक्षण", "पीठ", "चेहरा", "वमन"],
    "en": ["pain", "fever", "vomit", "headache", "sickness", "discomfort", "symptom", "back", "face", "nausea"]
}

# Common medical condition mappings for better translation
MEDICAL_TERM_MAPPING = {
    "ml": {"വേദന": "pain", "പുറം വേദന": "back pain", "തലവേദന": "headache", "ഛർദ്ദി": "vomiting"},
    "ta": {"வலி": "pain", "முதுகுவலி": "back pain", "தலைவலி": "headache", "வாந்தி": "vomiting"},
    "hi": {"दर्द": "pain", "पीठ दर्द": "back pain", "सिरदर्द": "headache", "उल्टी": "vomiting"}
}

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
        """Accurate language detection with medical keyword matching"""
        if not text or len(text.strip()) < 3:
            return "en"
        
        # First try keyword-based detection for medical terms (more reliable)
        text_lower = text.lower()
        for lang_code, keywords in MEDICAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    logging.info(f"Detected language {lang_code} via medical keyword '{keyword}'")
                    return lang_code
        
        # Then use langdetect for general detection
        try:
            detected_lang = detect(text)
            logging.info(f"langdetect identified language: {detected_lang}")
            
            # Map to supported languages and handle variations
            lang_mapping = {
                'zh-cn': 'zh', 'zh-tw': 'zh', 
                'pt-br': 'pt', 'pt-pt': 'pt',
                'ja': 'ja', 'ko': 'ko', 'ru': 'ru',
                'ar': 'ar', 'es': 'es', 'fr': 'fr',
                'de': 'de', 'it': 'it', 'nl': 'nl'
            }
            
            final_lang = lang_mapping.get(detected_lang, detected_lang)
            
            # Only return if it's a supported language
            if final_lang in MBART_LANG_CODES:
                return final_lang
            else:
                logging.warning(f"Detected language {final_lang} not in supported MBART languages")
                return "en"
                
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
        
        return "en"

    def translate_medical_text(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Improved translation with medical term handling"""
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
            # For medical terms, try direct mapping first
            if source_lang in MEDICAL_TERM_MAPPING and target_lang == "en":
                for local_term, english_term in MEDICAL_TERM_MAPPING[source_lang].items():
                    if local_term in text:
                        text = text.replace(local_term, english_term)
                        logging.info(f"Translated medical term: {local_term} -> {english_term}")
            
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
        # Improved prompt template to prevent hallucinations
        prompt_template = """
        You are a medical assistant. Use ONLY the provided context to answer the question.
        If the context does not contain information about the specific symptom or condition mentioned, 
        state clearly: "Based on the available data, I cannot provide specific information about [the mentioned symptom/condition]."
        
        Do not make up information or provide information about unrelated conditions.
        Be precise and only discuss what is directly relevant to the question.

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
            return {"result": "Error: Domain not available", "source_documents": [], "trace": trace}

        try:
            # Detect language of the query
            original_lang = self.detect_language(user_query)
            trace["detected_lang"] = original_lang
            
            # If target language is not specified, use the detected language
            if target_lang is None:
                target_lang = original_lang
                
            # Translate to English if needed for processing
            if original_lang != "en":
                english_query = self.translate_medical_text(user_query, "en", source_lang=original_lang)
                trace["translation_status"] = f"Translated from {original_lang} to en: {english_query}"
            else:
                english_query = user_query
                trace["translation_status"] = "Already English"
            
            trace["english_query"] = english_query

            # Process the query
            response = self.chains[domain].invoke({"query": english_query})
            english_result = response.get("result", "No response generated")
            trace["llm_raw_output"] = english_result

            # Translate response back to target language if needed
            if target_lang != "en":
                translated = self.translate_medical_text(english_result, target_lang, source_lang="en")
                response["result"] = translated
                trace["translated_result"] = translated
            else:
                trace["translated_result"] = english_result

            if debug:
                response["trace"] = trace
                
            return response
        except Exception as e:
            logging.error("Error during RAG query: %s", e)
            trace["error"] = str(e)
            return {"result": "Sorry, an error occurred while processing your query.", "source_documents": [], "trace": trace}

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