import torch
torch.set_num_threads(1)
import os
import json
import logging
import requests
import chromadb
import re
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llams import LlamaCpp
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = str(BASE_DIR / "data" / "processed")
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
PERSIST_DIRECTORY = str(BASE_DIR / "vector_db")

# mBART Supported Language Codes
MBART_LANG_CODES = {
    "en": "en_XX", "hi": "hi_IN", "ta": "ta_IN", "te": "te_IN", "ml": "ml_IN",
    "bn": "bn_IN", "gu": "gu_IN", "mr": "mr_IN", "kn": "kn_IN", "pa": "pa_IN",
    "es": "es_XX", "fr": "fr_XX", "de": "de_DE", "it": "it_IT", "ja": "ja_XX",
    "ru": "ru_RU", "zh": "zh_CN", "ar": "ar_AR", "pt": "pt_XX", "tr": "tr_TR",
}

# Language name mapping for UI - ADD THIS MISSING CONSTANT
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
    "bn": "Bengali", "gu": "Gujarati", "mr": "Marathi", "kn": "Kannada", "pa": "Punjabi",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "ja": "Japanese",
    "ru": "Russian", "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese", "tr": "Turkish",
}

class MedicalRAG:
    def __init__(self):
        logging.info("Initializing MedicalRAG system...")
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.llm = self._initialize_llm()
        self.mbart_model = None
        self.mbart_tokenizer = None
        self.chains = self._initialize_chains()
        logging.info("MedicalRAG system initialized.")

    def _initialize_llm(self):
        try:
            model_path = os.path.join(BASE_DIR / "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
            if not os.path.exists(model_path):
                logging.warning("GGUF model not found; skipping LLM init.")
                return None

            llm = LlamaCpp(
                model_path=model_path, n_ctx=4096, n_threads=8, n_batch=512,
                temperature=0.1, top_p=0.9, repeat_penalty=1.15, max_tokens=512, verbose=False,
            )
            logging.info("Local LLM initialized via LlamaCpp.")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize local LLM: {e}")
            return None

    def _initialize_mbart(self):
        if self.mbart_model is not None:
            return
        try:
            logging.info("Loading mBART model and tokenizer...")
            self.mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            self.mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            logging.info("mBART model and tokenizer loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load mBART model: {e}")

    def detect_language(self, text: str) -> str:
        """Robust language detection with fallback mechanisms"""
        if not text or len(text.strip()) < 3:
            return "en"
        
        try:
            # First pass with langdetect
            detected_lang = detect(text)
            
            # Validate if detected language is supported
            if detected_lang in MBART_LANG_CODES:
                return detected_lang
            
            # Second pass: Check for specific language patterns
            if any(char in text for char in ["ം", "ഃ", "അ"]):  # Malayalam characters
                return "ml"
            elif any(char in text for char in ["ஂ", "ஃ", "அ"]):  # Tamil characters
                return "ta"
            elif any(char in text for char in ["ँ", "ं", "ः"]):  # Hindi characters
                return "hi"
                
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
        
        return "en"

    def translate_with_fallback(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Robust translation with multiple fallback strategies"""
        if not text or target_lang == source_lang:
            return text
            
        if source_lang is None:
            source_lang = self.detect_language(text)
            
        # Don't translate if either language isn't supported
        if source_lang not in MBART_LANG_CODES or target_lang not in MBART_LANG_CODES:
            return text
            
        # Initialize translation model
        self._initialize_mbart()
        if self.mbart_model is None:
            return text
            
        try:
            src_code = MBART_LANG_CODES[source_lang]
            tgt_code = MBART_LANG_CODES[target_lang]
            
            self.mbart_tokenizer.src_lang = src_code
            encoded_input = self.mbart_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            generated_tokens = self.mbart_model.generate(
                **encoded_input,
                forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id[tgt_code],
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            return self.mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return text

    def build_vector_store(self, domain):
        collection_name = f"medical_rag_{domain}"
        json_path = os.path.join(PROCESSED_DATA_DIR, f"{domain}_chunks.json")
        if not os.path.exists(json_path):
            logging.error("Processed data file not found: %s", json_path)
            return
            
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
        You are a medical assistant. Use ONLY the provided context to answer the question.
        If the context does not contain information about the specific symptom or condition mentioned, 
        state clearly: "Based on the available data, I cannot provide specific information about this."

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
            except Exception as e:
                logging.warning("Could not init chain for '%s': %s", domain, e)
        return chains

    def query(self, user_query: str, domain: str, target_lang: Optional[str] = None, debug: bool = True):
        trace = {"input_query": user_query, "domain": domain}

        if domain not in self.chains:
            return {"result": "Error: Domain not available", "source_documents": [], "trace": trace}

        try:
            # Detect language
            original_lang = self.detect_language(user_query)
            trace["detected_lang"] = original_lang
            
            if target_lang is None:
                target_lang = original_lang
                
            # Translate query to English for processing
            if original_lang != "en":
                english_query = self.translate_with_fallback(user_query, "en", original_lang)
                trace["translation_status"] = f"Translated from {original_lang} to en: {english_query}"
            else:
                english_query = user_query
                trace["translation_status"] = "Already English"
            
            trace["english_query"] = english_query

            # Process query
            response = self.chains[domain].invoke({"query": english_query})
            english_result = response.get("result", "No response generated")
            trace["llm_raw_output"] = english_result

            # Translate response back if needed
            if target_lang != "en":
                translated = self.translate_with_fallback(english_result, target_lang, "en")
                response["result"] = translated
                trace["translated_result"] = translated
            else:
                trace["translated_result"] = english_result

            if debug:
                response["trace"] = trace
                
            return response
            
        except Exception as e:
            logging.error("Error during RAG query: %s", e)
            return {"result": "Sorry, an error occurred.", "source_documents": [], "trace": trace}