# src/rag_core.py

import os
import json
import logging
import requests
import chromadb
from pathlib import Path

# LangChain / Vector / LLM Imports
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

# Other imports
from langdetect import detect
from transformers import pipeline
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration (Paths corrected for src/ location) ---
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


class MedicalRAG:
    """
    A class to manage the Retrieval-Augmented Generation pipeline for the medical domains.
    Uses:
      - local LLM via llama-cpp (quantized GGUF)
      - Chroma vector DB with HuggingFace embeddings
      - lazy translation fallback to mBART for languages not covered by specified Helsinki models
    """

    def __init__(self):
        logging.info("Initializing MedicalRAG system...")
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Ensure local quantized model is available
        self._ensure_model_downloaded()

        # embeddings and vector client
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

        # initialize local LLM
        self.llm = self._initialize_llm()

        # translation maps & lazy translation pipeline
        self._initialize_translator()

        # initialize retrieval chains
        self.chains = self._initialize_chains()
        logging.info("MedicalRAG system initialized.")

    # -------------------------
    # Model Download & LLM Init
    # -------------------------
    def _ensure_model_downloaded(self):
        """Download model if missing. Resilient: if download fails, we log and continue."""
        if os.path.exists(MODEL_PATH):
            logging.info("Local GGUF model already present; skipping download.")
            return

        logging.info("Local GGUF model not found; attempting to download (this may take a while)...")
        try:
            with requests.get(MODEL_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # optional simple progress log
                            if total:
                                pct = downloaded / total * 100
                                if downloaded % (1024 * 1024 * 10) < 8192:
                                    logging.info(f"Downloaded {pct:.1f}%")
            logging.info("Model downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download model automatically: {e}")
            logging.error("Please download the GGUF model manually and place it at: %s", MODEL_PATH)

    def _initialize_llm(self):
        """Initialize a local LLM via LlamaCpp wrapper.
        Uses conservative settings to reduce repetition and ambiguity.
        """
        try:
            if not os.path.exists(MODEL_PATH):
                logging.warning("GGUF model not found at %s; LLM initialization skipped.", MODEL_PATH)
                return None

            # n_threads and other parameters can be tuned; M4 Mac usually benefits from several threads but keep conservative defaults
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_threads=8,
                n_batch=256,
                temperature=0.0,        # deterministic
                top_p=0.9,
                repeat_penalty=1.15,   # reduce repetitive phrasing
                max_tokens=512,
                verbose=False,
            )
            logging.info("Local LLM initialized via LlamaCpp.")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize local LLM: {e}")
            return None

    # -------------------------
    # Translation (lazy + fallback)
    # -------------------------
    def _initialize_translator(self):
        logging.info("Initializing translator maps...")

        # Explicit en <-> lang mappings (offline HuggingFace models)
        self.lang_to_model_map = {
            # Indian languages
            "ta": "Helsinki-NLP/opus-mt-ta-en",  # Tamil
            "ml": "Helsinki-NLP/opus-mt-ml-en",  # Malayalam
            "kn": "Helsinki-NLP/opus-mt-kn-en",  # Kannada
            "te": "Helsinki-NLP/opus-mt-te-en",  # Telugu
            "mr": "Helsinki-NLP/opus-mt-mr-en",  # Marathi
            "hi": "Helsinki-NLP/opus-mt-hi-en",  # Hindi
            "gu": "Helsinki-NLP/opus-mt-gu-en",  # Gujarati
            "bn": "Helsinki-NLP/opus-mt-bn-en",  # Bengali
            "pa": "Helsinki-NLP/opus-mt-pa-en",  # Punjabi
            "as": "Helsinki-NLP/opus-mt-as-en",  # Assamese

            # Foreign languages
            "es": "Helsinki-NLP/opus-mt-es-en",  # Spanish
            "de": "Helsinki-NLP/opus-mt-de-en",  # German
            "it": "Helsinki-NLP/opus-mt-it-en",  # Italian
            "fr": "Helsinki-NLP/opus-mt-fr-en",  # French
            "ja": "Helsinki-NLP/opus-mt-ja-en",  # Japanese
        }

        # Reverse maps (English → other lang)
        self.reverse_lang_to_model_map = {
            "ta": "Helsinki-NLP/opus-mt-en-ta",
            "ml": "Helsinki-NLP/opus-mt-en-ml",
            "kn": "Helsinki-NLP/opus-mt-en-kn",
            "te": "Helsinki-NLP/opus-mt-en-te",
            "mr": "Helsinki-NLP/opus-mt-en-mr",
            "hi": "Helsinki-NLP/opus-mt-en-hi",
            "gu": "Helsinki-NLP/opus-mt-en-gu",
            "bn": "Helsinki-NLP/opus-mt-en-bn",
            "pa": "Helsinki-NLP/opus-mt-en-pa",
            "as": "Helsinki-NLP/opus-mt-en-as",

            "es": "Helsinki-NLP/opus-mt-en-es",
            "de": "Helsinki-NLP/opus-mt-en-de",
            "it": "Helsinki-NLP/opus-mt-en-it",
            "fr": "Helsinki-NLP/opus-mt-en-fr",
            "ja": "Helsinki-NLP/opus-mt-en-ja",
        }

        # Cache + lazy mbart
        self.translation_cache = {}
        self.mbart_pipeline: Optional[pipeline] = None
        logging.info("Translator maps initialized. All models are local HuggingFace models.")

    def _get_mbart_pipeline(self):
        """Create and cache mBART pipeline for many-to-many translation fallback."""
        if self.mbart_pipeline is not None:
            return self.mbart_pipeline
        try:
            logging.info("Initializing fallback mBART translation pipeline (facebook/mbart-large-50-many-to-many-mmt)...")
            self.mbart_pipeline = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")
            return self.mbart_pipeline
        except Exception as e:
            logging.error("Failed to initialize mBART pipeline: %s", e)
            return None

    def _load_translation_pipeline_for(self, model_name):
        """Load a transformers pipeline and cache it."""
        if not model_name:
            return None
        if model_name in self.translation_cache:
            return self.translation_cache[model_name]
        try:
            logging.info("Loading translation pipeline: %s", model_name)
            p = pipeline("translation", model=model_name)
            self.translation_cache[model_name] = p
            return p
        except Exception as e:
            logging.warning("Could not load model %s: %s", model_name, e)
            return None

    def _translate_generic(self, text, source_lang, target_lang):
        """Translate text from source_lang to target_lang. Uses Helsinki models when available; falls back to mBART."""
        if not text:
            return text

        # If both languages are english/no-op
        if source_lang == target_lang:
            return text

        # Attempt to use a Helsinki-style model from map
        try:
            if source_lang == "en":
                model_name = self.lang_to_model_map.get(target_lang)
            elif target_lang == "en":
                model_name = self.reverse_lang_to_model_map.get(source_lang)
            else:
                model_name = None

            if model_name:
                p = self._load_translation_pipeline_for(model_name)
                if p:
                    return p(text)[0]["translation_text"]
        except Exception as e:
            logging.debug(f"Helsinki translation attempt failed: {e}")

        # Fallback: use mBART many-to-many (handles many languages including Indic)
        mbart = self._get_mbart_pipeline()
        if mbart:
            try:
                # For mBART you often specify src & tgt via task string in the pipeline config.
                # The Hugging Face pipeline accepts "src_lang" & "tgt_lang" in the model card for mbart.
                return mbart(text, src_lang=source_lang, tgt_lang=target_lang)[0]["translation_text"]
            except Exception as e:
                logging.warning("mBART pipeline translation failed: %s", e)

        # If everything fails, return original text and let caller handle it
        logging.warning("Translation fallback failed; returning original text.")
        return text

    def translate_text(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Public translation wrapper. Detects source_lang if not provided."""
        try:
            if not source_lang:
                source_lang = detect(text)
        except Exception:
            source_lang = "en"
        try:
            return self._translate_generic(text, source_lang, target_lang)
        except Exception as e:
            logging.error("Translation failed: %s", e)
            return text

    # -------------------------
    # Vector Store & Build
    # -------------------------
    def build_vector_store(self, domain):
        collection_name = f"medical_rag_{domain}"
        json_path = os.path.join(PROCESSED_DATA_DIR, f"{domain}_chunks.json")
        if not os.path.exists(json_path):
            logging.error("Processed data file not found for domain '%s' at %s", domain, json_path)
            return
        try:
            collection = self.vector_store_client.get_collection(name=collection_name)
            if collection.count() > 0:
                logging.info("Vector store for '%s' already exists. Skipping rebuild.", domain)
                return
        except Exception:
            logging.info("Collection not found; building new collection for '%s'.", domain)

        with open(json_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        Chroma.from_texts(
            texts=[d["page_content"] for d in docs],
            embedding=self.embedding_function,
            metadatas=[d.get("metadata", {}) for d in docs],
            collection_name=collection_name,
            client=self.vector_store_client,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Vector store for '%s' built successfully.", domain)

    # -------------------------
    # QA Chains
    # -------------------------
    def _initialize_chains(self):
        chains = {}

        prompt_template = """
You are a medical and public health assistant.
Use ONLY the provided context to answer.
Answer in a **concise, structured, and tabular format**.
Do not write long paragraphs. Use bullet points or tables where possible.
If the context does not contain an answer, say:
"Based on the available data, I cannot provide an answer."

Context:
{context}

Question:
{question}

Answer (structured):
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        for domain in ["outbreak", "symptom", "misinformation"]:
            try:
                vector_store = Chroma(
                    collection_name=f"medical_rag_{domain}",
                    embedding_function=self.embedding_function,
                    client=self.vector_store_client,
                    persist_directory=PERSIST_DIRECTORY,
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                chains[domain] = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT},
                )
                logging.info("QA chain for domain '%s' initialized.", domain)
            except Exception as e:
                logging.warning("Could not initialize chain for domain '%s'. Error: %s", domain, e)
        return chains

    # -------------------------
    # Query
    # -------------------------
    def query(self, user_query: str, domain: str):
        """Run a RAG query: translate to English if needed, run chain, translate back."""
        if domain not in self.chains:
            return {"result": f"Error: The '{domain}' domain is not ready.", "source_documents": []}

        # Detect language and translate to English for retrieval & LLM prompting
        try:
            original_lang = detect(user_query)
        except Exception:
            original_lang = "en"

        if original_lang != "en":
            english_query = self.translate_text(user_query, "en", source_lang=original_lang)
        else:
            english_query = user_query

        try:
            response = self.chains[domain].invoke({"query": english_query})
            # The RetrievalQA returns fields like 'result' and 'source_documents'
            if original_lang != "en" and isinstance(response, dict) and "result" in response:
                # translate result back to user language
                response["result"] = self.translate_text(response["result"], original_lang, source_lang="en")
            return response
        except Exception as e:
            logging.error("Error during RAG query: %s", e)
            return {"result": "Sorry, an error occurred. Please try again.", "source_documents": []}


if __name__ == "__main__":
    # Build vector stores on demand (only when script executed directly)
    rag_system = MedicalRAG()
    for domain in ["outbreak", "symptom", "misinformation"]:
        rag_system.build_vector_store(domain)
