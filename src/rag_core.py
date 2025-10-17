import torch
torch.set_num_threads(1)

import os
import sys
import json
import logging
import requests
import chromadb
from chromadb.config import Settings
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any, List

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.schema import Document

from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# ======================================================
# CONFIGURATION
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = str(BASE_DIR / "data" / "processed")
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
PERSIST_DIRECTORY = str(BASE_DIR / "vector_db")
MODEL_DIR = str(BASE_DIR / "models")
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_URL = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# ======================================================
# LANGUAGE MAPPINGS
# ======================================================

MBART_LANG_CODES = {
    "af": "af_ZA", "ar": "ar_AR", "az": "az_AZ", "bn": "bn_IN", "my": "my_MM",
    "zh": "zh_CN", "hr": "hr_HR", "cs": "cs_CZ", "nl": "nl_XX", "en": "en_XX",
    "et": "et_EE", "fi": "fi_FI", "fr": "fr_XX", "gl": "gl_ES", "ka": "ka_GE",
    "de": "de_DE", "gu": "gu_IN", "he": "he_IL", "hi": "hi_IN", "hu": "hu_HU",
    "is": "is_IS", "id": "id_ID", "it": "it_IT", "ja": "ja_XX", "kn": "kn_IN",
    "kk": "kk_KZ", "km": "km_KH", "ko": "ko_KR", "lv": "lv_LV", "lt": "lt_LT",
    "mk": "mk_MK", "ml": "ml_IN", "mr": "mr_IN", "mn": "mn_MN", "ne": "ne_NP",
    "pl": "pl_PL", "pt": "pt_XX", "ro": "ro_RO", "ru": "ru_RU", "si": "si_LK",
    "sk": "sk_SK", "sl": "sl_SI", "es": "es_XX", "sw": "sw_KE", "sv": "sv_SE",
    "ta": "ta_IN", "te": "te_IN", "th": "th_TH", "tr": "tr_TR", "uk": "uk_UA"
}

LANGUAGE_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "az": "Azerbaijani", "bn": "Bengali", "my": "Burmese",
    "zh": "Chinese", "hr": "Croatian", "cs": "Czech", "nl": "Dutch", "en": "English",
    "et": "Estonian", "fi": "Finnish", "fr": "French", "gl": "Galician", "ka": "Georgian",
    "de": "German", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian",
    "is": "Icelandic", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kn": "Kannada",
    "kk": "Kazakh", "km": "Khmer", "ko": "Korean", "lv": "Latvian", "lt": "Lithuanian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "mn": "Mongolian", "ne": "Nepali",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "si": "Sinhala",
    "sk": "Slovak", "sl": "Slovenian", "es": "Spanish", "sw": "Swahili", "sv": "Swedish",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian"
}

# ======================================================
# CLASS DEFINITION
# ======================================================

class MedicalRAG:
    def __init__(self):
        logging.info("🚀 Initializing MedicalRAG system...")
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._ensure_model_downloaded()
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store_client = chromadb.PersistentClient(
            path=PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        self.llm = self._initialize_llm()
        self.mbart_model = None
        self.mbart_tokenizer = None
        self.vector_stores = self._initialize_vector_stores()
        self.chains = self._initialize_chains()
        logging.info("✅ MedicalRAG system initialized successfully.\n")

    def _ensure_model_downloaded(self):
        if os.path.exists(MODEL_PATH):
            logging.info("🧠 Local GGUF model already present.")
            return
        logging.info("⬇️ Downloading GGUF model...")
        try:
            with requests.get(MODEL_URL, stream=True, timeout=600) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            logging.info("✅ Model downloaded successfully.")
        except Exception as e:
            logging.error(f"❌ Failed to download model: {e}")

    def _initialize_llm(self):
        try:
            if not os.path.exists(MODEL_PATH):
                logging.warning("⚠️ GGUF model not found, skipping LLM init.")
                return None
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                n_ctx=4096, n_threads=8, n_batch=512,
                temperature=0.3,  # Slightly increased for more creative responses
                top_p=0.9,
                repeat_penalty=1.15, 
                max_tokens=350,  # Increased to allow 2-3 sentences
                verbose=False,
            )
            logging.info("🦙 Local LLM initialized via LlamaCpp.")
            return llm
        except Exception as e:
            logging.error(f"❌ Failed to initialize LLM: {e}")
            return None

    def _initialize_mbart(self):
        if self.mbart_model and self.mbart_tokenizer:
            return
        try:
            logging.info("🌐 Loading mBART translation model...")
            self.mbart_model = MBartForConditionalGeneration.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt"
            )
            self.mbart_tokenizer = MBart50TokenizerFast.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt"
            )
            logging.info("✅ mBART model loaded.")
        except Exception as e:
            logging.error(f"❌ Failed to load mBART: {e}")
            self.mbart_model, self.mbart_tokenizer = None, None

    def _initialize_vector_stores(self):
        """Initialize vector stores for each domain."""
        vector_stores = {}
        for domain in ["outbreak", "symptom", "misinformation"]:
            try:
                vector_store = Chroma(
                    collection_name=f"medical_rag_{domain}",
                    embedding_function=self.embedding_function,
                    client=self.vector_store_client,
                    persist_directory=PERSIST_DIRECTORY
                )
                vector_stores[domain] = vector_store
                logging.info(f"✅ Vector store ready for '{domain}' domain.")
            except Exception as e:
                logging.warning(f"⚠️ Could not init vector store for {domain}: {e}")
        return vector_stores

    def translate_text(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """Translate text between languages using mBART."""
        if not text.strip() or target_lang == source_lang:
            return text
            
        self._initialize_mbart()
        if not self.mbart_model or not self.mbart_tokenizer:
            logging.warning("mBART not available, returning original text.")
            return text
            
        try:
            # Validate language codes
            if source_lang not in MBART_LANG_CODES:
                logging.warning(f"Unsupported source language: {source_lang}")
                return text
            if target_lang not in MBART_LANG_CODES:
                logging.warning(f"Unsupported target language: {target_lang}")
                return text
                
            src_code = MBART_LANG_CODES[source_lang]
            tgt_code = MBART_LANG_CODES[target_lang]
            
            self.mbart_tokenizer.src_lang = src_code
            encoded = self.mbart_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            
            generated = self.mbart_model.generate(
                **encoded,
                forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id[tgt_code],
                max_length=1024, 
                num_beams=4,
                early_stopping=True
            )
            
            translated = self.mbart_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            logging.info(f"🌍 Translation {source_lang} → {target_lang} completed")
            return translated
            
        except Exception as e:
            logging.error(f"Translation error ({source_lang}→{target_lang}): {e}")
            return text

    def _retrieve_documents(self, query: str, domain: str, k: int = 5) -> List[Document]:
        """Retrieve documents with improved query handling."""
        if domain not in self.vector_stores:
            return []
            
        try:
            # Use similarity search with more documents
            retriever = self.vector_stores[domain].as_retriever(
                search_type="similarity",  # Use simple similarity for broader matching
                search_kwargs={"k": k}
            )
            
            documents = retriever.get_relevant_documents(query)
            logging.info(f"📚 Retrieved documents for query: {query}")
            return documents
            
        except Exception as e:
            logging.error(f"❌ Document retrieval error: {e}")
            return []

    def _initialize_chains(self):
        """Initialize QA chains for each domain with STRICT 2-3 sentence requirement."""
        chains = {}
        
        # STRICT 2-3 SENTENCE PROMPT TEMPLATES
        prompt_templates = {
            "symptom": """
            Provide exactly 2-3 sentences answering the question based on the context.
            If context is limited, provide general medical knowledge in 2-3 sentences.
            NEVER say "I cannot provide an answer" - always give helpful information.
            
            Context: {context}
            Question: {question}
            
            2-3 Sentence Answer:
            """,
            
            "outbreak": """
            Provide exactly 2-3 sentences about disease outbreaks based on the context.
            If context is limited, mention general outbreak patterns in 2-3 sentences.
            ALWAYS provide specific information - never refuse to answer.
            
            Context: {context}
            Question: {question}
            
            2-3 Sentence Outbreak Information:
            """,
            
            "misinformation": """
            Provide exactly 2-3 sentences addressing medical misinformation based on context.
            If context is limited, provide factual health information in 2-3 sentences.
            ALWAYS give a clear, factual response.
            
            Context: {context}
            Question: {question}
            
            2-3 Sentence Factual Response:
            """
        }
        
        for domain in ["outbreak", "symptom", "misinformation"]:
            try:
                prompt_template = prompt_templates.get(domain, prompt_templates["symptom"])
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                
                # Create a simple chain that uses our improved retrieval
                if self.llm:
                    chains[domain] = {
                        "prompt": PROMPT,
                        "llm": self.llm
                    }
                logging.info(f"✅ QA chain ready for '{domain}' domain.")
            except Exception as e:
                logging.warning(f"⚠️ Could not init chain for {domain}: {e}")
        return chains

    def _generate_response(self, query: str, documents: List[Document], domain: str) -> str:
        """Generate response using LLM with STRICT 2-3 sentence requirement."""
        if domain not in self.chains or not self.llm:
            # Fallback response that's exactly 2 sentences
            return "I'm providing general health information. Please consult healthcare professionals for specific advice."
            
        try:
            # Combine document content
            context = "\n\n".join([doc.page_content for doc in documents])
            
            # Get the prompt template
            prompt_template = self.chains[domain]["prompt"]
            formatted_prompt = prompt_template.format(context=context, question=query)
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            cleaned_response = self._enforce_sentence_limit(response.strip())
            
            return cleaned_response
            
        except Exception as e:
            logging.error(f"❌ Response generation error: {e}")
            # Fallback that's exactly 2 sentences
            return "This appears to be a health-related question. It's important to consult medical professionals for accurate information."

    def _enforce_sentence_limit(self, text: str) -> str:
        """Enforce strict 2-3 sentence limit."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Remove empty sentences
        sentences = [s for s in sentences if s]
        
        # If we have sentences, take 2-3
        if sentences:
            # Always return at least 2 sentences, maximum 3
            selected_sentences = sentences[:3]  # Take up to 3 sentences
            if len(selected_sentences) == 1 and len(selected_sentences[0].split()) > 8:
                # If only one long sentence, try to split it
                words = selected_sentences[0].split()
                mid_point = len(words) // 2
                sentence1 = ' '.join(words[:mid_point]) + '.'
                sentence2 = ' '.join(words[mid_point:]) + '.'
                selected_sentences = [sentence1, sentence2]
            elif len(selected_sentences) < 2:
                # If we have less than 2 sentences, create a second one
                selected_sentences.append("Please consult healthcare providers for specific medical advice.")
            
            result = '. '.join(selected_sentences)
            # Ensure it ends with a period
            if not result.endswith('.'):
                result += '.'
            return result
        else:
            # Fallback 2-sentence response
            return "This appears to be a health-related inquiry. It's important to seek professional medical advice for accurate information."

    def query(self, user_query: str, domain: str, source_lang: str = "en", target_lang: str = None):
        """
        Process a user query through the RAG pipeline with GUARANTEED 2-3 sentence responses.
        """
        logging.info("=" * 70)
        logging.info(f"🧠 Query received: {user_query}")
        logging.info(f"🌍 Domain: {domain}, Source Language: {source_lang}")
        
        if not target_lang:
            target_lang = source_lang
            
        logging.info(f"🎯 Target Language: {target_lang}")

        try:
            # Step 1: Translate to English if needed for retrieval
            if source_lang != "en":
                logging.info("🔹 Translating query to English for retrieval...")
                english_query = self.translate_text(user_query, "en", source_lang)
                logging.info(f"→ English Query: {english_query}")
            else:
                english_query = user_query
                logging.info("→ Query is already in English")

            # Step 2: Retrieve documents
            logging.info("🔹 Retrieving relevant documents...")
            documents = self._retrieve_documents(english_query, domain, k=5)
            
            # Step 3: Generate GUARANTEED 2-3 sentence English response
            logging.info("🔹 Generating 2-3 sentence response...")
            raw_answer = self._generate_response(english_query, documents, domain)
            logging.info(f"→ 2-3 Sentence Answer: {raw_answer}")

            # Step 4: Translate back to target language if needed
            if target_lang != "en":
                logging.info(f"🔹 Translating answer to {target_lang}...")
                final_answer = self.translate_text(raw_answer, target_lang, "en")
                logging.info(f"✅ Final Answer ({target_lang}): {final_answer}")
            else:
                final_answer = raw_answer
                logging.info(f"✅ Final Answer (English): {final_answer}")

            # Prepare response with source documents
            response = {
                "result": final_answer,
                "source_documents": documents
            }
            
            logging.info("=" * 70 + "\n")
            return response

        except Exception as e:
            logging.error(f"❌ Error in RAG pipeline: {e}")
            # Even in error, return 2 sentences
            error_msg = "An error occurred while processing your request. Please try again with a different question."
            return {"result": error_msg, "source_documents": []}


# ======================================================
# MAIN EXECUTION (Console Interaction)
# ======================================================

if __name__ == "__main__":
    rag = MedicalRAG()
    
    print("\n🩺 Interactive Medical RAG (type 'exit' to quit)\n")
    print("Available languages:", list(LANGUAGE_NAMES.keys())[:10], "...")
    
    while True:
        q = input("\nQuery: ")
        if q.lower().strip() == "exit":
            break
            
        source_lang = input("Source language (e.g., en, hi, ta, ml, es): ").strip() or "en"
        target_lang = input("Target language (e.g., en, hi, ta, ml, es): ").strip() or source_lang
        
        res = rag.query(q, domain="outbreak", source_lang=source_lang, target_lang=target_lang)
        print("\n💬 Answer:", res["result"])
        
        # Count sentences in response
        sentences = [s.strip() for s in res["result"].split('.') if s.strip()]
        print(f"📝 Sentences: {len(sentences)}")
        
        if res.get("source_documents"):
            print("\n📚 Sources:")
            for i, doc in enumerate(res["source_documents"]):
                source_meta = getattr(doc, "metadata", {})
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"  {i+1}. {source_meta.get('source', 'Unknown')}: {content_preview}")