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
from typing import Optional, Dict, Any, List

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
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
        self._build_vector_stores_if_empty()
        self.prompts = self._initialize_prompts()
        logging.info("✅ MedicalRAG system initialized successfully.\n")

    def _ensure_model_downloaded(self):
        if os.path.exists(MODEL_PATH):
            logging.info("🧠 Local GGUF model already present.")
            return
        logging.info("⬇️ Downloading GGUF model...")
        try:
            with requests.get(MODEL_URL, stream=True, timeout=600) as r:
                r.raise_for_status()
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
            
            # Better balanced settings for quality responses
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                n_ctx=4096,  # Increased context for better understanding
                n_threads=8,
                n_batch=512,
                n_gpu_layers=1,
                temperature=0.3,  # Slightly higher for better creativity
                top_p=0.9,
                repeat_penalty=1.1,
                max_tokens=512,  # Increased for medium-length responses
                verbose=False,
                use_mlock=False,
                use_mmap=True,
            )
            logging.info("🦙 Mistral 7B initialized with balanced settings")
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

    def _build_vector_stores_if_empty(self):
        """Build vector stores from processed data if they are empty."""
        logging.info("🔍 Checking if vector stores need to be built...")
        
        for domain in ["outbreak", "symptom", "misinformation"]:
            try:
                collection = self.vector_store_client.get_collection(f"medical_rag_{domain}")
                count = collection.count()
                logging.info(f"✅ Vector store for '{domain}' has {count} documents.")
                
                if count == 0:
                    logging.warning(f"⚠️ Vector store for '{domain}' is empty. Building from processed data...")
                    self._build_vector_store(domain)
                    
            except Exception as e:
                logging.warning(f"⚠️ Vector store for '{domain}' doesn't exist or is empty. Building...")
                self._build_vector_store(domain)

    def _build_vector_store(self, domain: str):
        """Build vector store from processed JSON data."""
        processed_file = os.path.join(PROCESSED_DATA_DIR, f"{domain}_chunks.json")
        
        if not os.path.exists(processed_file):
            logging.error(f"❌ Processed data file not found: {processed_file}")
            logging.error("Please run data_processing.py first to create processed data.")
            return

        try:
            logging.info(f"📖 Loading processed data from: {processed_file}")
            with open(processed_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            if not chunks:
                logging.warning(f"⚠️ No chunks found in {processed_file}")
                return

            documents = []
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk["page_content"],
                    metadata=chunk["metadata"]
                ))

            logging.info(f"📚 Adding {len(documents)} documents to {domain} vector store...")
            
            vector_store = Chroma(
                collection_name=f"medical_rag_{domain}",
                embedding_function=self.embedding_function,
                client=self.vector_store_client,
                persist_directory=PERSIST_DIRECTORY
            )
            
            vector_store.add_documents(documents)
            logging.info(f"✅ Successfully built vector store for '{domain}' with {len(documents)} documents.")

        except Exception as e:
            logging.error(f"❌ Failed to build vector store for {domain}: {e}")

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
        """Translate text between languages."""
        if not text.strip() or target_lang == source_lang:
            return text
            
        self._initialize_mbart()
        if not self.mbart_model or not self.mbart_tokenizer:
            return text
            
        try:
            if source_lang not in MBART_LANG_CODES or target_lang not in MBART_LANG_CODES:
                return text
                
            src_code = MBART_LANG_CODES[source_lang]
            tgt_code = MBART_LANG_CODES[target_lang]
            
            logging.info(f"🌍 Translating from {source_lang} to {target_lang}")
            
            self.mbart_tokenizer.src_lang = src_code
            encoded = self.mbart_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            generated = self.mbart_model.generate(
                **encoded,
                forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id[tgt_code],
                max_length=600,
                num_beams=4,
                early_stopping=True
            )
            
            translated = self.mbart_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return translated
            
        except Exception as e:
            logging.error(f"❌ Translation error: {e}")
            return text

    def _initialize_prompts(self):
        """BETTER prompts for accurate, medium-length responses."""
        prompts = {
            "symptom": """<s>[INST] You are a medical assistant. Provide a clear, accurate 3-4 sentence response.

CONTEXT:
{context}

QUESTION: {question}

GUIDELINES:
- Provide specific medical information based on the context
- Include key symptoms, causes, and when to seek medical attention
- Be precise but comprehensive (3-4 sentences)
- Only use information from the provided context
- If context is insufficient, state "Consult a healthcare professional for proper diagnosis"

RESPONSE: [/INST]""",
            
            "outbreak": """<s>[INST] You are a public health official. Provide detailed outbreak information.

CONTEXT:
{context}

QUESTION: {question}

GUIDELINES:
- List specific outbreaks with locations, dates, and case numbers when available
- Include affected regions and key statistics
- Mention public health recommendations
- Provide 4-5 sentences with concrete information
- If no specific outbreaks match, say "No matching outbreak information found in current data"

RESPONSE: [/INST]""",
            
            "misinformation": """<s>[INST] You are a medical fact-checker. Provide a thorough fact-check.

CONTEXT:
{context}

QUESTION: {question}

GUIDELINES:
- Clearly state if the claim is TRUE, FALSE, or MISLEADING
- Provide specific evidence from the context
- Explain the scientific basis for the verdict
- Include 3-4 sentences with detailed explanation
- If evidence is insufficient, state "Insufficient evidence to verify this claim"

RESPONSE: [/INST]"""
        }
        return prompts

    def _retrieve_documents(self, query: str, domain: str) -> List[Document]:
        """Retrieve relevant documents with better coverage."""
        if domain not in self.vector_stores:
            return []
            
        try:
            retriever = self.vector_stores[domain].as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}  # More documents for better context
            )
            documents = retriever.invoke(query)
            return documents
        except Exception as e:
            logging.error(f"❌ Document retrieval error: {e}")
            return []

    def _generate_response(self, query: str, documents: List[Document], domain: str) -> str:
        """Generate accurate, medium-length response using LLM."""
        if not self.llm:
            return "Consult a healthcare professional for accurate medical advice."
            
        try:
            # Combine document content with better context
            context_parts = []
            for i, doc in enumerate(documents):
                if doc.page_content.strip():
                    # Take more content for better answers
                    content = doc.page_content.strip()
                    if len(content) > 600:
                        content = content[:600] + "..."
                    context_parts.append(f"Document {i+1}: {content}")
            
            context = "\n\n".join(context_parts)
            
            prompt_template = self.prompts.get(domain, self.prompts["symptom"])
            formatted_prompt = prompt_template.format(context=context, question=query)
            
            response = self.llm.invoke(formatted_prompt)
            
            # Better response cleaning and validation
            cleaned_response = self._clean_and_validate_response(response.strip(), domain)
            return cleaned_response
            
        except Exception as e:
            logging.error(f"❌ Response generation error: {e}")
            return "Consult a healthcare professional for accurate medical advice."

    def _clean_and_validate_response(self, text: str, domain: str) -> str:
        """Clean response while maintaining quality and appropriate length."""
        import re
        
        # Remove prompt artifacts
        text = re.sub(r'\[/INST\]|\[INST\]|</s>|<s>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common hallucinations but be less restrictive
        invalid_patterns = [
            r'insert your name', r'my name is', r'as an ai', r'as a language model',
            r'this is a test', r'hello there', r'hey there', r'good morning', r'good afternoon'
        ]
        
        text_lower = text.lower()
        for pattern in invalid_patterns:
            if re.search(pattern, text_lower):
                if domain == "misinformation":
                    return "Insufficient evidence to verify this claim."
                elif domain == "outbreak":
                    return "No matching outbreak information found in current data."
                else:
                    return "Consult a healthcare professional for proper diagnosis."
        
        # Domain-specific validation
        if domain == "outbreak":
            # Ensure outbreak responses have specific information
            if not any(word in text_lower for word in ['outbreak', 'cases', 'reported', 'confirmed', 'alert', 'incident']):
                return "No matching outbreak information found in current data."
        
        elif domain == "misinformation":
            # Ensure misinformation responses have verdict
            if not any(word in text_lower for word in ['true', 'false', 'misleading', 'verdict', 'evidence']):
                return "Insufficient evidence to verify this claim."
        
        # Ensure reasonable length (3-6 sentences, 300-600 chars)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 6:
            text = '. '.join(sentences[:6]) + '.'
        elif len(sentences) < 2:
            # If response is too short, provide appropriate fallback
            if domain == "misinformation":
                text = "Insufficient evidence to verify this claim. Consult reliable medical sources."
            elif domain == "outbreak":
                text = "No specific outbreak information available for this query in current data."
            else:
                text = "Consult a healthcare professional for proper medical advice."
        
        # Reasonable character limits for translation
        if len(text) > 800:
            text = text[:797] + '...'
            
        return text

    def query(self, user_query: str, domain: str, source_lang: str = "en", target_lang: str = None):
        """
        Improved RAG pipeline with better response quality.
        """
        logging.info("=" * 60)
        logging.info(f"🩺 QUERY: {domain} | {source_lang}")
        logging.info("=" * 60)
        
        if not target_lang:
            target_lang = source_lang

        try:
            # Step 1: Log initial query
            logging.info(f"📥 INPUT: {user_query}")

            # Step 2: Translate query to English if needed
            if source_lang != "en":
                english_query = self.translate_text(user_query, "en", source_lang)
                logging.info(f"🔗 TRANSLATED: {english_query}")
            else:
                english_query = user_query

            # Step 3: Document retrieval
            documents = self._retrieve_documents(english_query, domain)
            logging.info(f"📚 Retrieved {len(documents)} documents")

            # Step 4: Generate English response
            english_response = self._generate_response(english_query, documents, domain)
            logging.info(f"📝 RESPONSE: {english_response}")

            # Step 5: Translate response back if needed
            if target_lang != "en":
                final_response = self.translate_text(english_response, target_lang, "en")
                logging.info(f"🌐 TRANSLATED: {final_response}")
            else:
                final_response = english_response

            response = {
                "result": final_response,
                "source_documents": documents
            }
            
            logging.info("✅ COMPLETED")
            logging.info("=" * 60)
            return response

        except Exception as e:
            logging.error(f"❌ ERROR: {e}")
            return {
                "result": "Consult a healthcare professional for accurate medical advice.",
                "source_documents": []
            }


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":
    rag = MedicalRAG()
    
    print("\n" + "="*50)
    print("🩺 MEDICAL RAG - MISTRAL 7B OPTIMIZED")
    print("="*50)
    print("Type 'exit' to quit\n")
    
    # Test queries to demonstrate improved responses
    test_queries = [
        ("What outbreaks are there in Kerala in 2025?", "outbreak", "en", "en"),
        ("I read that drinking hot water prevents COVID, is this true?", "misinformation", "en", "en"),
        ("What are the symptoms of dengue fever?", "symptom", "en", "en")
    ]
    
    for query, domain, src_lang, tgt_lang in test_queries:
        print(f"\n🧪 TEST: {query}")
        res = rag.query(query, domain=domain, source_lang=src_lang, target_lang=tgt_lang)
        print(f"💡 ANSWER: {res['result']}")
        print(f"📚 Sources: {len(res['source_documents'])} documents")
        print("-" * 50)
    
    while True:
        try:
            q = input("\n💬 Your Query: ").strip()
            if q.lower() == "exit":
                break
            if not q:
                continue
                
            source_lang = input("🌐 Source language (e.g., en, hi, ta, ml, es): ").strip() or "en"
            target_lang = input("🎯 Target language: ").strip() or source_lang
            domain = input("📊 Domain (outbreak/symptom/misinformation): ").strip() or "symptom"
            
            if domain not in ["outbreak", "symptom", "misinformation"]:
                domain = "symptom"
            
            res = rag.query(q, domain=domain, source_lang=source_lang, target_lang=target_lang)
            print("\n💡 ANSWER:")
            print(res["result"])
            print(f"\n📚 Sources: {len(res['source_documents'])} documents")
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue