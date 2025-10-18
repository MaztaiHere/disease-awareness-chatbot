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
            
            # Balanced for quality and speed - slightly more tokens for better responses
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_threads=8,
                n_batch=512,
                n_gpu_layers=1,
                temperature=0.3,
                top_p=0.85,
                repeat_penalty=1.1,
                max_tokens=512,  # Increased for better quality responses
                verbose=False,
                use_mlock=True,
                use_mmap=True,
            )
            logging.info("🦙 Local LLM initialized with quality focus")
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

    def _enhance_translation_query(self, query: str, source_lang: str) -> str:
        """Add context to improve translation quality for medical terms."""
        if source_lang == "ta":  # Tamil
            # Add context for common Tamil medical terms
            enhanced_query = query
            if "உணவு விஷம்" in query:  # Food poisoning
                enhanced_query = query + " [food poisoning outbreak cases reported]"
            elif "காய்ச்சல்" in query:  # Fever
                enhanced_query = query + " [fever disease outbreak]"
            elif "இருமல்" in query:  # Cough
                enhanced_query = query + " [cough respiratory disease]"
            return enhanced_query
        return query

    def translate_text(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """Improved translation with better quality settings."""
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
            
            # Enhance query for better translation of medical terms
            enhanced_text = self._enhance_translation_query(text, source_lang)
            
            logging.info(f"🌍 Translating from {source_lang} to {target_lang}...")
            
            self.mbart_tokenizer.src_lang = src_code
            # Better quality translation with more tokens
            encoded = self.mbart_tokenizer(enhanced_text, return_tensors="pt", truncation=True, max_length=512)
            
            generated = self.mbart_model.generate(
                **encoded,
                forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id[tgt_code],
                max_length=512,      # Increased for better quality
                num_beams=4,         # More beams for better translation
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            translated = self.mbart_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            logging.info(f"✅ Translation completed: {source_lang} → {target_lang}")
            return translated
            
        except Exception as e:
            logging.error(f"Translation error ({source_lang}→{target_lang}): {e}")
            return text

    def _initialize_prompts(self):
        """Improved prompts for detailed, natural responses."""
        prompts = {
            "symptom": """
Based on the medical context below, provide a comprehensive answer in 4-5 complete sentences.
Focus on providing helpful, detailed information about symptoms and care.
Write in natural, flowing language - avoid bullet points or technical formatting.

Context: {context}
Question: {question}

Detailed Answer:
""",
            
            "outbreak": """
Based on the outbreak reports below, provide a comprehensive summary in 4-5 complete sentences.
Include specific details about diseases, locations, timeframes, and impacts when available.
Write in natural, flowing paragraphs - avoid technical formatting or bullet points.
If specific outbreak data is available, summarize the key findings clearly.

Context: {context}
Question: {question}

Comprehensive Outbreak Summary:
""",
            
            "misinformation": """
Based on the factual information below, provide a thorough response in 4-5 complete sentences.
Address the question with clear, evidence-based information.
Write in natural, flowing language that is easy to understand.

Context: {context}
Question: {question}

Factual Response:
"""
        }
        return prompts

    def _retrieve_documents(self, query: str, domain: str, k: int = 8) -> List[Document]:
        """Retrieve more documents for better context."""
        if domain not in self.vector_stores:
            return []
            
        try:
            retriever = self.vector_stores[domain].as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            documents = retriever.invoke(query)
            logging.info(f"📚 Retrieved {len(documents)} documents for {domain} domain")
            
            # Log document content for debugging
            if documents:
                for i, doc in enumerate(documents[:3]):
                    content_preview = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
                    logging.info(f"   Doc {i+1}: {content_preview}")
            
            return documents
            
        except Exception as e:
            logging.error(f"❌ Document retrieval error: {e}")
            return []

    def _clean_context(self, context: str) -> str:
        """Better context cleaning that preserves meaningful information."""
        import re
        # Remove report IDs but keep the actual content
        cleaned = re.sub(r'(Outbreak Report ID|Report ID|ID)\s*\d+', '', context)
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _extract_meaningful_content(self, documents: List[Document]) -> str:
        """Extract and combine the most meaningful parts of documents."""
        meaningful_parts = []
        
        for doc in documents:
            content = self._clean_context(doc.page_content)
            # Look for patterns that indicate useful information
            if any(keyword in content.lower() for keyword in [
                'cases', 'reported', 'outbreak', 'disease', 'illness', 
                'confirmed', 'symptoms', 'treatment', 'prevention'
            ]):
                meaningful_parts.append(content)
        
        # Combine and limit to reasonable length
        combined = "\n".join(meaningful_parts[:6])  # Use up to 6 meaningful documents
        return combined[:2500]  # Limit total context

    def _generate_response(self, query: str, documents: List[Document], domain: str) -> str:
        """Generate high-quality, detailed responses."""
        if not self.llm:
            return self._get_fallback_response(domain)
            
        try:
            # Use meaningful content extraction instead of raw concatenation
            context = self._extract_meaningful_content(documents)
            
            if not context.strip():
                logging.info("🔍 No meaningful context found, using fallback response")
                return self._get_fallback_response(domain)
            
            prompt_template = self.prompts.get(domain, self.prompts["symptom"])
            formatted_prompt = prompt_template.format(context=context, question=query)
            
            logging.info("🔹 Generating detailed response...")
            response = self.llm.invoke(formatted_prompt)
            
            cleaned_response = self._clean_response(response.strip())
            logging.info(f"✅ Generated {len(cleaned_response.split())} words response")
            
            return cleaned_response
            
        except Exception as e:
            logging.error(f"❌ Response generation error: {e}")
            return self._get_fallback_response(domain)

    def _clean_response(self, text: str) -> str:
        """Clean response and ensure 4-5 quality sentences."""
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        # Filter out very short or poorly formed sentences
        sentences = [s for s in sentences if len(s) > 20 and not s.startswith('-') and not s.startswith('•')]
        
        if not sentences:
            return self._get_fallback_response("general")
        
        # Take 4-5 quality sentences
        selected_sentences = sentences[:5]
        result = '. '.join(selected_sentences)
        if not result.endswith('.'):
            result += '.'
        return result

    def _get_fallback_response(self, domain: str) -> str:
        """High-quality fallback responses."""
        fallbacks = {
            "outbreak": "Based on available outbreak data, specific information about this query is not currently available. It's important to monitor official health sources like the World Health Organization and local health departments for the most current outbreak information. Practice good hygiene measures including regular hand washing and food safety practices. If you suspect an outbreak in your area, contact local health authorities for guidance and follow their recommendations.",
            "symptom": "For accurate information about symptoms and appropriate care, it's essential to consult with healthcare professionals. They can provide personalized advice based on your specific situation and medical history. Keep track of any symptoms you're experiencing, including their duration and severity. Share this information with your healthcare provider for proper assessment and guidance.",
            "misinformation": "When evaluating health information, it's crucial to rely on verified medical sources and healthcare professionals. Look for information from reputable organizations like government health agencies and established medical institutions. Be cautious of claims that lack scientific evidence or come from unverified sources. Always consult qualified healthcare providers for medical advice tailored to your specific needs.",
            "general": "For reliable health information, consult qualified healthcare professionals and official medical sources. They can provide evidence-based guidance appropriate for your specific situation. Stay informed through reputable health organizations and verify information from multiple trusted sources."
        }
        return fallbacks.get(domain, fallbacks["general"])

    def query(self, user_query: str, domain: str, source_lang: str = "en", target_lang: str = None):
        """
        Quality-focused RAG pipeline with better translation and detailed responses.
        """
        logging.info("=" * 70)
        logging.info(f"🧠 Query received: {user_query}")
        logging.info(f"🌍 Domain: {domain}, Source Language: {source_lang}")
        
        if not target_lang:
            target_lang = source_lang
            
        logging.info(f"🎯 Target Language: {target_lang}")

        try:
            # Step 1: Enhanced translation with medical context
            logging.info("🔹 Translating query with medical context enhancement...")
            if source_lang != "en":
                english_query = self.translate_text(user_query, "en", source_lang)
                logging.info(f"→ Enhanced English Query: {english_query}")
            else:
                english_query = user_query
                logging.info("→ Query is already in English")

            # Step 2: Retrieve more documents for better context
            logging.info("🔹 Retrieving documents for comprehensive context...")
            documents = self._retrieve_documents(english_query, domain, k=8)
            
            # Step 3: Generate detailed, high-quality response
            logging.info("🔹 Generating comprehensive response...")
            raw_answer = self._generate_response(english_query, documents, domain)
            logging.info(f"→ Raw Answer: {raw_answer}")

            # Step 4: Quality translation back to target language
            if target_lang != "en":
                logging.info(f"🔹 Translating answer to {target_lang}...")
                final_answer = self.translate_text(raw_answer, target_lang, "en")
                logging.info(f"✅ Final Answer ({target_lang}): {final_answer}")
            else:
                final_answer = raw_answer
                logging.info(f"✅ Final Answer (English): {final_answer}")

            response = {
                "result": final_answer,
                "source_documents": documents
            }
            
            logging.info("=" * 70 + "\n")
            return response

        except Exception as e:
            logging.error(f"❌ Error in RAG pipeline: {e}")
            return {"result": self._get_fallback_response(domain), "source_documents": []}


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":
    rag = MedicalRAG()
    
    print("\n🩺 Medical RAG Assistant (Quality Optimized) - type 'exit' to quit\n")
    print("Available languages:", list(LANGUAGE_NAMES.keys())[:10], "...")
    
    while True:
        q = input("\nQuery: ")
        if q.lower().strip() == "exit":
            break
            
        source_lang = input("Source language (e.g., en, hi, ta, ml, es): ").strip() or "en"
        target_lang = input("Target language: ").strip() or source_lang
        
        res = rag.query(q, domain="outbreak", source_lang=source_lang, target_lang=target_lang)
        print("\n💬 Answer:", res["result"])
        
        sentences = [s.strip() for s in res["result"].split('.') if s.strip()]
        print(f"📝 Sentences: {len(sentences)}")
        
        if res.get("source_documents"):
            print(f"📚 Sources: {len(res['source_documents'])} documents retrieved")