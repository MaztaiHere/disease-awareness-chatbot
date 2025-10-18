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
            
            # Optimized for concise responses suitable for translation
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                n_ctx=2048,  # Reduced context for faster processing
                n_threads=8,
                n_batch=512,
                n_gpu_layers=1,
                temperature=0.1,  # Lower temperature for more factual responses
                top_p=0.9,
                repeat_penalty=1.1,
                max_tokens=350,  # Reduced for concise responses suitable for translation
                verbose=False,
                use_mlock=True,
                use_mmap=True,
            )
            logging.info("🦙 Local LLM initialized with concise response settings")
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
        medical_context = ""
        
        # Medical context enhancement
        medical_terms = {
            "fever": ["காய்ச்சல்", "ज्वर", "বুখার", "حمى"],
            "cough": ["இருமல்", "खांसी", "কাশি", "سعال"],
            "headache": ["தலைவலி", "सिरदर्द", "মাথাব্যথা", "صداع الرأس"],
            "outbreak": ["தொற்று", "प्रकोप", "প্রাদুর্ভাব", "تفشي"],
            "symptom": ["அறிகுறி", "लक्षण", "লক্ষণ", "عرض"]
        }
        
        for term, translations in medical_terms.items():
            if any(translation in query for translation in translations):
                medical_context = f" [medical context: {term}]"
                break
                
        return query + medical_context

    def translate_text(self, text: str, target_lang: str, source_lang: str = "en", step_name: str = "Translation") -> str:
        """Improved translation with better quality settings and detailed logging."""
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
            
            source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
            target_lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
            
            logging.info(f"🌍 {step_name}: {source_lang_name} → {target_lang_name}")
            logging.info(f"   📥 Input: {text}")
            if enhanced_text != text:
                logging.info(f"   🔧 Enhanced: {enhanced_text}")
            
            self.mbart_tokenizer.src_lang = src_code
            encoded = self.mbart_tokenizer(enhanced_text, return_tensors="pt", truncation=True, max_length=512)
            
            generated = self.mbart_model.generate(
                **encoded,
                forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id[tgt_code],
                max_length=384,  # Reduced for shorter responses
                num_beams=4,  # Slightly reduced for speed
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=0.8  # Slightly shorter outputs
            )
            
            translated = self.mbart_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            logging.info(f"   📤 Output: {translated}")
            logging.info(f"   ✅ {step_name} completed")
            return translated
            
        except Exception as e:
            logging.error(f"❌ {step_name} error ({source_lang}→{target_lang}): {e}")
            return text

    def _initialize_prompts(self):
        """Concise prompts for shorter responses suitable for translation."""
        prompts = {
            "symptom": """
Based on the medical context, provide a clear and concise answer in 2-3 sentences maximum.
Focus on the most relevant information. Be precise and avoid unnecessary details.

Context: {context}
Question: {question}

Concise Medical Answer:
""",
            
            "outbreak": """
Based on the outbreak data, provide a clear and concise summary in 2-3 sentences.
Include only the most relevant details about locations, diseases, and key facts.

Context: {context}
Question: {question}

Concise Outbreak Summary:
""",
            
            "misinformation": """
Based on the factual information, provide a clear and concise response in 2-3 sentences.
Address the question directly with evidence-based facts. Be straightforward.

Context: {context}
Question: {question}

Factual Response:
"""
        }
        return prompts

    def _retrieve_documents(self, query: str, domain: str, k: int = 6) -> List[Document]:
        """Retrieve documents with detailed logging."""
        if domain not in self.vector_stores:
            logging.error(f"❌ Domain '{domain}' not found in vector stores")
            return []
            
        try:
            logging.info(f"🔍 Retrieving documents for domain: {domain}")
            logging.info(f"   📝 Query: {query}")
            
            retriever = self.vector_stores[domain].as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            documents = retriever.invoke(query)
            logging.info(f"   📚 Retrieved {len(documents)} documents")
            
            # Log document snippets
            for i, doc in enumerate(documents[:2]):  # Show first 2 documents
                snippet = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
                logging.info(f"   📄 Doc {i+1}: {snippet}")
            
            if len(documents) > 2:
                logging.info(f"   ... and {len(documents) - 2} more documents")
                
            return documents
            
        except Exception as e:
            logging.error(f"❌ Document retrieval error: {e}")
            return []

    def _clean_context(self, context: str) -> str:
        """Clean context while preserving key information."""
        import re
        # Remove report IDs but keep the actual content
        cleaned = re.sub(r'(Outbreak Report ID|Report ID|ID)\s*\d+', '', context)
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _extract_meaningful_content(self, documents: List[Document]) -> str:
        """Extract and combine the most meaningful parts of documents with scoring."""
        scored_documents = []
        
        for doc in documents:
            content = self._clean_context(doc.page_content)
            score = 0
            
            # Score based on relevance indicators
            if any(keyword in content.lower() for keyword in ['confirmed', 'reported', 'cases', 'outbreak']):
                score += 2
            if any(keyword in content.lower() for keyword in ['symptoms', 'treatment', 'prevention', 'diagnosis']):
                score += 2
            if len(content) > 100:  # Prefer substantial content
                score += 1
            if 'metadata' in doc.metadata and doc.metadata.get('source'):
                score += 1
                
            scored_documents.append((score, content))
        
        # Sort by score and take top documents
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        meaningful_parts = [content for score, content in scored_documents if score > 0]
        
        # Combine and limit to reasonable length (reduced for shorter responses)
        combined = "\n\n".join(meaningful_parts[:4])  # Use up to 4 meaningful documents
        return combined[:1800]  # Reduced context limit for faster processing

    def _generate_response(self, query: str, documents: List[Document], domain: str) -> str:
        """Generate concise, accurate responses suitable for translation."""
        if not self.llm:
            logging.warning("⚠️ LLM not available, using fallback response")
            return self._get_fallback_response(domain)
            
        try:
            # Use meaningful content extraction
            context = self._extract_meaningful_content(documents)
            
            logging.info(f"🧠 Generating response with {len(context)} characters of context")
            
            if not context.strip():
                logging.warning("⚠️ No meaningful context extracted, using fallback")
                return self._get_fallback_response(domain)
            
            prompt_template = self.prompts.get(domain, self.prompts["symptom"])
            formatted_prompt = prompt_template.format(context=context, question=query)
            
            logging.info(f"   📋 Using {domain} prompt template")
            logging.info(f"   💭 Generating LLM response...")
            
            response = self.llm.invoke(formatted_prompt)
            cleaned_response = self._clean_response(response.strip())
            
            logging.info(f"   📝 Raw LLM response: {response[:150]}...")
            logging.info(f"   ✅ Final cleaned response: {cleaned_response}")
            
            return cleaned_response
            
        except Exception as e:
            logging.error(f"❌ Response generation error: {e}")
            return self._get_fallback_response(domain)

    def _clean_response(self, text: str) -> str:
        """Clean response to ensure 2-3 concise sentences for translation."""
        import re
        
        # Remove prompt artifacts and extra whitespace
        text = re.sub(r'^(Concise Medical Answer|Concise Outbreak Summary|Factual Response):\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentences = [s for s in sentences if 10 < len(s) < 100]  # Filter for good sentence length
        
        if not sentences:
            return self._get_fallback_response("general")
        
        # Take 2-3 concise sentences (reduced from 3-5)
        selected_sentences = sentences[:3]
        result = '. '.join(selected_sentences)
        if not result.endswith('.'):
            result += '.'
        
        # Ensure response isn't too long for translation
        if len(result) > 250:
            result = result[:247] + '...'
            
        return result

    def _get_fallback_response(self, domain: str) -> str:
        """Concise fallback responses suitable for translation."""
        fallbacks = {
            "outbreak": "Specific outbreak information is currently limited. Check official health sources like WHO for current updates.",
            "symptom": "Consult healthcare professionals for accurate symptom assessment and treatment advice.",
            "misinformation": "Verify health information with reliable medical sources and healthcare providers.",
            "general": "Consult healthcare professionals for reliable medical information."
        }
        return fallbacks.get(domain, fallbacks["general"])

    def query(self, user_query: str, domain: str, source_lang: str = "en", target_lang: str = None):
        """
        Enhanced RAG pipeline with concise responses suitable for translation.
        """
        logging.info("=" * 70)
        logging.info("🩺 MEDICAL RAG QUERY PROCESSING")
        logging.info("=" * 70)
        
        if not target_lang:
            target_lang = source_lang

        try:
            # Step 1: Log initial query details
            source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
            target_lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
            
            logging.info(f"📥 INPUT QUERY:")
            logging.info(f"   💬 Query: {user_query}")
            logging.info(f"   🎯 Domain: {domain}")
            logging.info(f"   🌐 Languages: {source_lang_name} → {target_lang_name}")

            # Step 2: Translation to English (if needed)
            if source_lang != "en":
                english_query = self.translate_text(user_query, "en", source_lang, "Query Translation")
                logging.info(f"🔗 TRANSLATED QUERY: {english_query}")
            else:
                english_query = user_query
                logging.info(f"🔗 USING ORIGINAL QUERY (English)")

            # Step 3: Document retrieval
            documents = self._retrieve_documents(english_query, domain, k=6)  # Reduced k for faster processing
            total_context_chars = sum(len(doc.page_content) for doc in documents)
            logging.info(f"📊 RETRIEVAL SUMMARY: {len(documents)} documents, {total_context_chars} total characters")

            # Step 4: Response generation
            logging.info("🤖 GENERATING RESPONSE...")
            raw_answer = self._generate_response(english_query, documents, domain)
            logging.info(f"📝 GENERATED ENGLISH RESPONSE: {raw_answer}")

            # Step 5: Back translation (if needed)
            if target_lang != "en":
                final_answer = self.translate_text(raw_answer, target_lang, "en", "Response Translation")
                logging.info(f"🌐 FINAL TRANSLATED RESPONSE: {final_answer}")
            else:
                final_answer = raw_answer
                logging.info(f"🌐 USING ORIGINAL RESPONSE (English)")

            # Prepare response
            response = {
                "result": final_answer,
                "source_documents": documents,
                "processing_details": {
                    "original_query": user_query,
                    "translated_query": english_query if source_lang != "en" else user_query,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "domain": domain,
                    "documents_retrieved": len(documents)
                }
            }
            
            logging.info("✅ QUERY PROCESSING COMPLETED")
            logging.info("=" * 70)
            return response

        except Exception as e:
            logging.error(f"❌ QUERY PROCESSING ERROR: {e}")
            logging.info("=" * 70)
            return {
                "result": self._get_fallback_response(domain), 
                "source_documents": [],
                "processing_details": {"error": str(e)}
            }


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":
    rag = MedicalRAG()
    
    print("\n" + "="*60)
    print("🩺 MEDICAL RAG ASSISTANT - CONCISE VERSION")
    print("="*60)
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            q = input("\n💬 Query: ").strip()
            if q.lower() == "exit":
                break
            if not q:
                continue
                
            source_lang = input("🌐 Source language (e.g., en, hi, ta, ml, es): ").strip() or "en"
            target_lang = input("🎯 Target language: ").strip() or source_lang
            domain = input("📊 Domain (outbreak/symptom/misinformation): ").strip() or "symptom"
            
            if domain not in ["outbreak", "symptom", "misinformation"]:
                print("❌ Invalid domain. Using 'symptom' as default.")
                domain = "symptom"
            
            print("\n" + "="*50)
            res = rag.query(q, domain=domain, source_lang=source_lang, target_lang=target_lang)
            print("\n💡 ANSWER:")
            print(res["result"])
            print(f"\n📚 Sources: {len(res['source_documents'])} documents retrieved")
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue