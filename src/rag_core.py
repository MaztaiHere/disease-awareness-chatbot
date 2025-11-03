import torch
torch.set_num_threads(1)
torch.classes.__path__ = []
import re
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

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Add these additional warning filters at the top with other imports
import warnings
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
# CLASS DEFINITION
# ======================================================

class MedicalRAG:
    def __init__(self):
        logging.info("🚀 Initializing MedicalRAG system...")
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._ensure_model_downloaded()
        self.config_path = BASE_DIR / "language_config.json" # Adjust path if needed
        self.nllb_lang_codes, self.language_names = self._load_language_config()
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store_client = chromadb.PersistentClient(
            path=PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        self.llm = self._initialize_llm()
        self.nllb_model = None
        self.nllb_tokenizer = None
        self._initialize_nllb()
        self.vector_stores = self._initialize_vector_stores()
        self._build_vector_stores_if_empty()
        self.prompts = self._initialize_prompts()
        logging.info("✅ MedicalRAG system initialized successfully.\n")
    def _load_language_config(self):
        """Loads language config from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logging.info(f"✅ Language config loaded from {self.config_path}")
            return config.get("NLLB_LANG_CODES", {}), config.get("LANGUAGE_NAMES", {})
        except FileNotFoundError:
            logging.error(f"❌ Language config file not found: {self.config_path}")
            return {}, {}
        except json.JSONDecodeError:
            logging.error(f"❌ Failed to decode language config JSON: {self.config_path}")
            return {}, {}

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
            
            # Balanced settings for better quality while maintaining speed
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                n_ctx=2048,  # Increased for better context understanding
                n_threads=8,
                n_batch=256,
                n_gpu_layers=1,
                temperature=0.3,  # Slightly higher for better creativity
                top_p=0.9,
                repeat_penalty=1.1,
                max_tokens=256,  # Increased for more detailed but still concise responses
                verbose=False,
                use_mlock=False,
                use_mmap=True,
            )
            logging.info("🦙 Mistral 7B initialized with balanced settings")
            return llm
        except Exception as e:
            logging.error(f"❌ Failed to initialize LLM: {e}")
            return None

    def _initialize_nllb(self):
        """Initialize NLLB-200 model and tokenizer with optimizations"""
        if self.nllb_model and self.nllb_tokenizer:
            return
        try:
            logging.info("🌐 Loading NLLB-200 translation model...")
            model_name = "facebook/nllb-200-1.3B"
            
            # Load tokenizer
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with optimizations for faster inference
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                dtype=torch.float16,  # Use float16 for faster inference
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Disable gradient calculation for inference
            self.nllb_model.eval()
            
            logging.info("✅ NLLB-200 model loaded successfully.")
            
        except Exception as e:
            logging.error(f"❌ Failed to load NLLB-200 model: {e}")
            self.nllb_model, self.nllb_tokenizer = None, None

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
        """Optimized translation with faster inference settings."""
        if not text.strip() or target_lang == source_lang:
            return text

        if not self.nllb_model or not self.nllb_tokenizer:
            return text
            
        try:
            if source_lang not in self.nllb_lang_codes or target_lang not in self.nllb_lang_codes:
                logging.warning(f"⚠️ Unsupported language pair: {source_lang} -> {target_lang}")
                return text
                
            src_code = self.nllb_lang_codes[source_lang]
            tgt_code = self.nllb_lang_codes[target_lang]
            # --- END MODIFY ---
            
            logging.info(f"🌍 Translating from {source_lang} to {target_lang}")
            
            # Tokenize with source language - optimized settings
            inputs = self.nllb_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256  # Increased for better quality
            )
            
            # Move inputs to the same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Faster generation with optimized settings
            with torch.no_grad():  # Disable gradient calculation
                generated_tokens = self.nllb_model.generate(
                    **inputs,
                    forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(tgt_code),
                    max_length=128,  # Increased for better quality
                    num_beams=2,     # Balanced for speed and quality
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    do_sample=False   # Disable sampling for deterministic output
                )
            
            # Decode the generated tokens
            translated = self.nllb_tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            
            logging.info(f"✅ Translation completed")
            return translated
            
        except Exception as e:
            logging.error(f"❌ NLLB translation error: {e}")
            return text

    def _initialize_prompts(self):
        """IMPROVED prompts for specific, actionable responses."""
        prompts = {
            "symptom": """<s>[INST] You are a medical assistant. Based on the medical context, provide specific, actionable advice.

MEDICAL CONTEXT:
{context}

PATIENT QUERY: {question}

Provide a response with:
1. Specific possible conditions based on symptoms
2. Immediate actions to take
3. General self-care recommendations

Use • bullet points (MAX 3 BULLET POINTS). 20 WORDS PER BULLET POINT .ONLY 80 WORDS [/INST]""",
            
            "outbreak": """<s>[INST] You are a public health official. Provide specific outbreak information.

OUTBREAK CONTEXT:
{context}

QUERY: {question}

Provide response with:
1. Specific affected locations and dates
2. Case numbers and statistics when available
3. Official health recommendations

Use • bullet points (MAX 3 BULLET POINTS). 20 WORDS PER BULLET POINT .ONLY 80 WORDS [/INST]""",
            
            "misinformation": """<s>[INST] You are a medical fact-checker. Provide clear, evidence-based verdict.

FACT-CHECKING CONTEXT:
{context}

CLAIM: {question}

Provide response with:
1. Clear TRUE/FALSE/MISLEADING verdict
2. Specific evidence from medical sources
3. Explanation of scientific basis

Use • bullet points (MAX 3 BULLET POINTS). 20 WORDS PER BULLET POINT .ONLY 80 WORDS [/INST]"""
        }
        return prompts

    def _retrieve_documents(self, query: str, domain: str) -> List[Document]:
        """Optimized document retrieval."""
        if domain not in self.vector_stores:
            return []
            
        try:
            retriever = self.vector_stores[domain].as_retriever(
                search_type="similarity",  # Simpler and faster than MMR
                search_kwargs={
                    "k": 6,  # Balanced for coverage and speed
                }
            )
            documents = retriever.invoke(query)
            return documents
        except Exception as e:
            logging.error(f"❌ Document retrieval error: {e}")
            return []
    
    def _generate_response(self, query: str, documents: List[Document], domain: str) -> str:
        """Generate specific, actionable response using LLM."""
        if not self.llm:
            return "Based on your symptoms:\n- Seek immediate medical attention\n- Stay hydrated and rest\n- Monitor for worsening symptoms\n- Contact healthcare professionals for proper diagnosis"
            
        try:
            # Combine document content with focus on medical details
            context_parts = []
            for i, doc in enumerate(documents):
                if doc.page_content.strip():
                    content = doc.page_content.strip()
                    # Take enough content for good context
                    if len(content) > 400:
                        content = content[:400] + "..."
                    context_parts.append(f"{content}")
            
            context = "\n\n".join(context_parts)
            
            prompt_template = self.prompts.get(domain, self.prompts["symptom"])
            formatted_prompt = prompt_template.format(context=context, question=query)
            
            response = self.llm.invoke(formatted_prompt)
            
            # Clean and structure the response
            cleaned_response = self._clean_and_structure_response(response.strip(), domain)
            return cleaned_response
            
        except Exception as e:
            logging.error(f"❌ Response generation error: {e}")
            return "Based on your symptoms:\n- Seek immediate medical attention\n- Stay hydrated and rest\n- Monitor for worsening symptoms\n- Contact healthcare professionals for proper diagnosis"
    def _fix_translated_formatting(self, text: str) -> str:
        """Fix formatting issues after NLLB translation."""
        import re
        
        # Fix common translation formatting issues
        text = re.sub(r'[\-\*]\\s*', '• ', text)  # Fix \- or \* issues
        text = re.sub(r'•\\s*', '• ', text)       # Fix •\ issues
        text = re.sub(r'\s+', ' ', text)          # Normalize spaces
        text = re.sub(r'•\s+', '\n• ', text)      # Ensure proper line breaks
        # Ensure each bullet point is on its own line
        lines = text.split('•')
        if len(lines) > 1:
            # First line is the introduction
            formatted_text = lines[0].strip()
            # Add bullet points properly
            for bullet in lines[1:]:
                if bullet.strip():
                    formatted_text += '\n• ' + bullet.strip()
            text = formatted_text
        return text
    def _clean_and_structure_response(self, text: str, domain: str) -> str:
        """Better formatting preservation for translation."""
        
        # Remove prompt artifacts
        text = re.sub(r'\[/INST\]|\[INST\]|</s>|<s>', '', text)
        
        # Preserve bullet points and fix formatting
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Convert any bullet format to proper • format
            if re.match(r'^[\-\*•·]\s+', line):
                line = re.sub(r'^[\-\*•·]\s+', '• ', line)
            elif re.match(r'^\d+\.\s*', line):
                line = re.sub(r'^\d+\.\s*', '• ', line)
                
            lines.append(line)
        
        text = '\n'.join(lines)
        
        # Ensure proper structure
        if "•" not in text and len(lines) > 1:
            # Reformat as bullet points if missing
            formatted_lines = [lines[0]]  # Keep first line as intro
            for line in lines[1:]:
                if line.strip() and not line.startswith('•'):
                    formatted_lines.append('• ' + line)
            text = '\n'.join(formatted_lines)
        
        # Add domain-specific closing if missing
        if domain == "symptom" and "consult" not in text.lower():
            text += "\n\nConsult healthcare professionals for proper diagnosis."
        elif domain == "misinformation" and "verify" not in text.lower():
            text += "\n\nVerify with trusted medical sources."
        
        return text.strip()
    def query(self, user_query: str, domain: str, source_lang: str = "en", target_lang: str = None):
        """
        Improved RAG pipeline with specific, actionable responses.
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
                # ADD THIS LINE: Fix formatting after translation
                final_response = self._fix_translated_formatting(final_response)
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
                "result": "Based on your symptoms:\n- Seek immediate medical attention\n- Stay hydrated and rest\n- Monitor for worsening symptoms\n- Contact healthcare professionals for proper diagnosis",
                "source_documents": []
            }


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":
    rag = MedicalRAG()
    