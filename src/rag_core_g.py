# src/rag_core.py
import os
import json
import logging

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
PROCESSED_DATA_DIR = "./data/processed"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
PERSIST_DIRECTORY = "./vector_db"
LLM_MODEL_ID = "google/flan-t5-large"


class MedicalRAG:
    """Retrieval-Augmented Generation system for medical domains."""

    def __init__(self):
        logging.info("Initializing MedicalRAG system...")

        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

        self.llm = self._initialize_llm()
        self._initialize_translator()
        self._precache_translation_models()
        self.chains = self._initialize_chains()

        logging.info("MedicalRAG system initialized.")

    def _initialize_llm(self):
        """Initializes the LLM with HuggingFace pipeline."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
            model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_ID)
            pipe = pipeline(
                "text2text-generation", model=model, tokenizer=tokenizer, max_length=256
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logging.error(f"Failed to initialize LLM. Error: {e}")
            return None

    def _initialize_translator(self):
        """Initializes translation model mappings and cache."""
        logging.info("Initializing Hugging Face translation engine...")

        self.translator_cache, self.reverse_translator_cache = {}, {}

        self.lang_to_model_map = {
            "ta": "Helsinki-NLP/opus-mt-en-dra",
            "ml": "Helsinki-NLP/opus-mt-en-dra",
            "kn": "Helsinki-NLP/opus-mt-en-dra",
            "te": "Helsinki-NLP/opus-mt-en-dra",
            "mr": "Helsinki-NLP/opus-mt-en-inc",
            "hi": "Helsinki-NLP/opus-mt-en-hi",
            "gu": "Helsinki-NLP/opus-mt-en-inc",
            "bn": "Helsinki-NLP/opus-mt-en-bn",
            "pa": "Helsinki-NLP/opus-mt-en-inc",
            "as": "Helsinki-NLP/opus-mt-en-inc",
            "es": "Helsinki-NLP/opus-mt-en-es",
            "de": "Helsinki-NLP/opus-mt-en-de",
            "it": "Helsinki-NLP/opus-mt-en-it",
            "fr": "Helsinki-NLP/opus-mt-en-fr",
            "ja": "Helsinki-NLP/opus-mt-en-jap",
        }

        self.reverse_lang_to_model_map = {
            "ta": "Helsinki-NLP/opus-mt-dra-en",
            "ml": "Helsinki-NLP/opus-mt-dra-en",
            "kn": "Helsinki-NLP/opus-mt-dra-en",
            "te": "Helsinki-NLP/opus-mt-dra-en",
            "mr": "Helsinki-NLP/opus-mt-inc-en",
            "hi": "Helsinki-NLP/opus-mt-hi-en",
            "gu": "Helsinki-NLP/opus-mt-inc-en",
            "bn": "Helsinki-NLP/opus-mt-bn-en",
            "pa": "Helsinki-NLP/opus-mt-inc-en",
            "as": "Helsinki-NLP/opus-mt-inc-en",
            "es": "Helsinki-NLP/opus-mt-es-en",
            "de": "Helsinki-NLP/opus-mt-de-en",
            "it": "Helsinki-NLP/opus-mt-it-en",
            "fr": "Helsinki-NLP/opus-mt-fr-en",
            "ja": "Helsinki-NLP/opus-mt-jap-en",
        }

        logging.info("Hugging Face translation engine ready.")

    def _precache_translation_models(self):
        """Downloads and caches translation models at startup."""
        logging.info("Precaching all translation models...")

        all_models = set(self.lang_to_model_map.values()) | set(
            self.reverse_lang_to_model_map.values()
        )

        for model_name in all_models:
            try:
                _ = pipeline("translation", model=model_name)
                logging.info(f"Cached model: {model_name}")
            except Exception:
                logging.warning(f"Could not cache model {model_name}.")

        logging.info("Translation model precaching complete.")

    def _translate_generic(self, text, target_lang, source_lang, model_map, cache):
        model_name = model_map.get(source_lang if target_lang == "en" else target_lang)
        if not model_name:
            return text

        try:
            cache_key = source_lang if target_lang == "en" else target_lang
            if cache_key not in cache:
                cache[cache_key] = pipeline("translation", model=model_name)
            return cache[cache_key](text)[0]["translation_text"]
        except Exception as e:
            logging.error(f"Failed to translate: {e}")
            return text

    def translate_text(self, text, target_lang, source_lang="en"):
        if source_lang == target_lang:
            return text
        if target_lang == "en":
            return self._translate_generic(
                text, "en", source_lang, self.reverse_lang_to_model_map, self.reverse_translator_cache
            )
        return self._translate_generic(
            text, target_lang, "en", self.lang_to_model_map, self.translator_cache
        )

    def build_vector_store(self, domain):
        collection_name = f"medical_rag_{domain}"
        json_path = os.path.join(PROCESSED_DATA_DIR, f"{domain}_chunks.json")

        if not os.path.exists(json_path):
            logging.error(f"Processed data not found for domain '{domain}' at {json_path}")
            return

        try:
            collection = self.vector_store_client.get_collection(name=collection_name)
            if collection.count() > 0:
                logging.info(f"Vector store for '{domain}' already exists. Skipping build.")
                return
        except Exception:
            logging.info(f"No collection found for '{domain}'. Building new one.")

        with open(json_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        Chroma.from_texts(
            texts=[d["page_content"] for d in docs],
            embedding=self.embedding_function,
            metadatas=[d["metadata"] for d in docs],
            collection_name=collection_name,
            client=self.vector_store_client,
            persist_directory=PERSIST_DIRECTORY,
        )

        logging.info(f"Vector store for '{domain}' built successfully.")

    def _initialize_chains(self):
        """Initializes RetrievalQA chains for each domain."""
        chains = {}

        prompt_template = """
        You are a helpful and precise public health AI assistant.
        Use the context provided to answer the user's question.

        --- EXAMPLE 1
        Context: Outbreak Report ID 123. State: Minnesota. Year: 2022. Etiology: Norovirus. Setting: Restaurant. Illnesses: 50.
        Question: How many illnesses from Norovirus were in Minnesota in 2022?
        Helpful Answer: According to the provided data, there were 50 illnesses from Norovirus in Minnesota in 2022.

        --- EXAMPLE 2
        Context: Outbreak Report ID 456. State: Florida. Year: 2023. Etiology: Salmonella. Setting: Restaurant. Illnesses: 20.
        Question: Were there any outbreaks in schools in New York?
        Helpful Answer: Based on the provided context, there is no information about outbreaks in schools in New York.

        --- REAL CONTEXT AND QUESTION:
        Context: {context}
        Question: {question}
        Helpful Answer:
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

                logging.info(f"QA chain for '{domain}' initialized.")
            except Exception as e:
                logging.warning(f"Could not initialize chain for '{domain}'. Error: {e}")

        return chains

    def query(self, user_query, domain):
        if domain not in self.chains:
            return {"result": f"Error: Domain '{domain}' not ready.", "source_documents": []}

        try:
            original_lang = detect(user_query)
            english_query = (
                self.translate_text(user_query, "en", source_lang=original_lang)
                if original_lang != "en"
                else user_query
            )
        except Exception:
            original_lang, english_query = "en", user_query

        try:
            response = self.chains[domain].invoke({"query": english_query})
            if original_lang != "en":
                response["result"] = self.translate_text(response["result"], original_lang, source_lang="en")
            return response
        except Exception as e:
            logging.error(f"Error during RAG query: {e}")
            return {"result": "Sorry, an error occurred. Please try again.", "source_documents": []}


if __name__ == "__main__":
    rag_system = MedicalRAG()
    for domain in ["outbreak", "symptom", "misinformation"]:
        rag_system.build_vector_store(domain)
