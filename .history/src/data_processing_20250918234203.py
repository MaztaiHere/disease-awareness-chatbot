# data_processing.py - Improved version
import pandas as pd
import json
import logging
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_chunks(text, source):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return [{"page_content": chunk, "metadata": {"source": source}} for chunk in chunks]

def preprocess_triage_data():
    logging.info("Preprocessing Symptom data...")
    input_path = RAW_DATA_DIR / "symptoms_data.csv"
    if not input_path.exists():
        logging.error(f"Symptom data file not found")
        return

    df = pd.read_csv(input_path)
    
    # Clean and standardize data
    df = df.dropna(subset=['Disease'])
    symptom_cols = [col for col in df.columns if 'Symptom_' in col]
    
    def combine_symptoms(row):
        symptoms = [
            str(row[col]).strip().replace('_', ' ').lower()
            for col in symptom_cols
            if pd.notna(row[col]) and str(row[col]).strip() != 'nan'
        ]
        symptoms_text = ", ".join(symptoms) if symptoms else "no specific symptoms"
        return f"Disease: {row.get('Disease', 'Unknown')}. Symptoms: {symptoms_text}."

    df['combined_text'] = df.apply(combine_symptoms, axis=1)

    all_chunks = []
    for index, row in df.iterrows():
        source_id = f"Symptom {index} ({row.get('Disease', 'Unknown')})"
        all_chunks.extend(create_chunks(row['combined_text'], source_id))

    with open(PROCESSED_DATA_DIR / "symptom_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    logging.info("Finished preprocessing symptom data.")

if __name__ == "__main__":
    logging.info("Starting data preprocessing...")
    preprocess_triage_data()
    logging.info("Data preprocessing finished.")