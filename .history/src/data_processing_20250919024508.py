import os
import pandas as pd
import json
import logging
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Ensures paths are relative to the project's root directory
BASE_DIR = Path(_file_).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- Data Preprocessing and Chunking Functions ---

def create_chunks(text, source):
    """Creates overlapping chunks from a given text."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(text)
    return [{"page_content": chunk, "metadata": {"source": source}} for chunk in chunks]

def preprocess_outbreak_data():
    """Preprocesses NORS data for the outbreak alerts domain."""
    logging.info("Preprocessing Outbreak data...")
    df = pd.read_csv(RAW_DATA_DIR / "outbreaks_data.csv", low_memory=False)
    df['combined_text'] = df.apply(
        lambda row: f"Outbreak Report ID {row.name}. "
                    f"State: {row.get('State', 'N/A')}. "
                    f"Year: {row.get('Year', 'N/A')}. "
                    f"Primary Mode: {row.get('Primary Mode', 'N/A')}. "
                    f"Etiology: {row.get('Etiology', 'N/A')}. "
                    f"Setting: {row.get('Setting', 'N/A')}. "
                    f"Illnesses: {row.get('Illnesses', 'N/A')}.",
        axis=1
    )
    all_chunks = []
    for index, row in df.iterrows():
        source_id = f"Outbreaks Report {index}"
        all_chunks.extend(create_chunks(row['combined_text'], source_id))

    with open(PROCESSED_DATA_DIR / "outbreak_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    logging.info("Finished preprocessing outbreak data.")

def preprocess_triage_data():
    """
    Preprocesses Kaggle symptom data for the symptom triage domain.
    This version correctly combines all symptom columns.
    """
    logging.info("Preprocessing Symptom (Triage) data...")
    input_path = RAW_DATA_DIR / "symptoms_data.csv"
    if not input_path.exists():
        logging.error(f"Symptom data file not found at {input_path}")
        return

    df = pd.read_csv(input_path)

    # Helper function to combine symptoms from multiple columns
    def combine_symptoms(row):
        symptom_cols = [col for col in df.columns if 'Symptom_' in col]
        symptoms = [
            str(row[col]).strip().replace('_', ' ')
            for col in symptom_cols
            if pd.notna(row[col])
        ]
        symptoms_text = ", ".join(symptoms) if symptoms else "Not specified"
        return (f"Disease Profile: {row.get('Disease', 'N/A')}. "
                f"Symptoms: {symptoms_text}.")

    df['combined_text'] = df.apply(combine_symptoms, axis=1)

    all_chunks = []
    for index, row in df.iterrows():
        source_id = f"Symptom Profile {index} ({row.get('Disease', 'Unknown')})"
        all_chunks.extend(create_chunks(row['combined_text'], source_id))

    # Save the processed data to the requested filename
    with open(PROCESSED_DATA_DIR / "symptom_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    logging.info("Finished preprocessing symptom (triage) data.")


def preprocess_misinformation_data():
    """Preprocesses misinformation data for the misinformation classification domain."""
    logging.info("Preprocessing Misinformation data...")
    df = pd.read_csv(RAW_DATA_DIR / "misinformation_data.csv")
    df.dropna(subset=['title', 'text'], inplace=True)
    df['combined_text'] = "Title: " + df['title'].astype(str) + "; Text: " + df['text'].astype(str)
    all_chunks = []
    for index, row in df.iterrows():
        label = "Real" if row['label'] == 1 else "Fake"
        source_id = f"Misinformation Article {row.get('Unnamed: 0', index)} ({label})"
        all_chunks.extend(create_chunks(row['combined_text'], source_id))

    with open(PROCESSED_DATA_DIR / "misinformation_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    logging.info("Finished preprocessing misinformation data.")

if _name_ == "_main_":
    logging.info("Starting data preprocessing...")

    if (RAW_DATA_DIR / "outbreaks_data.csv").exists():
        preprocess_outbreak_data()
    else:
        logging.warning(f"File not found: {RAW_DATA_DIR / 'outbreaks_data.csv'}. Skipping outbreak data.")

    if (RAW_DATA_DIR / "symptoms_data.csv").exists():
        preprocess_triage_data()
    else:
        logging.warning(f"File not found: {RAW_DATA_DIR / 'symptoms_data.csv'}. Skipping triage data.")

    if (RAW_DATA_DIR / "misinformation_data.csv").exists():
        preprocess_misinformation_data()
    else:
        logging.warning(f"File not found: {RAW_DATA_DIR / 'misinformation_data.csv'}. Skipping misinformation data.")

    logging.info("Data preprocessing finished.")
