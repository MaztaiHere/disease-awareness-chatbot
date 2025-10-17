import os
import pandas as pd
import json
import logging
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging to show timestamps and severity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Ensures paths are relative to the project's root directory
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for interactive environments like Jupyter
    BASE_DIR = Path.cwd()

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# --- Main Processing Logic ---

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

def process_file(file_name, processing_function):
    """A wrapper function to handle file checking, processing, and error logging."""
    input_path = RAW_DATA_DIR / file_name
    logging.info(f"Checking for '{file_name}' at: {input_path}")

    if not input_path.exists():
        logging.warning(f"-> [SKIP] File not found. Please place it in the data/raw directory.")
        return

    logging.info(f"-> [FOUND] File '{file_name}' found. Starting processing...")
    try:
        processing_function(input_path)
        logging.info(f"✅ [SUCCESS] Finished processing '{file_name}'.")
    except Exception as e:
        logging.error(f"❌ [ERROR] An unexpected error occurred while processing '{file_name}': {e}", exc_info=True)

def preprocess_outbreak_data(input_path):
    """Preprocesses NORS data for the outbreak alerts domain."""
    df = pd.read_csv(input_path, low_memory=False)
    logging.info(f"   - Loaded {len(df)} rows from CSV.")
    if df.empty:
        logging.warning("   - The CSV file is empty. Nothing to process.")
        return

    logging.info("   - Combining text columns into a single field...")
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

    logging.info("   - Splitting all texts into chunks...")
    all_chunks = []
    for index, row in df.iterrows():
        source_id = f"Outbreaks Report {index}"
        all_chunks.extend(create_chunks(row['combined_text'], source_id))
    
    if not all_chunks:
        logging.warning("   - No chunks were created from the data. Skipping JSON file creation.")
        return
        
    logging.info(f"   - Created a total of {len(all_chunks)} chunks.")
    
    output_path = PROCESSED_DATA_DIR / "outbreak_chunks.json"
    logging.info(f"   - Saving chunks to JSON file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)

def preprocess_triage_data(input_path):
    """Preprocesses Kaggle symptom data for the symptom triage domain."""
    df = pd.read_csv(input_path)
    logging.info(f"   - Loaded {len(df)} rows from CSV.")
    if df.empty:
        logging.warning("   - The CSV file is empty. Nothing to process.")
        return

    def combine_symptoms(row):
        symptom_cols = [col for col in df.columns if 'Symptom_' in col]
        symptoms = [str(row[col]).strip().replace('_', ' ') for col in symptom_cols if pd.notna(row[col])]
        symptoms_text = ", ".join(symptoms) if symptoms else "Not specified"
        return f"Disease Profile: {row.get('Disease', 'N/A')}. Symptoms: {symptoms_text}."

    logging.info("   - Combining symptom columns into a single field...")
    df['combined_text'] = df.apply(combine_symptoms, axis=1)

    logging.info("   - Splitting all texts into chunks...")
    all_chunks = []
    for index, row in df.iterrows():
        source_id = f"Symptom Profile {index} ({row.get('Disease', 'Unknown')})"
        all_chunks.extend(create_chunks(row['combined_text'], source_id))
    
    if not all_chunks:
        logging.warning("   - No chunks were created from the data. Skipping JSON file creation.")
        return

    logging.info(f"   - Created a total of {len(all_chunks)} chunks.")

    output_path = PROCESSED_DATA_DIR / "symptom_chunks.json"
    logging.info(f"   - Saving chunks to JSON file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)

def preprocess_misinformation_data(input_path):
    """Preprocesses misinformation data for the misinformation classification domain."""
    df = pd.read_csv(input_path)
    logging.info(f"   - Loaded {len(df)} rows from CSV.")
    df.dropna(subset=['title', 'text'], inplace=True)
    logging.info(f"   - {len(df)} rows remaining after removing entries with missing titles/text.")
    if df.empty:
        logging.warning("   - No valid data after cleaning. Nothing to process.")
        return

    logging.info("   - Combining title and text columns...")
    df['combined_text'] = "Title: " + df['title'].astype(str) + "; Text: " + df['text'].astype(str)
    
    logging.info("   - Splitting all texts into chunks...")
    all_chunks = []
    for index, row in df.iterrows():
        label = "Real" if row['label'] == 1 else "Fake"
        source_id = f"Misinformation Article {row.get('Unnamed: 0', index)} ({label})"
        all_chunks.extend(create_chunks(row['combined_text'], source_id))
    
    if not all_chunks:
        logging.warning("   - No chunks were created from the data. Skipping JSON file creation.")
        return
        
    logging.info(f"   - Created a total of {len(all_chunks)} chunks.")

    output_path = PROCESSED_DATA_DIR / "misinformation_chunks.json"
    logging.info(f"   - Saving chunks to JSON file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)

if __name__ == "__main__":
    logging.info("--- Starting Data Preprocessing Pipeline ---")
    
    # Ensure the target directories exist before we start
    logging.info(f"Ensuring raw data directory exists at: {RAW_DATA_DIR}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensuring processed data directory exists at: {PROCESSED_DATA_DIR}")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("-" * 50)
    process_file("outbreaks_data.csv", preprocess_outbreak_data)
    print("-" * 50)
    process_file("symptoms_data.csv", preprocess_triage_data)
    print("-" * 50)
    process_file("misinformation_data.csv", preprocess_misinformation_data)
    print("-" * 50)
    
    logging.info("--- Data Preprocessing Pipeline Finished ---")