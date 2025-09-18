# rag_core.py

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

class MedicalRAG:
    def __init__(self, data_dir="src/data"):
        self.data_dir = data_dir
        self.client = chromadb.PersistentClient(path="chroma_db")

        # Embedding function
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Collections
        self.outbreaks_col = self.client.get_or_create_collection(
            name="outbreaks", embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"}
        )
        self.misinfo_col = self.client.get_or_create_collection(
            name="misinformation", embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"}
        )
        self.symptoms_col = self.client.get_or_create_collection(
            name="symptoms", embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"}
        )

        # Load datasets
        self._load_outbreaks()
        self._load_misinformation()
        self._load_symptoms()

    # -----------------------------
    # Data Loading
    # -----------------------------
    def _load_outbreaks(self):
        csv_path = os.path.join(self.data_dir, "outbreaks_data.csv")
        df = pd.read_csv(csv_path)

        if self.outbreaks_col.count() == 0:
            for idx, row in df.iterrows():
                text = f"Outbreak Report {row['Outbreak Report ID']}. State: {row['State']}. Year: {row['Year']}. Etiology: {row['Etiology']}. Illnesses: {row['Illnesses']}."
                metadata = {
                    "disease": str(row["Etiology"]).lower(),
                    "year": int(row["Year"]),
                    "state": str(row["State"])
                }
                self.outbreaks_col.add(
                    documents=[text],
                    ids=[f"outbreak_{row['Outbreak Report ID']}"],
                    metadatas=[metadata]
                )

    def _load_misinformation(self):
        csv_path = os.path.join(self.data_dir, "misinformation_data.csv")
        df = pd.read_csv(csv_path)

        if self.misinfo_col.count() == 0:
            for idx, row in df.iterrows():
                text = f"Claim: {row['Claim']} | Verdict: {row['Verdict']}"
                self.misinfo_col.add(
                    documents=[text],
                    ids=[f"misinfo_{idx}"],
                    metadatas=[{"verdict": row["Verdict"].lower()}]
                )

    def _load_symptoms(self):
        csv_path = os.path.join(self.data_dir, "symptoms_data.csv")
        df = pd.read_csv(csv_path)

        if self.symptoms_col.count() == 0:
            for idx, row in df.iterrows():
                disease = row["Disease"]
                symptoms = row["Symptoms"]
                text = f"Disease: {disease}. Symptoms: {symptoms}"
                self.symptoms_col.add(
                    documents=[text],
                    ids=[f"symptom_{idx}"],
                    metadatas=[{"disease": disease.lower()}]
                )

    # -----------------------------
    # Query Methods
    # -----------------------------
    def query_outbreaks(self, diseases: list, years: list):
        results = []
        for disease in diseases:
            for year in years:
                res = self.outbreaks_col.query(
                    query_texts=[f"{disease} {year}"],
                    n_results=3,
                    where={"disease": disease.lower(), "year": year}
                )
                results.extend(res["documents"][0] if res["documents"] else [])
        return results if results else ["No matching outbreak data found."]

    def query_misinformation(self, claim: str):
        res = self.misinfo_col.query(query_texts=[claim], n_results=1)
        if res["documents"]:
            return res["documents"][0][0]
        return "No misinformation record found."

    def query_symptoms(self, symptom_query: str):
        res = self.symptoms_col.query(query_texts=[symptom_query], n_results=3)
        return res["documents"][0] if res["documents"] else ["No disease match found."]

    # -----------------------------
    # Smart Dispatcher
    # -----------------------------
    def answer_query(self, query: str):
        query_lc = query.lower()

        # Outbreak queries
        if any(d in query_lc for d in ["malaria", "dengue", "rabies", "disease", "outbreak", "cases"]):
            diseases = []
            if "malaria" in query_lc: diseases.append("malaria")
            if "dengue" in query_lc: diseases.append("dengue")
            if "rabies" in query_lc: diseases.append("rabies")

            years = []
            if "2024" in query_lc: years.append(2024)
            if "2025" in query_lc: years.append(2025)

            return self.query_outbreaks(diseases, years)

        # Misinformation queries
        if any(x in query_lc for x in ["safe", "dangerous", "fake", "myth", "claim"]):
            return self.query_misinformation(query)

        # Symptom checker
        if any(x in query_lc for x in ["fever", "cough", "headache", "pain", "nausea", "symptom"]):
            return self.query_symptoms(query)

        return ["Sorry, I couldn’t understand your query."]
