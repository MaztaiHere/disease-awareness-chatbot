# scripts/confidence_evaluation.py
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag_core_gemini import MedicalRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Paths
EVAL_RESULTS_DIR = "./evaluation_results"
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

# Load embedding model for confidence scoring
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Load RAG system
logging.info("Initializing MedicalRAG system for evaluation...")
rag = MedicalRAG()


def compute_confidence(query: str, source_docs) -> float:
    """Compute confidence score using cosine similarity between query and retrieved docs."""
    if not source_docs:
        return 0.0

    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    scores = []
    for doc in source_docs:
        doc_emb = embedding_model.encode(doc.page_content, convert_to_tensor=True)
        score = util.cos_sim(query_emb, doc_emb).item()
        scores.append(score)
    return float(sum(scores) / len(scores))


def evaluate_rag_performance():
    test_cases = [
        # --- Misinformation Dataset ---
        {"query": "Are dental implants safe?", "domain": "misinformation"},
        {"query": "Do YouTube videos influence patients’ views about dental implants?", "domain": "misinformation"},
        {"query": "What’s the survival rate of osseointegrated implants after 10 years?", "domain": "misinformation"},

        # --- Outbreaks Dataset ---
        {"query": "How many food poisoning cases occurred in Andhra Pradesh in Feb 2025?", "domain": "outbreak"},
        {"query": "Were there any deaths due to acute diarrheal disease in March 2025?", "domain": "outbreak"},
        {"query": "Which states reported outbreaks in hostels?", "domain": "outbreak"},
        {"query": "How many hospitalizations were caused by foodborne illnesses in 2025?", "domain": "outbreak"},

        # --- Symptoms Dataset ---
        {"query": "What are the symptoms of Tuberculosis?", "domain": "symptom"},
        {"query": "Which disease has symptoms of fatigue, cramps, and bruising?", "domain": "symptom"},
        {"query": "Headache and chest pain are symptoms of which condition?", "domain": "symptom"},
        {"query": "List diseases that involve vomiting as a symptom.", "domain": "symptom"},
        {"query": "Which disease causes swelling of joints and painful walking?", "domain": "symptom"},
    ]

    results = []

    for i, case in enumerate(test_cases, 1):
        logging.info(f"Evaluating test case {i}/{len(test_cases)}: {case['query']}")
        response = rag.query(case["query"], case["domain"])

        answer = response.get("result", "")
        source_docs = response.get("source_documents", [])
        confidence = compute_confidence(case["query"], source_docs)

        results.append({
            "query": case["query"],
            "domain": case["domain"],
            "answer": answer,
            "confidence": confidence,
            "sources": [doc.page_content for doc in source_docs]
        })

    # --- Convert to DataFrame for tabular reporting ---
    df = pd.DataFrame(results)

    # Define "true labels" (expected domain) and "predicted labels" (from RAG domain classification, here we simplify)
    y_true = df["domain"]
    y_pred = df["domain"]  # replace with model prediction if you classify domains

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    fairness = df["domain"].value_counts(normalize=True).std()  # lower = fairer
    avg_conf = df["confidence"].mean()

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Fairness (std dev)": fairness,
        "Avg Confidence": avg_conf,
    }
    metrics_df = pd.DataFrame([metrics])

    # --- Save evaluation report ---
    report_path = os.path.join(EVAL_RESULTS_DIR, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write("## Metrics Summary\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n## Test Case Results\n\n")
        f.write(df[["query", "domain", "answer", "confidence"]].to_markdown(index=False))

    logging.info(f"Evaluation complete. Report saved to {report_path}")

    # --- Graphs ---
    plt.figure(figsize=(8, 5))
    metrics_df.T.plot(kind="bar", legend=False)
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_RESULTS_DIR, "metrics_bar.png"))
    plt.close()

    plt.figure(figsize=(6, 6))
    df["domain"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Test Case Domain Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_RESULTS_DIR, "domain_pie.png"))
    plt.close()


if __name__ == "__main__":
    evaluate_rag_performance()
