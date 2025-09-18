# src/app.py
import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag_core import MedicalRAG

st.set_page_config(page_title="Medical AI Assistant", page_icon="⚕️", layout="wide")

@st.cache_resource
def load_rag_system():
    """Loads the RAG system once and caches it."""
    return MedicalRAG()

# This is the only time MedicalRAG() is called.
# It handles all initialization, including building vector stores if needed.
rag_system = load_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = {}

st.title("⚕️ Multilingual Medical AI Assistant")
st.caption("Powered by Retrieval-Augmented Generation")

with st.sidebar:
    st.header("Settings")
    selected_domain_ui = st.selectbox(
        "Choose a Public Health Domain:",
        ("Symptom Analysis", "Outbreak Alerts", "Misinformation Classification")
    )
    st.divider()
    st.warning(
        "**Disclaimer:** This is a proof-of-concept AI assistant and not a substitute for professional medical advice."
    )

domain_map = {
    "Symptom Analysis": "symptom",
    "Outbreak Alerts": "outbreak",
    "Misinformation Classification": "misinformation"
}
current_domain_key = domain_map[selected_domain_ui]

if current_domain_key not in st.session_state.messages:
    st.session_state.messages[current_domain_key] = []

for message in st.session_state.messages[current_domain_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, doc in enumerate(message["sources"]):
                    # Some source documents may be dict-like instead of proper metadata object
                    source_meta = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})
                    page_content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else str(doc))
                    st.info(f"**Source {i+1}:** {source_meta.get('source', 'N/A')}\n\n**Content:** {page_content}")

if prompt := st.chat_input(f"Ask about {selected_domain_ui}..."):
    st.session_state.messages[current_domain_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_system.query(prompt, current_domain_key)
            # response is expected to be a dict: {"result": "...", "source_documents": [...]}
            result_text = response.get("result") if isinstance(response, dict) else str(response)
            st.markdown(result_text)

            source_docs = response.get("source_documents") if isinstance(response, dict) else None
            if source_docs:
                with st.expander("View Sources"):
                    for i, doc in enumerate(source_docs):
                        source_meta = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})
                        page_content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else str(doc))
                        st.info(f"**Source {i+1}:** {source_meta.get('source', 'N/A')}\n\n**Content:** {page_content}")

            assistant_message = {"role": "assistant", "content": result_text, "sources": source_docs}
            st.session_state.messages[current_domain_key].append(assistant_message)
