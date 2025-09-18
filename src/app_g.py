# src/app.py
import streamlit as st
from src.rag_core_g import MedicalRAG

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
                    st.info(f"**Source {i+1}:** {doc.metadata.get('source', 'N/A')}\n\n**Content:** {doc.page_content}")

if prompt := st.chat_input(f"Ask about {selected_domain_ui}..."):
    st.session_state.messages[current_domain_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_system.query(prompt, current_domain_key)
            st.markdown(response["result"])
            source_docs = response.get("source_documents")
            if source_docs:
                with st.expander("View Sources"):
                    for i, doc in enumerate(source_docs):
                        st.info(f"**Source {i+1}:** {doc.metadata.get('source', 'N/A')}\n\n**Content:** {doc.page_content}")
            
            assistant_message = {"role": "assistant", "content": response["result"], "sources": source_docs}
            st.session_state.messages[current_domain_key].append(assistant_message)