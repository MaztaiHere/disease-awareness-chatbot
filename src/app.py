import streamlit as st
import sys, os,time

# Fix imports when running from inside "src" directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from rag_core import MedicalRAG, LANGUAGE_NAMES

st.set_page_config(page_title="Medical AI Assistant", page_icon="⚕️", layout="wide")


@st.cache_resource
def load_rag_system():
    """Load the RAG once and keep models in memory"""
    return MedicalRAG()

rag_system = load_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = {}
if "language" not in st.session_state:
    st.session_state.language = "en"
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = "Symptom Analysis"

st.title("⚕️ Medical AI Assistant")
st.caption("Multilingual Symptom Analysis, Outbreak Alerts & Misinformation Detection")

with st.sidebar:
    st.header("Settings")
    selected_language = st.selectbox(
        "Choose Language:",
        options=list(LANGUAGE_NAMES.keys()),
        format_func=lambda x: LANGUAGE_NAMES.get(x, x),
        index=list(LANGUAGE_NAMES.keys()).index(st.session_state.language)
    )
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.experimental_rerun()

    st.subheader("Choose Domain:")
    if st.button("🩺 Symptom Analysis", use_container_width=True):
        st.session_state.selected_domain = "Symptom Analysis"
        st.experimental_rerun()
    if st.button("🚨 Outbreak Alerts", use_container_width=True):
        st.session_state.selected_domain = "Outbreak Alerts"
        st.experimental_rerun()
    if st.button("📢 Misinfo Check", use_container_width=True):
        st.session_state.selected_domain = "Misinformation Classification"
        st.experimental_rerun()

    st.info(f"**Current:** {st.session_state.selected_domain}")
    st.divider()
    st.warning("*Disclaimer:* This is a proof-of-concept AI assistant and not a substitute for professional medical advice.")

domain_map = {
    "Symptom Analysis": "symptom",
    "Outbreak Alerts": "outbreak",
    "Misinformation Classification": "misinformation"
}
current_domain_key = domain_map[st.session_state.selected_domain]
if current_domain_key not in st.session_state.messages:
    st.session_state.messages[current_domain_key] = []

domain_titles = {
    "symptom": "🩺 Symptom Analysis",
    "outbreak": "🚨 Outbreak Alerts",
    "misinformation": "📢 Misinformation Check"
}

st.subheader(domain_titles[current_domain_key])

for message in st.session_state.messages[current_domain_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, doc in enumerate(message["sources"]):
                    source_meta = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})
                    page_content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else str(doc))
                    st.info(f"*Source {i+1}:* {source_meta.get('source', 'N/A')}\n\n*Content:* {page_content}")

prompt = st.chat_input(f"Ask about {st.session_state.selected_domain}...")
if prompt:
    st.session_state.messages[current_domain_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing — optimizing for speed..."):
            t0 = time.time()
            response = rag_system.query(
                prompt,
                current_domain_key,
                source_lang=st.session_state.language,
                target_lang=st.session_state.language
            )
            t1 = time.time()
            result_text = response.get("result") if isinstance(response, dict) else str(response)
            st.markdown(result_text)
            with st.expander("Performance"):
                st.write(f"⏱️ Total time: {t1 - t0:.2f}s (goal: 3–5s)")
            source_docs = response.get("source_documents") if isinstance(response, dict) else None
            if source_docs:
                with st.expander("View Sources"):
                    for i, doc in enumerate(source_docs):
                        source_meta = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})
                        page_content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else str(doc))
                        st.info(f"*Source {i+1}:* {source_meta.get('source', 'N/A')}\n\n*Content:* {page_content}")

            assistant_message = {"role": "assistant", "content": result_text, "sources": source_docs}
            st.session_state.messages[current_domain_key].append(assistant_message)
