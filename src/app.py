# src/app.py
import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag_core import MedicalRAG, LANGUAGE_NAMES

st.set_page_config(page_title="Medical AI Assistant", page_icon="⚕️", layout="wide")

@st.cache_resource
def load_rag_system():
    """Loads the RAG system once and caches it."""
    return MedicalRAG()

rag_system = load_rag_system()

# Initialize session state with proper structure
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    
if "current_domain" not in st.session_state:
    st.session_state.current_domain = "symptom"
    
if "source_language" not in st.session_state:
    st.session_state.source_language = "en"
    
if "target_language" not in st.session_state:
    st.session_state.target_language = "en"

# Domain mapping with descriptions
DOMAIN_CONFIG = {
    "symptom": {
        "ui_name": "Symptom Analysis",
        "description": "Ask about medical symptoms, conditions, and treatments",
        "placeholder": "Describe symptoms or ask about medical conditions..."
    },
    "outbreak": {
        "ui_name": "Outbreak Alerts", 
        "description": "Get information about disease outbreaks and public health alerts",
        "placeholder": "Ask about disease outbreaks, locations, or case numbers..."
    },
    "misinformation": {
        "ui_name": "Fact Checker",
        "description": "Verify medical claims and combat misinformation",
        "placeholder": "Paste a medical claim or ask about health misinformation..."
    }
}

st.title("⚕️ Multilingual Medical AI Assistant")
st.caption("Powered by Retrieval-Augmented Generation")

# Main layout with two columns
col1, col2 = st.columns([3, 1])

with col2:  # Sidebar moved to right column for better visibility
    st.header("⚙️ Settings")
    
    # Language Configuration
    st.subheader("🌐 Language Settings")
    
    source_lang = st.selectbox(
        "**Input Language** (Your question language):",
        options=list(LANGUAGE_NAMES.keys()),
        format_func=lambda x: f"{LANGUAGE_NAMES[x]} ({x})",
        index=list(LANGUAGE_NAMES.keys()).index(st.session_state.source_language),
        key="source_lang_select"
    )
    
    target_lang = st.selectbox(
        "**Output Language** (Answer language):",
        options=list(LANGUAGE_NAMES.keys()),
        format_func=lambda x: f"{LANGUAGE_NAMES[x]} ({x})",
        index=list(LANGUAGE_NAMES.keys()).index(st.session_state.target_language),
        key="target_lang_select"
    )
    
    # Update session state if languages changed
    if source_lang != st.session_state.source_language:
        st.session_state.source_language = source_lang
        st.rerun()
        
    if target_lang != st.session_state.target_language:
        st.session_state.target_language = target_lang
        st.rerun()
    
    st.divider()
    
    # Domain Selection
    st.subheader("📊 Domain Selection")
    
    for domain_key, domain_info in DOMAIN_CONFIG.items():
        if st.button(
            f"**{domain_info['ui_name']}**",
            key=f"domain_btn_{domain_key}",
            use_container_width=True,
            type="primary" if st.session_state.current_domain == domain_key else "secondary"
        ):
            st.session_state.current_domain = domain_key
            st.rerun()
        
        st.caption(f"_{domain_info['description']}_")
        st.write("")
    
    st.divider()
    st.warning(
        "**Disclaimer:** This is a proof-of-concept AI assistant and not a substitute for professional medical advice."
    )

with col1:  # Main chat area
    # Current configuration display
    current_domain_config = DOMAIN_CONFIG[st.session_state.current_domain]
    
    st.subheader(f"💬 {current_domain_config['ui_name']}")
    
    # Configuration status
    config_col1, config_col2, config_col3 = st.columns(3)
    with config_col1:
        st.metric("Input Language", f"{LANGUAGE_NAMES[st.session_state.source_language]}")
    with config_col2:
        st.metric("Output Language", f"{LANGUAGE_NAMES[st.session_state.target_language]}")
    with config_col3:
        st.metric("Domain", current_domain_config['ui_name'])
    
    st.write("---")
    
    # Initialize chat session for current domain if not exists
    if st.session_state.current_domain not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.current_domain] = []
    
    current_messages = st.session_state.chat_sessions[st.session_state.current_domain]
    
    # Display chat messages with avatars
    for message in current_messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "⚕️"):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander(f"📚 View {len(message['sources'])} Sources"):
                    for i, doc in enumerate(message["sources"]):
                        source_meta = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})
                        page_content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else str(doc))
                        
                        st.write(f"**Source {i+1}:** `{source_meta.get('source', 'Unknown')}`")
                        st.caption(page_content[:300] + "..." if len(page_content) > 300 else page_content)
                        st.divider()
    
    # Chat input with dynamic placeholder
    if prompt := st.chat_input(current_domain_config['placeholder']):
        # Add user message to chat
        user_message = {"role": "user", "content": prompt}
        current_messages.append(user_message)
        
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant", avatar="⚕️"):
            with st.spinner(f"🔍 Searching {current_domain_config['ui_name']} knowledge..."):
                try:
                    response = rag_system.query(
                        user_query=prompt,
                        domain=st.session_state.current_domain,
                        source_lang=st.session_state.source_language,
                        target_lang=st.session_state.target_language
                    )
                    
                    result_text = response.get("result", "I couldn't generate a response. Please try again.")
                    source_docs = response.get("source_documents", [])
                    
                    # Display response
                    st.markdown(result_text)
                    
                    # Display sources if available
                    if source_docs:
                        with st.expander(f"📚 View {len(source_docs)} Sources"):
                            for i, doc in enumerate(source_docs):
                                source_meta = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})
                                page_content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else str(doc))
                                
                                st.write(f"**Source {i+1}:** `{source_meta.get('source', 'Unknown')}`")
                                st.caption(page_content[:300] + "..." if len(page_content) > 300 else page_content)
                                st.divider()
                    
                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant", 
                        "content": result_text, 
                        "sources": source_docs
                    }
                    current_messages.append(assistant_message)
                    
                except Exception as e:
                    error_msg = "Sorry, I encountered an error processing your request. Please try again."
                    st.error(error_msg)
                    st.session_state.chat_sessions[st.session_state.current_domain].append({
                        "role": "assistant", 
                        "content": error_msg, 
                        "sources": []
                    })
    
    # Clear chat button for current domain
    if current_messages:
        st.write("---")
        if st.button(f"🗑️ Clear Chat History for {current_domain_config['ui_name']}", use_container_width=True):
            st.session_state.chat_sessions[st.session_state.current_domain] = []
            st.rerun()