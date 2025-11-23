# app.py
import os
from dotenv import load_dotenv
load_dotenv()
print("🔐 Loaded API key:", os.getenv("GROQ_API_KEY"))

import streamlit as st
print(">>> app.py executed")
st.write("UI loaded successfully.")
    
import logging
import uuid

from models.llm import call_llm
from models.embeddings import get_embeddings, get_embedding_dimension
from utils.rag_utils import FAISSStore, build_or_update_index_from_documents
from utils.web_search import serpapi_search, summarize_web_results
from config.config import TOP_K

st.set_page_config(page_title="RAG Chatbot (Groq)", layout="wide")

# --- initialize vector store using embedding dimension from sentence-transformers ---
EMBED_DIM = get_embedding_dimension() or 384  # fallback if model dimension not loaded
store = FAISSStore(dim=EMBED_DIM)

# --- UI ---
st.title("RAG Chatbot (Groq + Local Embeddings)")

with st.sidebar:
    st.header("Settings")
    response_mode = st.radio("Response mode", ("Concise", "Detailed"))
    use_web = st.checkbox("Allow live web search", value=True)
    top_k = st.number_input("Top-k retrieved chunks", min_value=1, max_value=10, value=TOP_K)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2)

st.markdown("Upload local documents to add to knowledge base (txt, md).")
uploaded = st.file_uploader(
    "Upload documents (multiple)", accept_multiple_files=True, type=["txt", "md"]
)

if uploaded:
    docs = []
    for f in uploaded:
        try:
            raw = f.read().decode("utf-8")
        except Exception:
            raw = f.read().decode("latin-1", errors="ignore")
        docs.append({"id": str(uuid.uuid4()), "text": raw, "source": f.name})

    build_or_update_index_from_documents(docs, store)
    st.success(f"Indexed {len(uploaded)} document(s).")

# Conversation history
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input(
    "Ask something about your documents or the web:",
    key="query",
    placeholder="e.g., Summarize chapter 1 from uploaded notes"
)

if st.button("Ask") and query:
    st.session_state.history.append({"role": "user", "content": query})

    # 1) Retrieve from vector store (RAG)
    q_emb_list = get_embeddings([query])
    if q_emb_list:
        q_emb = q_emb_list[0]
        retrieved = store.query(q_emb, k=top_k)
    else:
        retrieved = []

    context_text = ""
    if retrieved:
        context_text += "Relevant documents:\n"
        for meta, dist in retrieved:
            context_text += (
                f"- Source: {meta.get('source')} | chunk_index: {meta.get('chunk_index')}\n"
                f"{meta.get('text')}\n\n"
            )

    # 2) Live web search (optional)
    web_summary = ""
    if use_web:
        try:
            web_hits = serpapi_search(query, num_results=4)
            if web_hits:
                web_summary = summarize_web_results(web_hits, call_llm)
        except Exception as e:
            logging.exception("Web search failed: %s", e)
            web_summary = ""

    # 3) Response mode instructions
    mode_instructions = {
        "Concise": (
            "Give a short, 2–3 sentence answer with only necessary details. "
            "If you use sources, mention 1–2 briefly."
        ),
        "Detailed": (
            "Give an in-depth answer with explanation, examples, and 3 practical action items. "
            "If you use sources, clearly mention them."
        ),
    }

    prompt = f"""
You are an expert assistant. Use the following retrieved document context and web search summary
to answer the user's query. If the context is empty, fall back to general knowledge.

User query:
{query}

Response style:
{mode_instructions[response_mode]}

Document context:
{context_text}

Web search summary:
{web_summary}

Now provide the answer in the requested style.
""".strip()

    # 4) Call LLM via Groq
    resp = call_llm(prompt, max_tokens=600, temperature=temperature)
    st.session_state.history.append({"role": "assistant", "content": resp})

# Display conversation
st.markdown("---")
for turn in st.session_state.history[::-1]:
    if turn["role"] == "assistant":
        st.markdown(f"**Assistant:** {turn['content']}")
    else:
        st.markdown(f"**You:** {turn['content']}")
