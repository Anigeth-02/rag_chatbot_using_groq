import os
import logging
import uuid

import streamlit as st
from dotenv import load_dotenv

import logging
import uuid
import streamlit as st
from dotenv import load_dotenv

from models.llm import call_llm
from models.embeddings import get_embeddings, get_embedding_dimension
from utils.rag_utils import FAISSStore, build_or_update_index_from_documents
from utils.web_search import serpapi_search, summarize_web_results
from config.config import TOP_K

load_dotenv()

# --- Streamlit page config ---
st.set_page_config(
    page_title="Study Notes Assistant (RAG + Groq)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize FAISS in session_state so it persists ---
EMBED_DIM = get_embedding_dimension() or 384

if "store" not in st.session_state:
    st.session_state.store = FAISSStore(dim=EMBED_DIM)

store: FAISSStore = st.session_state.store


# --- Title and subtitle ---
st.markdown(
    """
    <h1 style="margin-bottom:0.2rem;">📚 Study Notes Assistant</h1>
    <p style="color:gray;margin-top:0;">
        Upload your lecture notes or textbook text, and ask questions to review, summarize, or clarify concepts.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# --- Layout: left = chat, right = settings/docs ---
left_col, right_col = st.columns([2, 1])

# ========== RIGHT COLUMN: SETTINGS & DOCS INFO ==========
with right_col:
    st.subheader("⚙️ Settings")

    response_mode = st.radio(
        "Response mode",
        ("Concise", "Detailed"),
        help="Choose how detailed the answer should be.",
    )
    use_web = st.checkbox("Allow live web search (if configured)", value=False)
    top_k = st.number_input("Top-k retrieved chunks", min_value=1, max_value=10, value=TOP_K)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2)

    st.markdown("---")

    st.subheader("📂 Knowledge Base")

    st.markdown("Upload your study notes (`.txt`, `.md`) to build a personal knowledge base:")

    uploaded = st.file_uploader(
        "Upload documents (multiple)",
        accept_multiple_files=True,
        type=["txt", "md"],
        label_visibility="collapsed",
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
        st.caption(f"📌 Total chunks in KB now: {len(store.metadata)}")

        st.success(f"✅ Indexed {len(uploaded)} document(s).")

    sources = store.list_sources()
    if sources:
        st.markdown("**Currently indexed documents:**")
        for src, count in sources:
            st.markdown(f"- `{src}`  _(chunks: {count})_")
    else:
        st.info("No documents indexed yet. Upload notes to get started.")

    if st.button("🧹 Clear knowledge base"):
        store.clear()
        st.success("Knowledge base cleared. Upload new documents to start fresh.")

    st.markdown("---")

    if st.button("🧼 Clear chat history"):
        st.session_state.history = []
        st.success("Chat history cleared.")


# ========== LEFT COLUMN: CHAT INTERFACE ==========
with left_col:
    if "history" not in st.session_state:
        st.session_state.history = []

    user_query = st.text_input(
        "Ask something about your notes or a topic:",
        key="query",
        placeholder="e.g., Explain photosynthesis based on my notes.",
    )

    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and user_query:
        st.session_state.history.append({"role": "user", "content": user_query})

        # 1) RAG retrieval
        q_emb_list = get_embeddings([user_query])
        if q_emb_list:
            q_emb = q_emb_list[0]
            retrieved = store.query(q_emb, k=top_k)
            st.caption(f"🔎 Retrieved {len(retrieved)} chunk(s) from your notes.")

        else:
            retrieved = []

        context_text = ""
        if retrieved:
            context_text += "Relevant notes:\n"
            for meta, dist in retrieved:
                context_text += (
                    f"- Source: {meta.get('source')} | chunk_index: {meta.get('chunk_index')}\n"
                    f"{meta.get('text')}\n\n"
                )

        # 2) Optional web search
        web_summary = ""
        if use_web:
            try:
                web_hits = serpapi_search(user_query, num_results=4)
                if web_hits:
                    web_summary = summarize_web_results(web_hits, call_llm)
            except Exception as e:
                logging.exception("Web search failed: %s", e)
                web_summary = ""

        # 3) Response mode instructions
        mode_instructions = {
            "Concise": (
                "Give a short, 2–3 sentence answer in simple language. "
                "Focus only on the main idea needed by a student."
            ),
            "Detailed": (
                "Give an in-depth explanation with examples, and end with 3 bullet-point revision notes. "
                "Use simple, student-friendly language."
            ),
        }

        prompt = f"""
You are a Study Assistant for students. You help summarize notes, extract definitions, explain concepts,
rewrite in simpler terms, and help the student revise.

Use the following retrieved notes (document context) and optional web search summary to answer the student's query.
If the context is empty, fall back to general educational knowledge.

Student question:
{user_query}

Response style:
{mode_instructions[response_mode]}

Document context:
{context_text}

Web search summary:
{web_summary}

Now provide the answer in the requested style.
""".strip()

        answer = call_llm(prompt, max_tokens=600, temperature=temperature)
        st.session_state.history.append({"role": "assistant", "content": answer})

    st.markdown("### 💬 Conversation")
    if not st.session_state.history:
        st.info("Start by uploading notes and asking a question, or just ask a general study question.")
    else:
        for turn in st.session_state.history:
            if turn["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(turn["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(turn["content"])
