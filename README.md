 📚 Study Notes Assistant (RAG + Groq + Streamlit)

An AI-powered **Study Notes Assistant** that helps students revise and understand their own notes.

Upload your lecture notes as text files, and the chatbot will:
- summarize them,
- explain concepts in simple language,
- answer questions based on your notes,
- and optionally fall back to general knowledge.

🟢 Live Demo:  
👉 https://ragchatbotusinggroq-fdyvexhhl9gagmjmnv2zjt.streamlit.app/

## 🎯 Use Case Objective

Many students:
- have lots of scattered notes,
- struggle to revise large content,
- find it hard to quickly recall definitions or key concepts before exams.

This app acts as a **personal study assistant**:
- You upload your notes,
- ask questions in normal language,
- and get answers grounded in the uploaded documents.

---

## ✨ Features

- 📂 **Upload your study notes** (`.txt`, `.md`)  
- 🧠 **RAG (Retrieval-Augmented Generation)**  
  - Notes are chunked and embedded using sentence-transformers  
  - Stored in a FAISS vector index  
  - Relevant chunks are retrieved when you ask a question
- 🤖 **Groq LLM integration**  
  - Uses a Groq-hosted LLaMA model to generate answers
- 🎛 **Response modes**  
  - **Concise** → 2–3 sentence short answers  
  - **Detailed** → deeper explanation + revision bullet points
- 💬 **Chat-style interface**  
  - Conversation history visible as chat bubbles
- 🧹 **Utility buttons**  
  - Clear Knowledge Base (reset all indexed notes)  
  - Clear Chat History

---

## 🧱 Tech Stack

- Frontend / UI: Streamlit  
- **LLM Provider:** Groq  
- **Model:** `llama-3.3-70b-versatile` (configurable)  
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2` by default)  
- **Vector Store:** FAISS (CPU)  
- **Other:** `numpy`, `python-dotenv`, `requests`

---

## 🗂 Project Structure

```text
project_root/
├── app.py                     # Main Streamlit app
├── requirements.txt           # Python dependencies
├── .gitignore
├── config/
│   └── config.py              # API keys, model names, basic config
├── models/
│   ├── llm.py                 # Groq LLM wrapper (call_llm)
│   └── embeddings.py          # Embedding model wrapper
├── utils/
│   ├── rag_utils.py           # Chunking, FAISS index, KB utilities
│   └── web_search.py          # Optional SerpAPI web search integration
└── vectors/                   # FAISS index + metadata (auto-created, gitignored)


1️⃣ Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2️⃣ Create virtual environment
python -m venv chatbotvenv
chatbotvenv\Scripts\activate   # on Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set your environment variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here
# Optional:
# SERPAPI_KEY=your_serpapi_key_here


⚠️ Never commit .env to Git. It should be in .gitignore.

▶️ Running the App Locally
chatbotvenv\Scripts\activate
streamlit run app.py


Then open your browser at:

http://localhost:8501

🧠 How It Works (RAG Flow)

Upload Notes

User uploads .txt / .md files from the sidebar.

Each file is split into overlapping chunks.

Embeddings & Indexing

Chunks are passed through sentence-transformers to get embeddings.

Embeddings + metadata are stored in a FAISS index (vectors/ folder).

Question Answering

When the user asks a question:

The question is embedded.

Top-K similar chunks are retrieved from FAISS.

A prompt is built with:

Student question

Retrieved document context

Optional web search summary

Response mode (Concise / Detailed)

The prompt is sent to Groq LLM (call_llm), and response is shown in the UI.

Modes

Concise → short, focused answer for quick revision.

Detailed → full explanation with revision bullet points.

🌐 Deployment (Streamlit Cloud)

Push your code to GitHub (without chatbotvenv/ or .env).

On Streamlit Cloud
:

Create a new app

Point to your GitHub repo and app.py

Set secrets in App → Settings → Secrets:

GROQ_API_KEY = "your_groq_key_here"
# SERPAPI_KEY = "your_serpapi_key_here"


Deploy ✅
🚧 Limitations & Future Improvements

Currently supports plain text (.txt, .md); PDF/DOCX support can be added via parsers.

No per-user authentication or persistent cloud DB yet.

Web search feature is optional and depends on SerpAPI key.

Possible enhancements:

PDF / DOCX lecture notes support

User login & saved workspaces

“Quiz mode” – auto-generate questions from notes

Export summaries/flashcards

👤 Author / Credits

Built by: Anigeth

Purpose: AI-powered study assistant for notes revision using RAG + Groq + Streamlit.

If you have suggestions or feedback, feel free to open an issue or pull request!
