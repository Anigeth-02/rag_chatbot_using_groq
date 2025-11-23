# config/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ==== LLM (Groq) ====
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

# Check Groq dashboard for exact model name; common ones:
# - "llama-3.1-70b-versatile"
# - "llama-3.1-8b-instant"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
# ==== Embeddings (local sentence-transformers) ====
# We'll use a small, fast model:
# - "all-MiniLM-L6-v2" → 384-dim embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ==== Vector store & RAG ====
VECTOR_DIR = os.getenv("VECTOR_DIR", "vectors")
TOP_K = int(os.getenv("TOP_K", 4))

# ==== Web search (optional; you can keep SerpAPI or remove) ====
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "YOUR_SERPAPI_KEY")
