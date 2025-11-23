# models/llm.py
"""
LLM wrapper using Groq.
Export: call_llm(prompt: str, max_tokens=512, temperature=0.2)
"""

import logging
from typing import Optional
from config.config import GROQ_API_KEY, LLM_MODEL

try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    client = None
    logging.warning("Groq client not initialized: %s", e)


def call_llm(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    stop: Optional[list] = None
) -> str:
    """
    Call Groq LLM and return text.
    """
    if client is None:
        return "ERROR: GROQ client not initialized. Check API key."

    try:
        resp = client.chat.completions.create(
           model=LLM_MODEL,

            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        logging.exception("Groq LLM call failed: %s", e)
        return "ERROR: LLM call failed."
