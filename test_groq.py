from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

print("Using GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello Groq! Does this work?"}],
)

print(resp.choices[0].message.content)
