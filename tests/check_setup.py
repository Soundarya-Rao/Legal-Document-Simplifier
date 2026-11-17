import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reconfigure stdout to UTF-8 so non-ASCII characters don't crash Windows cp1252 console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))  # reads .env into environment variables
key = os.getenv("GEMINI_API_KEY")
print("Key loaded:", key is not None)

# Initialize the Gemini model via LangChain's Google GenAI integration
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=key,
    max_retries=2,   # Retry up to 2 times to handle temporary network glitches
    timeout=120,     # Allow up to 120 seconds before timing out on slow connections
)
reply = llm.invoke("Say 'setup works'.")
print(reply.content)
