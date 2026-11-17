import os
import sys
from dotenv import load_dotenv
from google import genai

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list():
    print(m.name)
