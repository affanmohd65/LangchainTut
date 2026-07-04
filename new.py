from dotenv import load_dotenv
import os

load_dotenv()

print("API Key:", os.getenv("GOOGLE_API_KEY"))