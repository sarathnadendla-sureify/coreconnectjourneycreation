import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    print("Successfully imported and initialized Pinecone")
except Exception as e:
    print(f"Error: {e}")
