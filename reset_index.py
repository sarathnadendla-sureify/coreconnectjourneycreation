import os
from dotenv import load_dotenv
from pinecone import Pinecone
from utils.config_loader import load_config

# Load environment variables and config
load_dotenv()
config = load_config()

# Get Pinecone API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    print("Error: PINECONE_API_KEY not found in environment variables")
    exit(1)

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Get index name from config
index_name = config["vector_db"]["index_name"]

# Check if index exists
if pc.has_index(index_name):
    print(f"Deleting index: {index_name}")
    pc.delete_index(index_name)
    print(f"Index {index_name} deleted successfully")
else:
    print(f"Index {index_name} does not exist")

print("Done. You can now re-upload your CSV file to create a new index with improved processing.")
