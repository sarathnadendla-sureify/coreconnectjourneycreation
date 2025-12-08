import os
import tempfile
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
from utils.csv_processor import process_csv_for_vector_db, examine_csv_file
from pinecone import ServerlessSpec, Pinecone
from uuid import uuid4
import sys
from exception.exceptions import AlayticsBotException

class DataIngestion:
    """
    Class to handle document loading, transformation and ingestion into Pinecone vector store.
    """

    def __init__(self):
        try:
            print("Initializing DataIngestion pipeline...")
            self.model_loader = ModelLoader()
            self._load_env_variables()
            self.config = load_config()
        except Exception as e:
            raise AlayticsBotException(e, sys)

    def _load_env_variables(self):
        try:
            load_dotenv()

            required_vars = [
                "PINECONE_API_KEY",
                "GROQ_API_KEY",
                "HF_TOKEN"
            ]

            missing_vars = [var for var in required_vars if os.getenv(var) is None]
            if missing_vars:
                raise EnvironmentError(f"Missing environment variables: {missing_vars}")

            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            self.hf_token = os.getenv("HF_TOKEN")
        except Exception as e:
            raise AlayticsBotException(e, sys)

    def load_documents(self, uploaded_files) -> List[Document]:
        try:
            documents = []
            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
                suffix = file_ext if file_ext in [".pdf", ".docx", ".csv", ".txt",".ts",".tsx"] else ".tmp"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.file.read())
                    temp_path = temp_file.name

                if file_ext == ".pdf":
                    loader = PyPDFLoader(temp_path)
                    documents.extend(loader.load())
                elif file_ext == ".docx":
                    loader = Docx2txtLoader(temp_path)
                    documents.extend(loader.load())
                elif file_ext == ".csv":
                    # First, examine the CSV file to help with debugging
                    print(f"Examining CSV file: {uploaded_file.filename}")
                    examine_csv_file(temp_path)

                    # Use our custom CSV processor for better handling of structured data
                    print(f"Processing CSV file with custom processor: {uploaded_file.filename}")
                    csv_docs = process_csv_for_vector_db(temp_path)

                    # Add additional processing for better retrieval
                    for doc in csv_docs:
                        # Add filename to metadata
                        doc.metadata["source"] = uploaded_file.filename

                        # Add a special section for common queries about user IDs and event types
                        if "userid" in doc.metadata and "eventtype" in doc.metadata:
                            doc.page_content += f"\n\nThis record contains user ID {doc.metadata['userid']} with event type {doc.metadata['eventtype']}."

                    print(f"Processed {len(csv_docs)} rows from CSV file")
                    documents.extend(csv_docs)
                elif file_ext == ".txt":
                    # Simple text loader for .txt files
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    doc = Document(page_content=text_content, metadata={"source": uploaded_file.filename})
                    documents.append(doc)
                elif file_ext == ".ts":
                    # Simple text loader for .ts files
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    doc = Document(page_content=text_content, metadata={"source": uploaded_file.filename})
                    documents.append(doc)
                elif file_ext == ".tsx":
                    # Simple text loader for .tsx files
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    doc = Document(page_content=text_content, metadata={"source": uploaded_file.filename})
                    documents.append(doc)
                else:
                    print(f"Unsupported file type: {uploaded_file.filename}")
            return documents
        except Exception as e:
            raise AlayticsBotException(e, sys)

    def store_in_vector_db(self, documents: List[Document]):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Increased chunk size to keep more context together
                chunk_overlap=200,
                length_function=len
            )
            documents = text_splitter.split_documents(documents)

            pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            index_name = self.config["vector_db"]["index_name"]

            if index_name not in [i.name for i in pinecone_client.list_indexes()]:
                pinecone_client.create_index(
                    name=index_name,
                    dimension=384,  # adjust if needed based on embedding model
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            index = pinecone_client.Index(index_name)
            vector_store = PineconeVectorStore(index=index, embedding=self.model_loader.load_embeddings())

            # Process documents in smaller batches to avoid size limits
            batch_size = 20  # Smaller batch size to avoid message size limits
            total_docs = len(documents)

            print(f"Processing {total_docs} documents in batches of {batch_size}")

            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch = documents[i:batch_end]
                batch_uuids = [str(uuid4()) for _ in range(len(batch))]

                print(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: documents {i+1}-{batch_end}")

                try:
                    vector_store.add_documents(documents=batch, ids=batch_uuids)
                    print(f"Successfully added batch {i//batch_size + 1}")
                except Exception as batch_error:
                    print(f"Error processing batch {i//batch_size + 1}: {str(batch_error)}")

                    # If the batch is still too large, try with an even smaller batch
                    if "message length too large" in str(batch_error):
                        smaller_batch_size = max(5, batch_size // 4)  # Even smaller batch size
                        print(f"Trying with smaller batch size: {smaller_batch_size}")

                        for j in range(i, batch_end, smaller_batch_size):
                            smaller_batch_end = min(j + smaller_batch_size, batch_end)
                            smaller_batch = documents[j:smaller_batch_end]
                            smaller_batch_uuids = [str(uuid4()) for _ in range(len(smaller_batch))]

                            try:
                                vector_store.add_documents(documents=smaller_batch, ids=smaller_batch_uuids)
                                print(f"Successfully added smaller batch {j}-{smaller_batch_end}")
                            except Exception as smaller_batch_error:
                                print(f"Error processing smaller batch {j}-{smaller_batch_end}: {str(smaller_batch_error)}")

                                # If even the smaller batch fails, try one by one
                                if "message length too large" in str(smaller_batch_error):
                                    print("Batch still too large, processing documents individually")
                                    for k, doc in enumerate(smaller_batch):
                                        try:
                                            # Try to reduce document size if it's too large
                                            if len(doc.page_content) > 2000:
                                                doc.page_content = doc.page_content[:2000] + "... (content truncated)"

                                            single_uuid = str(uuid4())
                                            vector_store.add_documents(documents=[doc], ids=[single_uuid])
                                            print(f"Added document {j+k+1}/{batch_end}")
                                        except Exception as single_doc_error:
                                            print(f"Error with document {j+k+1}: {str(single_doc_error)}")
                                            # Skip this document and continue
                                            continue

            print("Completed processing all document batches")

        except Exception as e:
            raise AlayticsBotException(e, sys)

    def run_pipeline(self, uploaded_files):
        try:
            documents = self.load_documents(uploaded_files)
            if not documents:
                print("No valid documents found.")
                return
            self.store_in_vector_db(documents)
        except Exception as e:
            raise AlayticsBotException(e, sys)

if __name__ == '__main__':
    pass
