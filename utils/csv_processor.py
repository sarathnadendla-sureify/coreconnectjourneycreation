import csv
import io
from typing import List, Dict, Any

# Define a simple Document class if langchain_core is not available
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def process_csv_for_vector_db(file_path: str) -> List[Document]:
    """
    Process a CSV file to create documents optimized for vector database storage and retrieval.

    Args:
        file_path: Path to the CSV file

    Returns:
        List of Document objects ready for vector database ingestion
    """
    documents = []

    # Try different delimiters in order of likelihood
    delimiters = [',', '\t', ';', '|', ' ']

    # First, read the file content
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return documents

    # Try to detect the delimiter using csv.Sniffer
    try:
        dialect = csv.Sniffer().sniff(content[:1024] if len(content) > 1024 else content)
        delimiters.insert(0, dialect.delimiter)  # Add detected delimiter at the beginning
    except:
        print("Could not automatically detect delimiter, will try common delimiters")

    # Try each delimiter until one works
    for delimiter in delimiters:
        try:
            print(f"Trying delimiter: '{delimiter}'")
            csv_file = io.StringIO(content)
            reader = csv.DictReader(csv_file, delimiter=delimiter)

            # Get the fieldnames
            fieldnames = reader.fieldnames if reader.fieldnames else []

            # Check if we have valid fieldnames
            if not fieldnames:
                print(f"No valid fieldnames found with delimiter '{delimiter}'")
                continue

            print(f"Found fieldnames: {fieldnames}")

            # If we got here, we have a valid CSV format
            # Reset the file pointer and process the data
            csv_file.seek(0)
            reader = csv.DictReader(csv_file, delimiter=delimiter)

            # Process each row
            for i, row in enumerate(reader):
                # Create metadata from the row
                metadata = {k: v for k, v in row.items()}
                metadata["row_index"] = i

                # Create a structured content string that's optimized for retrieval
                content_parts = []

                # Add a header that describes the data
                content_parts.append(f"CSV Row {i+1} Data:")

                # Add each field with its name for better context
                for field in fieldnames:
                    if field in row and row[field]:
                        content_parts.append(f"{field}: {row[field]}")

                # Create special sections for common query fields
                if "userid" in [f.lower() for f in fieldnames]:
                    userid_field = next(f for f in fieldnames if f.lower() == "userid")
                    content_parts.append(f"User ID: {row[userid_field]}")

                if "eventtype" in [f.lower() for f in fieldnames]:
                    eventtype_field = next(f for f in fieldnames if f.lower() == "eventtype")
                    content_parts.append(f"Event Type: {row[eventtype_field]}")

                # Join all parts with newlines for better readability
                content = "\n".join(content_parts)

                # Add a summary line that's optimized for common queries
                summary_parts = []
                for field in fieldnames:
                    if field.lower() in ["userid", "user_id", "eventtype", "event_type", "event", "user"]:
                        if field in row and row[field]:
                            summary_parts.append(f"{field}={row[field]}")

                if summary_parts:
                    content += "\n\nSummary: " + ", ".join(summary_parts)

                # Create the document
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            # If we successfully processed the CSV, break out of the loop
            if documents:
                print(f"Successfully processed CSV with delimiter: '{delimiter}'")
                break

        except Exception as e:
            print(f"Error with delimiter '{delimiter}': {e}")
            continue

    # If we tried all delimiters and none worked, print a message
    if not documents:
        print("Could not process CSV file with any of the common delimiters")

    return documents

def examine_csv_file(file_path: str) -> None:
    """
    Examine a CSV file to help with debugging.

    Args:
        file_path: Path to the CSV file
    """
    try:
        # Read the first few lines of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            print("First 5 lines of the file:")
            for i, line in enumerate(file):
                if i < 5:
                    print(f"Line {i+1}: {line.strip()}")
                else:
                    break
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                print("First 5 lines of the file (using latin-1 encoding):")
                for i, line in enumerate(file):
                    if i < 5:
                        print(f"Line {i+1}: {line.strip()}")
                    else:
                        break
        except Exception as e:
            print(f"Error reading file: {e}")
    except Exception as e:
        print(f"Error examining CSV file: {e}")

def extract_structured_data_from_documents(documents: List[Document], query_type: str) -> Dict[str, Any]:
    """
    Extract structured data from documents based on the query type.

    Args:
        documents: List of Document objects
        query_type: Type of query (e.g., "userid_by_eventtype", "eventtype_by_userid")

    Returns:
        Dictionary with structured data
    """
    result = {}

    if query_type == "userid_by_eventtype":
        # Extract all user IDs for a specific event type
        event_type_to_find = query_type.split("_")[-1]
        user_ids = set()

        for doc in documents:
            if "eventtype" in doc.metadata and doc.metadata["eventtype"].lower() == event_type_to_find.lower():
                if "userid" in doc.metadata:
                    user_ids.add(doc.metadata["userid"])

        result["user_ids"] = list(user_ids)

    return result
