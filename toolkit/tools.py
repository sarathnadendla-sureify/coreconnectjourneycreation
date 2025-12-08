import os
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_community.tools.polygon.financials import PolygonFinancials
from data_models.models import RagToolSchema
from langchain_pinecone import PineconeVectorStore
from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
from dotenv import load_dotenv
from pinecone import Pinecone
model_loader=ModelLoader()
config = load_config()
load_dotenv()

# Use Google embeddings for now, but we'll use GROQ for the LLM

def llm_rerank(question, documents, model_loader):
    """
    Rerank documents using an LLM by scoring each document's relevance to the question.
    Returns a list of documents sorted by relevance (highest first).
    """
    # Use the LLM to score each document
    scored_docs = []
    llm = model_loader.load_llm()
    for doc in documents:
        prompt = f"Question: {question}\nDocument: {doc.page_content}\nHow relevant is this document to the question? Reply with a score from 1 (not relevant) to 10 (highly relevant)."
        try:
            score_str = llm.invoke(prompt)
            # Extract score (assume LLM returns a number or a string containing a number)
            score = None
            for token in str(score_str).split():
                try:
                    score = float(token)
                    break
                except ValueError:
                    continue
            if score is None:
                score = 1  # fallback to lowest relevance
        except Exception:
            score = 1  # fallback on error
        scored_docs.append((score, doc))
    # Sort by score descending
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored_docs]

@tool(args_schema=RagToolSchema)
def retriever_tool(question):
    """Retrieves information from the vector database based on the question.
    Useful for answering questions about data stored in the system, including CSV data with user IDs and event types."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    # Create vector store with the index
    vector_store = PineconeVectorStore(
        index=pc.Index(config["vector_db"]["index_name"]),
        embedding=model_loader.load_embeddings()
    )

    # Increase k to get more results for better coverage
    k = config["retriever"]["top_k"] * 6  # Increased multiplier for more results

    # Lower the threshold to capture more potentially relevant results
    threshold = max(0.01, config["retriever"]["score_threshold"] - 0.4)  # Lowered threshold further

    # Create retriever with adjusted parameters
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": threshold,
        },
    )

    # Get results
    retriever_result = retriever.invoke(question)

    # LLM-based reranking
    reranked_results = llm_rerank(question, retriever_result, model_loader)

    # If the question is about specific fields like userids or eventtypes, add a note
    if any(keyword in question.lower() for keyword in ["userid", "user id", "eventtype", "event type"]):
        note = "\n\nNote: This data comes from the uploaded CSV file. If you need to extract specific user IDs or event types, please analyze the content carefully."
        if reranked_results:
            reranked_results[0].page_content += note

    return reranked_results

# tavilytool = TavilySearchResults(
#     max_results=config["tools"]["tavily"]["max_results"],
#     search_depth="advanced",
#     include_answer=True,
#     include_raw_content=True,
#     )
