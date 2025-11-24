import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config_loader import load_config

class ModelLoader:
    """
    A utility class to load embedding models and LLM models.
    """
    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config=load_config()

    def _validate_env(self):
        """
        Validate necessary environment variables.
        """
        # We need GROQ_API_KEY for the LLM and HF_TOKEN for HuggingFace embeddings
        required_vars = ["GROQ_API_KEY", "HF_TOKEN"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

    def load_embeddings(self):
        """
        Load and return the embedding model based on provider in config.
        """
        print("Loading Embedding model")
        provider = self.config["embedding_model"].get("provider", "huggingface")
        model_name = self.config["embedding_model"]["model_name"]
        if provider == "huggingface":
            hf_token = os.getenv("HF_TOKEN")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu', 'token': hf_token},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
            return SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def load_llm(self, provider="groq"):
        """
        Load and return the LLM model.

        Args:
            provider (str): The provider to use. Options: "google", "groq"
        """
        print(f"LLM loading using {provider} provider...")

        if provider == "google":
            model_name=self.config["llm"]["google"]["model_name"]
            # Configure the model with proper parameters
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,  # Lower temperature for more factual responses
                top_p=0.95,
                top_k=40,
                convert_system_message_to_human=True  # This handles system messages properly
            )
        elif provider == "groq":
            model_name=self.config["llm"]["groq"]["model_name"]
            # Configure the Groq model
            model = ChatGroq(
                model=model_name,
                temperature=0.2,
                top_p=0.95,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return model