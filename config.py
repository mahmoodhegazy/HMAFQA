# hmafqa/config.py
import os
from typing import Dict, Any, Optional

class Settings:
    """
    Configuration settings for the HMAFQA framework.
    """
    
    def __init__(
        self,
        document_index_path: str = "data/document_index",
        table_index_path: str = "data/table_index",
        faq_index_path: str = "data/faq_index.json",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        extractive_model: str = "deepset/roberta-base-squad2",
        expert_model_path: str = "models/expert_qa",
        llm_model: str = "gpt-4",
        top_k_docs: int = 5,
        top_k_tables: int = 3,
        max_hops: int = 3,
        api_key: Optional[str] = None
    ):
        """
        Initialize settings with default values.
        
        Args:
            document_index_path: Path to the document index
            table_index_path: Path to the table index
            faq_index_path: Path to the FAQ index
            embedding_model: Name of the embedding model
            extractive_model: Name of the extractive QA model
            expert_model_path: Path to the expert QA model
            llm_model: Name of the LLM model
            top_k_docs: Number of documents to retrieve
            top_k_tables: Number of tables to retrieve
            max_hops: Maximum number of reasoning hops
            api_key: OpenAI API key
        """
        self.document_index_path = document_index_path
        self.table_index_path = table_index_path
        self.faq_index_path = faq_index_path
        self.embedding_model = embedding_model
        self.extractive_model = extractive_model
        self.expert_model_path = expert_model_path
        self.llm_model = llm_model
        self.top_k_docs = top_k_docs
        self.top_k_tables = top_k_tables
        self.max_hops = max_hops
        
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        # Otherwise, use the one from environment
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Settings':
        """
        Create settings from a dictionary.
        
        Args:
            config_dict: Dictionary of configuration values
            
        Returns:
            Settings instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Settings':
        """
        Create settings from a JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            Settings instance
        """
        import json
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)