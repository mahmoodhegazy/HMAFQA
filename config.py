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
        # Model client settings
        model_provider: str = "openai",  # "openai" or "azure" 
        llm_model: str = "gpt-4",
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        certificate_path: Optional[str] = None,
        client_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        scope: str = "https://cognitiveservices.azure.com/.default",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        proxy_config: Optional[Dict[str, str]] = None,
        # Other settings
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
            model_provider: Provider for LLM access ("openai" or "azure")
            llm_model: Name of the LLM model
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: API version for Azure
            certificate_path: Path to certificate for Azure auth
            client_id: Client ID for Azure auth
            tenant_id: Tenant ID for Azure auth
            scope: OAuth scope for Azure auth
            max_retries: Maximum number of API call retries
            retry_delay: Delay between retries
            proxy_config: Proxy configuration settings
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
        
        # Model client settings
        self.model_provider = model_provider
        self.llm_model = llm_model
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.certificate_path = certificate_path
        self.client_id = client_id
        self.tenant_id = tenant_id
        self.scope = scope
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.proxy_config = proxy_config
        
        # Other settings
        self.top_k_docs = top_k_docs
        self.top_k_tables = top_k_tables
        self.max_hops = max_hops
        
        # Set API key if provided
        self.api_key = api_key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        # Otherwise, use the one from environment
        elif "OPENAI_API_KEY" not in os.environ and model_provider == "openai":
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