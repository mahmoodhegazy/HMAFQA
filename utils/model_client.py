# hmafqa/utils/model_client.py
import os
import logging
import time
from typing import Dict, Any, Optional, Union, List

import openai
from openai import OpenAI
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

class ModelClient:
    """
    Centralized client for accessing AI models (OpenAI, Azure OpenAI).
    Provides robust error handling, retries, and flexible authentication.
    """
    
    def __init__(
        self,
        provider: str = "openai",  # "openai" or "azure"
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        azure_endpoint: Optional[str] = None,
        certificate_path: Optional[str] = None,
        client_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        scope: str = "https://cognitiveservices.azure.com/.default",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        proxy_config: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the model client.
        
        Args:
            provider: The provider to use ("openai" or "azure")
            model: The model name to use
            api_key: API key for authentication (for OpenAI)
            api_version: API version (for Azure)
            azure_endpoint: Azure endpoint URL (for Azure)
            certificate_path: Path to the client certificate (for Azure)
            client_id: Client ID (for Azure)
            tenant_id: Tenant ID (for Azure)
            scope: OAuth scope (for Azure)
            max_retries: Maximum number of retries on error
            retry_delay: Delay between retries (seconds)
            proxy_config: Proxy configuration
        """
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Apply proxy configuration if provided
        if proxy_config:
            for key, value in proxy_config.items():
                os.environ[key] = value
        
        # Initialize the appropriate client
        if provider == "azure":
            if not all([azure_endpoint, api_version]):
                raise ValueError("Azure endpoint and API version are required for Azure provider")
            
            # For certificate-based auth
            if certificate_path and client_id and tenant_id:
                try:
                    from azure.identity import CertificateCredential
                    self.credential = CertificateCredential(
                        client_id=client_id,
                        certificate_path=certificate_path,
                        tenant_id=tenant_id,
                        scope=scope
                    )
                    self.access_token = self.credential.get_token(scope).token
                    self.client = AzureOpenAI(
                        api_key=self.access_token,
                        api_version=api_version,
                        azure_endpoint=azure_endpoint
                    )
                except ImportError:
                    logger.error("azure-identity package is required for certificate-based auth")
                    raise
            # For API key-based auth
            elif api_key:
                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint
                )
            else:
                raise ValueError("Either API key or certificate auth details are required for Azure")
            
        elif provider == "openai":
            if not api_key and "OPENAI_API_KEY" not in os.environ:
                raise ValueError("API key is required for OpenAI provider")
            
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized model client with provider={provider}, model={model}")
    
    def _refresh_token(self):
        """Refresh the access token for Azure certificate-based auth."""
        if self.provider == "azure" and hasattr(self, 'credential'):
            scope = "https://cognitiveservices.azure.com/.default"
            self.access_token = self.credential.get_token(scope).token
            # Update the client with the new token
            self.client.api_key = self.access_token
    
    def create_completion(
        self,
        system_message: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        other_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a completion using the configured model.
        
        Args:
            system_message: The system message to use
            user_message: The user message to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            other_params: Additional parameters to pass to the API
            
        Returns:
            The generated text
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return self.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            other_params=other_params
        )
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        other_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a chat completion using the configured model.
        
        Args:
            messages: The messages to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            other_params: Additional parameters to pass to the API
            
        Returns:
            The generated text
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add any additional parameters
        if other_params:
            params.update(other_params)
        
        for attempt in range(self.max_retries):
            try:
                # Make the API call
                response = self.client.chat.completions.create(**params)
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Error calling {self.provider} API (attempt {attempt+1}/{self.max_retries}): {e}")
                
                # Handle different error types
                if "rate limit" in str(e).lower():
                    # Exponential backoff for rate limits
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Rate limit hit, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                elif "token" in str(e).lower() and self.provider == "azure" and hasattr(self, 'credential'):
                    # Token expired, refresh it
                    logger.info("Token may have expired, refreshing")
                    self._refresh_token()
                elif attempt < self.max_retries - 1:
                    # Other errors, simple retry with delay
                    time.sleep(self.retry_delay)
                else:
                    # Last attempt failed
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    raise
        
        # If we get here, all retries failed
        raise Exception(f"Failed to get completion after {self.max_retries} attempts")
    
    def get_embedding(
        self, 
        text: str, 
        model: Optional[str] = None
    ) -> List[float]:
        """
        Get an embedding for the given text.
        
        Args:
            text: The text to embed
            model: Optional override for the embedding model
            
        Returns:
            The embedding vector
        """
        embedding_model = model or "text-embedding-ada-002"
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=embedding_model,
                    input=text
                )
                return response.data[0].embedding
                
            except Exception as e:
                logger.warning(f"Error getting embedding (attempt {attempt+1}/{self.max_retries}): {e}")
                
                # Handle different error types
                if "rate limit" in str(e).lower():
                    # Exponential backoff for rate limits
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Rate limit hit, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                elif "token" in str(e).lower() and self.provider == "azure" and hasattr(self, 'credential'):
                    # Token expired, refresh it
                    logger.info("Token may have expired, refreshing")
                    self._refresh_token()
                elif attempt < self.max_retries - 1:
                    # Other errors, simple retry with delay
                    time.sleep(self.retry_delay)
                else:
                    # Last attempt failed
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    raise
        
        # If we get here, all retries failed
        raise Exception(f"Failed to get embedding after {self.max_retries} attempts")