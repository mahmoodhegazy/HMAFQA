import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from ..retrieval.document_retriever import DocumentRetriever
from ..utils.model_client import ModelClient

logger = logging.getLogger(__name__)

class GenerativeQAAgent(BaseAgent):
    """
    Agent that generates answers using an LLM with retrieved context.
    """
    
    def __init__(
        self, 
        document_retriever: DocumentRetriever,
        model: str = "gpt-4",
        top_k_docs: int = 5,
        max_tokens: int = 500,
        model_client: Optional[ModelClient] = None
    ):
        """
        Initialize the Generative QA Agent.
        
        Args:
            document_retriever: Retriever for finding relevant documents
            model: The model to use
            top_k_docs: Number of documents to retrieve and consider
            max_tokens: Maximum number of tokens in the generated answer
            model_client: Optional ModelClient instance for model access
        """
        super().__init__(name="Generative_QA")
        self.document_retriever = document_retriever
        self.model = model
        self.top_k_docs = top_k_docs
        self.max_tokens = max_tokens
        self.model_client = model_client
    
    def is_applicable(self, question: str) -> bool:
        """Check if this agent is applicable to the question."""
        # The generative agent is generally applicable to all questions
        return True
    
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an answer using an LLM with retrieved context.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer, confidence, and evidence
        """
        # Retrieve relevant documents
        docs = self.document_retriever.retrieve(question, top_k=self.top_k_docs)
        
        if not docs:
            return {
                "answer": None,
                "confidence": 0,
                "evidence": "No relevant documents found."
            }
        
        # Prepare prompt with context from documents
        prompt = self._create_prompt(question, docs)
        
        try:
            # Use model client if available, otherwise fall back to direct API call
            if self.model_client:
                system_message = "You are an expert financial advisor assistant. Answer questions based only on the provided context. If the answer is not in the context, say 'I don't have enough information to answer this question.'"
                answer = self.model_client.create_completion(
                    system_message=system_message,
                    user_message=prompt,
                    temperature=0.3,
                    max_tokens=self.max_tokens
                )
            else:
                # Legacy direct API call (for backward compatibility)
                import openai
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": 
                         "You are an expert financial advisor assistant. Answer questions based only on the provided context. If the answer is not in the context, say 'I don't have enough information to answer this question.'"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                )
                answer = response.choices[0].message.content.strip()
            
            # Use token probability as a proxy for confidence
            # This is a simplified approach; a more sophisticated confidence estimation could be implemented
            confidence = 85  # Default confidence
            
            # In a real implementation, would use the token probabilities or logprobs
            # from the OpenAI response if available
            
            # Prepare evidence string
            evidence = "Based on: " + ", ".join([f"Document {i+1}" for i in range(len(docs))])
            
            # Include document IDs for traceability
            doc_ids = [doc["id"] for doc in docs]
            
            return {
                "answer": answer,
                "confidence": confidence,
                "evidence": evidence,
                "doc_ids": doc_ids
            }
            
        except Exception as e:
            logger.error(f"Error calling model API: {e}")
            return {
                "answer": None,
                "confidence": 0,
                "evidence": f"Error: {str(e)}"
            }
    
    def _create_prompt(self, question: str, docs: List[Dict[str, Any]]) -> str:
        """Create a prompt for the LLM with the question and document context."""
        prompt = f"Question: {question}\n\nContext:\n"
        
        # Add document context
        for i, doc in enumerate(docs, 1):
            prompt += f"Document {i}:\n{doc['text']}\n\n"
        
        prompt += "\nBased strictly on the above context, answer the question. If the context doesn't contain the answer, say 'I don't have enough information to answer this question.' Be concrete and precise. Cite specific documents when possible."
        
        return prompt