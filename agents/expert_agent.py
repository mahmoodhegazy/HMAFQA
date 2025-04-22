# hmafqa/agents/expert_agent.py
import logging
from typing import Dict, Any, List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base_agent import BaseAgent
from ..retrieval.document_retriever import DocumentRetriever

logger = logging.getLogger(__name__)

class ExpertQAAgent(BaseAgent):
    """
    Agent that uses a fine-tuned model specialized for financial QA.
    """
    
    def __init__(
        self, 
        document_retriever: DocumentRetriever,
        model_path: str = "path/to/finetuned/model",
        top_k_docs: int = 5
    ):
        """
        Initialize the Expert QA Agent.
        
        Args:
            document_retriever: Retriever for finding relevant documents
            model_path: Path to the fine-tuned model
            top_k_docs: Number of documents to retrieve and consider
        """
        super().__init__(name="Expert_QA")
        self.document_retriever = document_retriever
        self.top_k_docs = top_k_docs
        
        # Load model and tokenizer
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading expert model: {e}")
            raise e
    
    def is_applicable(self, question: str) -> bool:
        """
        Check if this agent is applicable to the question.
        
        Args:
            question: The user's question
            
        Returns:
            Boolean indicating if this agent should process the question
        """
        # The expert agent can handle most financial questions
        # Could add specific checks for financial domain questions
        return True
    
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an answer using the fine-tuned expert model.
        
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
        
        # Prepare input for the model
        input_text = self._prepare_input(question, docs)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the output
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence based on output probability
            # This is a simplified implementation - real confidence scoring would depend on model outputs
            logits = outputs.scores[0] if hasattr(outputs, 'scores') and outputs.scores else None
            confidence = self._estimate_confidence(logits) if logits is not None else 75
            
            # Prepare evidence string
            evidence = "Based on specialized financial knowledge from fine-tuned model.\n"
            evidence += "Retrieved from: " + ", ".join([doc["id"] for doc in docs[:3]])
            
            return {
                "answer": answer,
                "confidence": confidence,
                "evidence": evidence
            }
            
        except Exception as e:
            logger.error(f"Error generating expert answer: {e}")
            return {
                "answer": None,
                "confidence": 0,
                "evidence": f"Error: {str(e)}"
            }
    
    def _prepare_input(self, question: str, docs: List[Dict[str, Any]]) -> str:
        """
        Prepare the input for the expert model by combining the question and document context.
        
        Args:
            question: The user's question
            docs: List of retrieved documents
            
        Returns:
            Formatted input string
        """
        # Combine document texts
        context = "\n\n".join([doc["text"] for doc in docs[:3]])  # Limit to first 3 docs to avoid excessive length
        
        # Format input based on the model's training format
        input_text = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
        
        return input_text
    
    def _estimate_confidence(self, logits: torch.Tensor) -> int:
        """
        Estimate confidence score based on output logits.
        
        Args:
            logits: Output logits from the model
            
        Returns:
            Confidence score (0-100)
        """
        # This is a placeholder implementation
        # A real implementation would analyze the token probabilities
        
        # Assuming higher max logits indicate higher confidence
        if logits is None:
            return 75  # Default confidence
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Use the max probability of the most likely token as a confidence indicator
        max_prob = torch.max(probs).item()
        
        # Scale to 0-100
        confidence = int(max_prob * 100)
        
        return confidence