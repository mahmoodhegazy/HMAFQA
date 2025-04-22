# hmafqa/agents/extractive_agent.py
import logging
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from .base_agent import BaseAgent
from ..retrieval.document_retriever import DocumentRetriever

logger = logging.getLogger(__name__)

class ExtractiveQAAgent(BaseAgent):
    """
    Agent that extracts answer spans from relevant documents.
    """
    
    def __init__(
        self, 
        document_retriever: DocumentRetriever,
        model_name: str = "deepset/roberta-base-squad2",
        top_k_docs: int = 5
    ):
        """
        Initialize the Extractive QA Agent.
        
        Args:
            document_retriever: Retriever for finding relevant documents
            model_name: The name of the QA model to use
            top_k_docs: Number of documents to retrieve and consider
        """
        super().__init__(name="Extractive_QA")
        self.document_retriever = document_retriever
        self.top_k_docs = top_k_docs
        
        # Load QA model and tokenizer
        try:
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.qa_pipeline = pipeline(
                "question-answering", 
                model=self.qa_model, 
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Error loading QA model {model_name}: {e}")
            raise e
    
    def is_applicable(self, question: str) -> bool:
        """Check if this agent is applicable to the question."""
        # Extractive QA is applicable to most factoid questions
        # We could add more sophisticated checks here
        return True
    
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Extract answer spans from retrieved documents.
        
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
        
        # Run QA pipeline on each document and gather all answers
        all_answers = []
        
        for doc in docs:
            try:
                # Get answer from QA model
                result = self.qa_pipeline(
                    question=question, 
                    context=doc["text"],
                    handle_impossible_answer=True
                )
                
                # Store result with document info
                all_answers.append({
                    "answer": result["answer"],
                    "score": float(result["score"]),
                    "doc_id": doc["id"],
                    "doc_text": doc["text"],
                    "start": result.get("start", 0),
                    "end": result.get("end", 0)
                })
            except Exception as e:
                logger.warning(f"Error processing document {doc['id']}: {e}")
        
        # Sort answers by score
        all_answers.sort(key=lambda x: x["score"], reverse=True)
        
        # Return the top answer if we found any
        if all_answers:
            top_answer = all_answers[0]
            
            # Create a context snippet around the answer
            context = self._create_context_snippet(
                top_answer["doc_text"],
                top_answer["start"],
                top_answer["end"]
            )
            
            # Scale the score to a confidence between 0-100
            confidence = int(min(top_answer["score"] * 100, 100))
            
            return {
                "answer": top_answer["answer"],
                "confidence": confidence,
                "evidence": context,
                "doc_id": top_answer["doc_id"]
            }
        
        return {
            "answer": None,
            "confidence": 0,
            "evidence": "No answers found in the retrieved documents."
        }
    
    def _create_context_snippet(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Create a snippet of text around the answer for context."""
        snippet_start = max(0, start - window)
        snippet_end = min(len(text), end + window)
        
        # Get the snippet
        snippet = text[snippet_start:snippet_end]
        
        # Add ellipsis if we truncated
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(text):
            snippet = snippet + "..."
        
        return snippet