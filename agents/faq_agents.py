# hmafqa/agents/faq_agents.py
import json
import logging
from typing import Dict, Any, List, Optional

import openai
import numpy as np
from sentence_transformers import SentenceTransformer

from .base_agent import BaseAgent
from ..retrieval.document_retriever import DocumentRetriever

logger = logging.getLogger(__name__)

class FAQEmbeddingAgent(BaseAgent):
    """
    Agent that uses embeddings to find the most relevant FAQ.
    This is a wrapper around the existing embedding-based FAQ mapper.
    """
    
    def __init__(self, faq_index, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the FAQ Embedding Agent.
        
        Args:
            faq_index: Index containing FAQs (questions and answers)
            embedding_model: Model to use for embeddings
        """
        super().__init__(name="FAQ_Embedding")
        self.faq_index = faq_index
        self.model = SentenceTransformer(embedding_model)
        
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embeddings for a text."""
        return self.model.encode(text, show_progress_bar=False)
        
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Find the most relevant FAQ using embedding similarity.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer, confidence, and evidence
        """
        # Compute embedding for the question
        question_embedding = self.compute_embedding(question)
        
        # Find the most similar FAQs
        similarities = []
        for faq_id, faq in self.faq_index.items():
            faq_embedding = self.compute_embedding(faq["question"])
            similarity = np.dot(question_embedding, faq_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(faq_embedding)
            )
            similarities.append((faq_id, similarity, faq))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top FAQ as the answer
        if similarities:
            top_faq = similarities[0]
            faq_id, similarity, faq = top_faq
            
            # Scale similarity to confidence (0-100)
            confidence = int(similarity * 100)
            
            return {
                "answer": faq["answer"],
                "confidence": confidence,
                "evidence": f"FAQ: {faq['question']}",
                "faq_id": faq_id
            }
        
        return {
            "answer": None,
            "confidence": 0,
            "evidence": "No relevant FAQ found."
        }

class FAQPromptAgent(BaseAgent):
    """
    Agent that uses LLM prompting to find the most relevant FAQ.
    This is a wrapper around the existing prompt-based FAQ mapper.
    """
    
    def __init__(self, faq_index, model="gpt-4"):
        """
        Initialize the FAQ Prompt Agent.
        
        Args:
            faq_index: Index containing FAQs (questions and answers)
            model: OpenAI model to use for prompting
        """
        super().__init__(name="FAQ_Prompt")
        self.faq_index = faq_index
        self.model = model
        
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Find the most relevant FAQ using LLM prompting.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer, confidence, and evidence
        """
        # Create a prompt with the question and FAQs
        prompt = self._create_prompt(question)
        
        try:
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": 
                     "You are an expert FAQ mapping system for banking. Your task is to find the most relevant FAQ for a user's question."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            
            # Parse the response to get the ranked FAQs
            content = response.choices[0].message.content
            
            try:
                # Attempt to parse JSON response
                result = json.loads(content)
                relevant_faqs = result.get('relevant_faqs', [])
                
                if relevant_faqs:
                    top_faq = relevant_faqs[0]
                    faq_id = self._find_faq_id_by_title(top_faq['faq'])
                    
                    if faq_id:
                        return {
                            "answer": self.faq_index[faq_id]["answer"],
                            "confidence": top_faq['relevance_score'],
                            "evidence": f"FAQ: {top_faq['faq']}\nReasoning: {top_faq.get('reasoning', '')}",
                            "faq_id": faq_id
                        }
            except json.JSONDecodeError:
                # If JSON parsing fails, attempt to extract info from text
                logger.warning("Failed to parse JSON response from LLM. Attempting to extract info from text.")
                lines = content.strip().split("\n")
                for line in lines:
                    if line.startswith("1."):
                        parts = line.split("-")
                        if len(parts) >= 2:
                            faq_title = parts[0].strip()[3:].strip()
                            faq_id = self._find_faq_id_by_title(faq_title)
                            
                            if faq_id:
                                # Extract confidence score if available
                                confidence = 90  # Default
                                if "Relevance Score:" in parts[1]:
                                    try:
                                        score_text = parts[1].split("Relevance Score:")[1].strip()
                                        confidence = int(score_text)
                                    except:
                                        pass
                                
                                return {
                                    "answer": self.faq_index[faq_id]["answer"],
                                    "confidence": confidence,
                                    "evidence": f"FAQ: {faq_title}",
                                    "faq_id": faq_id
                                }
            
            # If we reach here, no valid FAQ was found
            return {
                "answer": None,
                "confidence": 0,
                "evidence": "No relevant FAQ found or failed to parse LLM response."
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {
                "answer": None,
                "confidence": 0,
                "evidence": f"Error: {str(e)}"
            }
    
    def _create_prompt(self, question: str) -> str:
        """Create a prompt for the LLM with the question and FAQs."""
        prompt = f"User Question: {question}\n\nAvailable FAQs:\n"
        
        # Add FAQs to the prompt
        for i, (faq_id, faq) in enumerate(self.faq_index.items(), 1):
            prompt += f"{i}. {faq['question']}\n"
        
        prompt += "\nPlease rank the top 5 most relevant FAQs for this question. Return your response in JSON format like this:\n"
        prompt += """{
  "intent_analysis": "Analysis of the user's intent",
  "relevant_faqs": [
    {
      "faq": "Exact title of the FAQ from the list",
      "relevance_score": 95,
      "reasoning": "Why this FAQ is relevant"
    },
    ...
  ]
}"""
        
        return prompt
    
    def _find_faq_id_by_title(self, title: str) -> Optional[str]:
        """Find an FAQ ID by its title (question)."""
        for faq_id, faq in self.faq_index.items():
            if faq["question"].lower() == title.lower():
                return faq_id
        return None

class FAQAnswerContextAgent(BaseAgent):
    """
    Agent that considers answer context when finding relevant FAQs.
    This looks for similarity between the question and both FAQ questions and answers.
    """
    
    def __init__(self, faq_index, model="gpt-4"):
        """
        Initialize the FAQ Answer Context Agent.
        
        Args:
            faq_index: Index containing FAQs (questions and answers)
            model: OpenAI model to use for prompting
        """
        super().__init__(name="FAQ_AnswerContext")
        self.faq_index = faq_index
        self.model = model
        
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Find the most relevant FAQ considering both questions and answers.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer, confidence, and evidence
        """
        # Create a prompt that includes both questions and answers
        prompt = self._create_prompt_with_answers(question)
        
        try:
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": 
                     "You are an expert FAQ mapping system that considers both questions and answers when finding relevant FAQs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            
            # Parse the response to get the ranked FAQs
            content = response.choices[0].message.content
            
            try:
                # Attempt to parse JSON response
                result = json.loads(content)
                relevant_faqs = result.get('relevant_faqs', [])
                
                if relevant_faqs:
                    top_faq = relevant_faqs[0]
                    faq_id = self._find_faq_id_by_title(top_faq['faq'])
                    
                    if faq_id:
                        return {
                            "answer": self.faq_index[faq_id]["answer"],
                            "confidence": top_faq['relevance_score'],
                            "evidence": f"FAQ: {top_faq['faq']}\nReasoning: {top_faq.get('reasoning', '')}",
                            "faq_id": faq_id
                        }
            except json.JSONDecodeError:
                # If JSON parsing fails, attempt to extract info from text
                logger.warning("Failed to parse JSON response from LLM. Attempting to extract info from text.")
                lines = content.strip().split("\n")
                for line in lines:
                    if line.startswith("1."):
                        parts = line.split("-")
                        if len(parts) >= 2:
                            faq_title = parts[0].strip()[3:].strip()
                            faq_id = self._find_faq_id_by_title(faq_title)
                            
                            if faq_id:
                                # Extract confidence score if available
                                confidence = 90  # Default
                                if "Relevance Score:" in parts[1]:
                                    try:
                                        score_text = parts[1].split("Relevance Score:")[1].strip()
                                        confidence = int(score_text)
                                    except:
                                        pass
                                
                                return {
                                    "answer": self.faq_index[faq_id]["answer"],
                                    "confidence": confidence,
                                    "evidence": f"FAQ: {faq_title}",
                                    "faq_id": faq_id
                                }
            
            # If we reach here, no valid FAQ was found
            return {
                "answer": None,
                "confidence": 0,
                "evidence": "No relevant FAQ found or failed to parse LLM response."
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {
                "answer": None,
                "confidence": 0,
                "evidence": f"Error: {str(e)}"
            }
    
    def _create_prompt_with_answers(self, question: str) -> str:
        """Create a prompt for the LLM with the question and FAQs including answers."""
        prompt = f"User Question: {question}\n\nAvailable FAQs with their Answers:\n"
        
        # Add FAQs with answers to the prompt
        for i, (faq_id, faq) in enumerate(self.faq_index.items(), 1):
            prompt += f"{i}. Question: {faq['question']}\n   Answer: {faq['answer']}\n\n"
        
        prompt += "\nPlease rank the top 5 most relevant FAQs for this question, considering both the FAQ questions and their answers. Return your response in JSON format like this:\n"
        prompt += """{
  "intent_analysis": "Analysis of the user's intent",
  "relevant_faqs": [
    {
      "faq": "Exact title of the FAQ from the list",
      "relevance_score": 95,
      "reasoning": "Why this FAQ is relevant based on question and answer content"
    },
    ...
  ]
}"""
        
        return prompt
    
    def _find_faq_id_by_title(self, title: str) -> Optional[str]:
        """Find an FAQ ID by its title (question)."""
        for faq_id, faq in self.faq_index.items():
            if faq["question"].lower() == title.lower():
                return faq_id
        return None