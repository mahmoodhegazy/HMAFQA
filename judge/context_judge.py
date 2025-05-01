import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple

from ..utils.model_client import ModelClient

logger = logging.getLogger(__name__)

class ContextJudge:
    """
    Context-aware judge that evaluates agent outputs and selects the best answer.
    """
    
    def __init__(self, model: str = "gpt-4", model_client: Optional[ModelClient] = None):
        """
        Initialize the Context Judge.
        
        Args:
            model: The model to use
            model_client: Optional ModelClient instance for model access
        """
        self.model = model
        self.model_client = model_client
    
    def evaluate(self, question: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate candidate answers and select the best one or synthesize a new answer.
        
        Args:
            question: The original user question
            candidates: List of candidate answers from agents
            
        Returns:
            Dictionary with final answer, explanation, and source
        """
        # Filter out None answers and those with very low confidence
        valid_candidates = [c for c in candidates if c.get("answer") and c.get("confidence", 0) > 10]
        
        if not valid_candidates:
            return {
                "answer": "I don't have enough information to answer this question.",
                "explanation": "None of the agents could provide a valid answer.",
                "source": "Judge",
                "agent_used": None
            }
        
        # If only one candidate, use it directly
        if len(valid_candidates) == 1:
            c = valid_candidates[0]
            return {
                "answer": c["answer"],
                "explanation": "Only one valid answer was provided.",
                "source": c.get("evidence", ""),
                "agent_used": c.get("agent", "Unknown")
            }
        
        # Use the judge to evaluate candidates
        final_result = self._judge_candidates(question, valid_candidates)
        
        return final_result
    
    def _judge_candidates(self, question: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use the judge LLM to evaluate and select the best answer.
        
        Args:
            question: The original user question
            candidates: List of valid candidate answers
            
        Returns:
            Dictionary with final answer, explanation, and source
        """
        try:
            # Create prompt for the judge
            prompt = self._create_judge_prompt(question, candidates)
            
            # Get judge decision using model client if available
            if self.model_client:
                system_message = self._get_system_prompt()
                response_text = self.model_client.create_completion(
                    system_message=system_message,
                    user_message=prompt,
                    temperature=0.2
                )
            else:
                # Legacy direct API call (for backward compatibility)
                import openai
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                )
                response_text = response.choices[0].message.content
            
            # Parse the response to get the final answer
            return self._parse_judge_response(response_text, candidates)
            
        except Exception as e:
            logger.error(f"Error in judge evaluation: {e}")
            # Use highest confidence answer as fallback
            highest_conf = max(candidates, key=lambda x: x.get("confidence", 0))
            return {
                "answer": highest_conf["answer"],
                "explanation": f"Judge error: {str(e)}. Using highest confidence answer.",
                "source": highest_conf.get("evidence", ""),
                "agent_used": highest_conf.get("agent", "Unknown") + " (fallback)"
            }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the judge."""
        return """
        You are an expert financial QA evaluator. You will be given a question and several proposed answers with their evidence. Your job is to decide the best final answer, ensuring it is:
        
        1. Faithful to the evidence (it should not contain facts that aren't supported by the evidence)
        2. Complete (it should address all parts of the question)
        3. Concise (it should be direct and to the point)
        4. Clear (it should be well-structured and easy to understand)
        
        You can either:
        1. Choose one of the provided answers as is (if it's clearly the best)
        2. Synthesize a new answer that combines the strengths of multiple answers
        
        For each candidate answer, carefully check if the claims made are supported by the provided evidence. Be skeptical of answers with low confidence scores or weak evidence. If multiple answers provide partial information, consider synthesizing a complete answer that includes all relevant parts.
        
        Return your decision in JSON format:
        {
          "choice": "0" or "1" or "2" etc. for selecting an existing answer, or "synthesize" for creating a new one,
          "final_answer": "The synthesized answer if creating a new one",
          "reasoning": "Your step-by-step evaluation of each answer and justification for your choice"
        }
        """
    
    def _create_judge_prompt(self, question: str, candidates: List[Dict[str, Any]]) -> str:
        """
        Create the prompt for the judge LLM.
        
        Args:
            question: The original user question
            candidates: List of candidate answers
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Question: {question}\n\nCandidate Answers:\n"
        
        for i, candidate in enumerate(candidates):
            prompt += f"\n[{i}] Agent: {candidate.get('agent', 'Unknown')}\n"
            prompt += f"Answer: {candidate['answer']}\n"
            prompt += f"Confidence: {candidate.get('confidence', 0)}/100\n"
            
            # Include evidence but truncate if too long
            evidence = candidate.get('evidence', '')
            if len(evidence) > 500:
                evidence = evidence[:500] + "... [truncated]"
            prompt += f"Evidence: {evidence}\n"
        
        prompt += """
        Please evaluate each answer carefully, considering:
        - Faithfulness to the evidence
        - Completeness (addresses all aspects of the question)
        - Clarity and conciseness
        - Confidence score and quality of evidence
        
        You can either choose one of the candidate answers as is (by specifying its number) or synthesize a new answer that combines the strengths of multiple candidates.
        
        Return your decision in JSON format:
        {
          "choice": "0" or "1" or "2" etc. for selecting an existing answer, or "synthesize" for creating a new one,
          "final_answer": "The synthesized answer if creating a new one",
          "reasoning": "Your step-by-step evaluation of each answer and justification for your choice"
        }
        """
        
        return prompt