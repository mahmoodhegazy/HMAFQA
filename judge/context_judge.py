# hmafqa/judge/context_judge.py
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple

import openai

logger = logging.getLogger(__name__)

class ContextJudge:
    """
    Context-aware judge that evaluates agent outputs and selects the best answer.
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the Context Judge.
        
        Args:
            model: The OpenAI model to use
        """
        self.model = model
    
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
            # Create prompt with all candidates
            prompt = self._create_judge_prompt(question, candidates)
            
            # Get judge decision
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            
            content = response.choices[0].message.content
            
            # Parse judge decision
            try:
                # Extract JSON decision if available
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group(1))
                    
                    # Check if we're selecting an existing answer or synthesizing
                    if decision.get("choice") == "synthesize":
                        # Judge created a new answer
                        return {
                            "answer": decision.get("final_answer", ""),
                            "explanation": decision.get("reasoning", ""),
                            "source": "Synthesized from multiple agents: " + ", ".join([c.get("agent", "Unknown") for c in candidates]),
                            "agent_used": "Judge (synthesized)"
                        }
                    else:
                        # Judge selected an existing answer
                        choice_idx = int(decision.get("choice", 0))
                        if 0 <= choice_idx < len(candidates):
                            chosen = candidates[choice_idx]
                            return {
                                "answer": chosen["answer"],
                                "explanation": decision.get("reasoning", ""),
                                "source": chosen.get("evidence", ""),
                                "agent_used": chosen.get("agent", "Unknown")
                            }
                
                # Fallback parsing if JSON extraction fails
                # Look for a direct choice indication
                choice_match = re.search(r'choice:?\s*(\d+)', content, re.IGNORECASE)
                if choice_match:
                    choice_idx = int(choice_match.group(1))
                    if 0 <= choice_idx < len(candidates):
                        chosen = candidates[choice_idx]
                        return {
                            "answer": chosen["answer"],
                            "explanation": content,
                            "source": chosen.get("evidence", ""),
                            "agent_used": chosen.get("agent", "Unknown")
                        }
                
                # If we couldn't find a choice, extract the answer directly
                answer_match = re.search(r'(?i)final answer:?\s*(.+?)(?=\n\n|\nreasoning|\n[0-9]|\n[A-Z]|$)', content, re.DOTALL)
                if answer_match:
                    final_answer = answer_match.group(1).strip()
                    return {
                        "answer": final_answer,
                        "explanation": content,
                        "source": "Judge synthesis based on multiple agents",
                        "agent_used": "Judge (direct extraction)"
                    }
                
                # Last resort: treat the whole response as the answer
                return {
                    "answer": content,
                    "explanation": "Direct judge output.",
                    "source": "Judge synthesis",
                    "agent_used": "Judge (full response)"
                }
                
            except Exception as e:
                logger.error(f"Error parsing judge decision: {e}")
                # Use highest confidence answer as fallback
                highest_conf = max(candidates, key=lambda x: x.get("confidence", 0))
                return {
                    "answer": highest_conf["answer"],
                    "explanation": f"Parsing error: {str(e)}. Using highest confidence answer.",
                    "source": highest_conf.get("evidence", ""),
                    "agent_used": highest_conf.get("agent", "Unknown") + " (fallback)"
                }
                
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