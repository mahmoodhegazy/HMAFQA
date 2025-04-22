# hmafqa/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

class BaseAgent(ABC):
    """
    Base class for all agents in the HMAFQA framework.
    
    All specialized agents inherit from this class and implement
    the answer_question method with their specific logic.
    """
    
    def __init__(self, name: str):
        """Initialize the base agent with a name."""
        self.name = name
    
    @abstractmethod
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Process a question and return an answer with supporting information.
        
        Args:
            question: The user's question
            **kwargs: Additional arguments specific to the agent
            
        Returns:
            A dictionary containing:
                - answer: The agent's answer to the question
                - confidence: A score between 0-100 indicating confidence
                - evidence: Supporting evidence or reasoning steps
                - agent: The name of the agent (added by the orchestrator)
        """
        pass
    
    def is_applicable(self, question: str) -> bool:
        """
        Determine if this agent is applicable to the given question.
        Default implementation returns True for all questions.
        
        Args:
            question: The user's question
            
        Returns:
            Boolean indicating if this agent should process the question
        """
        return True