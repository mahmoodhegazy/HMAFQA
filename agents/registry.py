# hmafqa/agents/registry.py
from typing import Dict, Type, Any
from .base_agent import BaseAgent

class AgentRegistry:
    """
    Registry for agent classes to enable dynamic agent creation.
    """
    
    _registry: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, agent_type: str, agent_class: Type[BaseAgent]):
        """
        Register an agent class.
        
        Args:
            agent_type: The type name for the agent
            agent_class: The agent class
        """
        cls._registry[agent_type] = agent_class
    
    @classmethod
    def get(cls, agent_type: str) -> Type[BaseAgent]:
        """
        Get an agent class by type.
        
        Args:
            agent_type: The type name for the agent
            
        Returns:
            The agent class
        """
        if agent_type not in cls._registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return cls._registry[agent_type]
    
    @classmethod
    def create(cls, agent_type: str, **kwargs) -> BaseAgent:
        """
        Create an agent instance.
        
        Args:
            agent_type: The type name for the agent
            **kwargs: Arguments to pass to the agent constructor
            
        Returns:
            An agent instance
        """
        agent_class = cls.get(agent_type)
        return agent_class(**kwargs)
    
    @classmethod
    def list_agents(cls) -> Dict[str, Type[BaseAgent]]:
        """
        List all registered agent types.
        
        Returns:
            Dictionary of agent types to agent classes
        """
        return cls._registry.copy()