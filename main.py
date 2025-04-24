# hmafqa/main.py
import logging
import os
import json
import re
from typing import Dict, Any, List, Optional

from .agents import (
    BaseAgent, 
    FAQEmbeddingAgent, FAQPromptAgent, FAQAnswerContextAgent,
    ExtractiveQAAgent, GenerativeQAAgent, CalculatorAgent, 
    TableQAAgent, MultiHopAgent, ExpertQAAgent
)
from .judge.context_judge import ContextJudge
from .retrieval.document_retriever import DocumentRetriever
from .retrieval.table_retriever import TableRetriever
from .config import Settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HMAFQA:
    """
    Main orchestrator for the Hybrid Multi-Agent Framework for Financial QA.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the HMAFQA framework.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or Settings()
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components of the framework."""
        # Initialize retrieval components
        self.document_retriever = DocumentRetriever(
            index_path=self.settings.document_index_path,
            embedding_model=self.settings.embedding_model
        )
        
        self.table_retriever = TableRetriever(
            index_path=self.settings.table_index_path,
            embedding_model=self.settings.embedding_model
        )
        
        # Initialize FAQ index
        self.faq_index = self._load_faq_index(self.settings.faq_index_path)
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {}
        
        # Original FAQ agents
        self.agents["FAQ_Embedding"] = FAQEmbeddingAgent(
            faq_index=self.faq_index,
            embedding_model=self.settings.embedding_model
        )
        
        self.agents["FAQ_Prompt"] = FAQPromptAgent(
            faq_index=self.faq_index,
            model=self.settings.llm_model
        )
        
        self.agents["FAQ_AnswerContext"] = FAQAnswerContextAgent(
            faq_index=self.faq_index,
            model=self.settings.llm_model
        )
        
        # New QA agents
        self.agents["Extractive_QA"] = ExtractiveQAAgent(
            document_retriever=self.document_retriever,
            model_name=self.settings.extractive_model,
            top_k_docs=self.settings.top_k_docs
        )
        
        self.agents["Generative_QA"] = GenerativeQAAgent(
            document_retriever=self.document_retriever,
            model=self.settings.llm_model,
            top_k_docs=self.settings.top_k_docs
        )
        
        self.agents["Calculator"] = CalculatorAgent(
            document_retriever=self.document_retriever,
            model=self.settings.llm_model,
            top_k_docs=self.settings.top_k_docs
        )
        
        self.agents["Table_QA"] = TableQAAgent(
            table_retriever=self.table_retriever,
            model=self.settings.llm_model,
            top_k_tables=self.settings.top_k_tables
        )
        
        # Expert agent
        if os.path.exists(self.settings.expert_model_path):
            self.agents["Expert_QA"] = ExpertQAAgent(
                document_retriever=self.document_retriever,
                model_path=self.settings.expert_model_path,
                top_k_docs=self.settings.top_k_docs
            )
        
        # Multi-hop agent (added last since it needs references to other agents)
        self.agents["MultiHop"] = MultiHopAgent(
            document_retriever=self.document_retriever,
            agents=self.agents,
            model=self.settings.llm_model,
            max_hops=self.settings.max_hops
        )
        
        # Initialize judge
        self.judge = ContextJudge(model=self.settings.llm_model)
        
        logger.info(f"Initialized HMAFQA with {len(self.agents)} agents")
    
    def _load_faq_index(self, index_path: str) -> Dict[str, Dict[str, str]]:
        """
        Load FAQ index from file.
        
        Args:
            index_path: Path to the FAQ index file
            
        Returns:
            Dictionary mapping FAQ IDs to FAQ data
        """
        try:
            if os.path.exists(index_path):
                with open(index_path, 'r', encoding='utf-8') as f:
                    faq_index = json.load(f)
                logger.info(f"Loaded {len(faq_index)} FAQs from {index_path}")
                return faq_index
            else:
                logger.warning(f"FAQ index file {index_path} not found, using empty index")
                return {}
        except Exception as e:
            logger.error(f"Error loading FAQ index: {e}")
            return {}
    
    def _select_applicable_agents(self, question: str) -> Dict[str, BaseAgent]:
        """
        Determine which agents are applicable to the given question.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary of applicable agents
        """
        applicable_agents = {}
        
        for agent_name, agent in self.agents.items():
            try:
                if agent.is_applicable(question):
                    applicable_agents[agent_name] = agent
                    logger.info(f"Agent {agent_name} is applicable to the question")
                else:
                    logger.info(f"Agent {agent_name} is not applicable to the question")
            except Exception as e:
                logger.error(f"Error checking if agent {agent_name} is applicable: {e}")
        
        if not applicable_agents:
            # If no agent is applicable, fallback to using all agents
            logger.warning("No agent is applicable to the question, using all agents as fallback")
            applicable_agents = self.agents
        
        return applicable_agents
    
    def answer_question(self, question: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a question using the multi-agent framework.
        
        Args:
            question: The user's question
            conversation_id: Optional ID for tracking conversation context
            
        Returns:
            Dictionary with answer, explanation, and source
        """
        logger.info(f"Processing question: {question}")
        
        # 1. Determine which agents to use based on question type
        applicable_agents = self._select_applicable_agents(question)
        
        # 2. Get answers from all applicable agents
        candidate_answers = []
        
        for agent_name, agent in applicable_agents.items():
            try:
                logger.info(f"Getting answer from {agent_name}")
                result = agent.answer_question(
                    question=question,
                    conversation_id=conversation_id
                )
                
                if result and result.get("answer"):
                    # Add agent name to the result
                    result["agent"] = agent_name
                    candidate_answers.append(result)
                    logger.info(f"Agent {agent_name} returned an answer with confidence {result.get('confidence', 0)}")
                else:
                    logger.info(f"Agent {agent_name} did not return a valid answer")
            except Exception as e:
                logger.error(f"Error from agent {agent_name}: {e}")
        
        # 3. Use the judge to select or synthesize the final answer
        if not candidate_answers:
            # If no agent provided a valid answer, return a default response
            logger.warning("No agent provided a valid answer")
            return {
                "answer": "I don't have enough information to answer this question.",
                "explanation": "None of the agents could provide a valid answer.",
                "source": "None",
                "agent_used": None
            }
        
        final_result = self.judge.evaluate(question, candidate_answers)
        
        logger.info(f"Final answer selected from {final_result.get('agent_used', 'unknown')}")
        
        return final_result