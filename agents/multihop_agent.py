# hmafqa/agents/multihop_agent.py
import logging
from typing import Dict, Any, List, Optional, Tuple

import openai

from .base_agent import BaseAgent
from ..retrieval.document_retriever import DocumentRetriever

logger = logging.getLogger(__name__)

class MultiHopAgent(BaseAgent):
    """
    Agent that handles questions requiring multi-step reasoning across multiple pieces of information.
    """
    
    def __init__(
        self, 
        document_retriever: DocumentRetriever,
        agents: Dict[str, BaseAgent] = None,
        model: str = "gpt-4",
        max_hops: int = 3
    ):
        """
        Initialize the Multi-Hop Agent.
        
        Args:
            document_retriever: Retriever for finding relevant documents
            agents: Dictionary of agent instances this agent can delegate to
            model: The OpenAI model to use
            max_hops: Maximum number of reasoning hops allowed
        """
        super().__init__(name="MultiHop")
        self.document_retriever = document_retriever
        self.agents = agents or {}  # Will be populated by the orchestrator if empty
        self.model = model
        self.max_hops = max_hops
    
    def is_applicable(self, question: str) -> bool:
        """
        Check if this agent is applicable to the question.
        
        Args:
            question: The user's question
            
        Returns:
            Boolean indicating if this agent should process the question
        """
        # Look for indicators of multi-step reasoning
        question_lower = question.lower()
        
        # Check for multiple conditions or parts
        if " and " in question_lower or " or " in question_lower:
            return True
        
        # Check for comparative questions
        if " than " in question_lower or " compared to " in question_lower:
            return True
        
        # Check for questions with multiple entities
        entities_count = len(set(re.findall(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', question)))
        if entities_count >= 2:
            return True
        
        # Check for questions requiring multiple retrievals
        multi_hop_indicators = [
            'before and after', 'first and then', 'following', 'subsequently',
            'based on', 'using', 'according to', 'relation between',
            'how does', 'why did', 'what caused', 'what led to',
            'what is the relationship', 'what is the connection',
            'what is the difference', 'what are the similarities',
            'how many times', 'who is the person'
        ]
        
        if any(indicator in question_lower for indicator in multi_hop_indicators):
            return True
        
        return False
    
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Answer questions requiring multi-hop reasoning.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer, confidence, and evidence
        """
        # 1. Decompose the question into sub-questions
        sub_questions = self._decompose_question(question)
        
        if not sub_questions:
            return {
                "answer": None,
                "confidence": 0,
                "evidence": "Failed to decompose the question into sub-questions."
            }
        
        # 2. Answer each sub-question in sequence
        intermediate_answers = []
        reasoning_chain = []
        
        for i, sub_q in enumerate(sub_questions):
            # Store the sub-question for the reasoning chain
            reasoning_chain.append(f"Sub-question {i+1}: {sub_q}")
            
            # Determine which agent is best for this sub-question
            best_agent = self._select_agent_for_subquestion(sub_q, intermediate_answers)
            
            # If we have previous answers, include them in the context
            context = {
                "intermediate_answers": intermediate_answers
            }
            
            # Get the answer from the selected agent
            result = best_agent.answer_question(sub_q, **context)
            
            # Store the intermediate answer
            if result["answer"]:
                answer = result["answer"]
                evidence = result.get("evidence", "")
                confidence = result.get("confidence", 50)
                
                intermediate_answers.append({
                    "question": sub_q,
                    "answer": answer,
                    "agent": best_agent.name,
                    "confidence": confidence
                })
                
                # Add to reasoning chain
                reasoning_chain.append(f"Answer {i+1}: {answer} (Confidence: {confidence}/100)")
                reasoning_chain.append(f"Evidence: {evidence}")
            else:
                # If we couldn't answer a sub-question, we may not be able to proceed
                reasoning_chain.append(f"Answer {i+1}: Failed to answer this sub-question.")
                
                # If this is a critical step, exit early
                if i == 0 or "depend" in sub_q.lower():
                    return {
                        "answer": None,
                        "confidence": 0,
                        "evidence": "Unable to answer a critical sub-question: " + sub_q
                    }
        
        # 3. Synthesize the final answer from intermediate answers
        final_answer, confidence = self._synthesize_answer(question, intermediate_answers)
        
        # 4. Prepare the evidence with the full reasoning chain
        evidence = "Reasoning Chain:\n" + "\n".join(reasoning_chain)
        
        return {
            "answer": final_answer,
            "confidence": confidence,
            "evidence": evidence
        }
    
    def _decompose_question(self, question: str) -> List[str]:
        """
        Decompose a complex question into simpler sub-questions.
        
        Args:
            question: The complex question
            
        Returns:
            List of sub-questions
        """
        try:
            # Use GPT to decompose the question
            prompt = f"""
            Break down the following complex question into a sequence of simpler sub-questions that can be answered step by step. Each sub-question should be answerable individually, and the answers to all sub-questions should help answer the original question.
            
            Question: {question}
            
            Format your response as a numbered list of sub-questions only, with no additional text.
            Example:
            1. What was Company X's revenue in 2020?
            2. What was Company X's revenue in 2021?
            3. What was the percentage change in revenue from 2020 to 2021?
            
            Remember:
            - Each sub-question should be specific and focused
            - Sub-questions should build upon each other, with later questions potentially depending on answers to earlier ones
            - Include only sub-questions, no explanations or other text
            - Do not exceed {self.max_hops} sub-questions
            - Start with simpler questions that gather basic information
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at breaking down complex questions into manageable steps."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            
            content = response.choices[0].message.content
            
            # Extract sub-questions (assumes numbered list format)
            sub_questions = []
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    # Remove the number and whitespace
                    sub_q = re.sub(r'^\d+\.\s*', '', line)
                    if sub_q:
                        sub_questions.append(sub_q)
            
            # Ensure we don't exceed the maximum number of hops
            sub_questions = sub_questions[:self.max_hops]
            
            return sub_questions
            
        except Exception as e:
            logger.error(f"Error decomposing question: {e}")
            # Fallback: treat the original question as a single sub-question
            return [question]
    
    def _select_agent_for_subquestion(self, sub_question: str, context: List[Dict[str, Any]]) -> BaseAgent:
        """
        Select the most appropriate agent for a sub-question.
        
        Args:
            sub_question: The sub-question to assign
            context: Previous answers and context
            
        Returns:
            The selected agent
        """
        # Default to extractive agent if none match
        default_agent = self.agents.get("Extractive_QA")
        
        # No agents to choose from
        if not self.agents:
            return default_agent
        
        # Determine the type of question and the best agent for it
        sub_q_lower = sub_question.lower()
        
        # Check for numerical/calculation questions
        if any(term in sub_q_lower for term in ['calculate', 'compute', 'how much', 'what is the total', 
                                              'percentage', 'difference', 'change', 'sum', 'average']):
            if 'Calculator' in self.agents:
                return self.agents['Calculator']
        
        # Check for table-related questions
        if any(term in sub_q_lower for term in ['table', 'row', 'column', 'balance sheet', 'income statement',
                                             'financial statement', 'report']):
            if 'Table_QA' in self.agents:
                return self.agents['Table_QA']
        
        # Check if a question is a follow-up or references previous answers
        if any(term in sub_q_lower for term in ['previous', 'above', 'mentioned', 'that', 'their']):
            # Generative agent might be better for context-heavy questions
            if 'Generative_QA' in self.agents:
                return self.agents['Generative_QA']
        
        # For simple lookups, use extractive QA
        if any(term in sub_q_lower for term in ['what is', 'who is', 'when did', 'where is']):
            if 'Extractive_QA' in self.agents:
                return self.agents['Extractive_QA']
        
        # For expert-knowledge questions or opinions
        if any(term in sub_q_lower for term in ['why', 'explain', 'reason', 'impact']):
            if 'Expert_QA' in self.agents:
                return self.agents['Expert_QA']
        
        # Default to extractive or generative QA
        if 'Extractive_QA' in self.agents:
            return self.agents['Extractive_QA']
        elif 'Generative_QA' in self.agents:
            return self.agents['Generative_QA']
        
        # Return the first available agent if none of the above
        return next(iter(self.agents.values()))
    
    def _synthesize_answer(self, original_question: str, intermediate_answers: List[Dict[str, Any]]) -> Tuple[str, int]:
        """
        Synthesize the final answer from intermediate answers.
        
        Args:
            original_question: The original complex question
            intermediate_answers: List of answers to sub-questions
            
        Returns:
            Tuple of (final answer, confidence)
        """
        try:
            # If no intermediate answers, we can't synthesize
            if not intermediate_answers:
                return None, 0
            
            # If only one answer, return it directly
            if len(intermediate_answers) == 1:
                return intermediate_answers[0]["answer"], intermediate_answers[0]["confidence"]
            
            # Use GPT to synthesize the final answer
            prompt = f"""
            Original Question: {original_question}
            
            Based on the following intermediate answers to sub-questions, provide a comprehensive answer to the original question. Make sure your answer directly addresses the original question and synthesizes information from all relevant intermediate answers.
            
            Intermediate Answers:
            """
            
            for i, ans in enumerate(intermediate_answers, 1):
                prompt += f"\nSub-question {i}: {ans['question']}\nAnswer {i}: {ans['answer']}\n"
            
            prompt += """
            Please provide:
            1. A comprehensive final answer to the original question
            2. A confidence score (0-100) indicating how certain you are about this answer
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at synthesizing information to answer complex questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            
            content = response.choices[0].message.content
            
            # Extract the final answer and confidence
            answer = content
            confidence = 50  # Default confidence
            
            # Try to extract a more structured answer
            answer_match = re.search(r'(?i)final answer:?\s*(.+?)(?=\n\n|\nconfidence|\n[0-9]|\n[A-Z]|$)', content, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            
            # Try to extract confidence
            confidence_match = re.search(r'(?i)confidence(?:\s+score)?:?\s*([0-9]+)', content)
            if confidence_match:
                try:
                    confidence = int(confidence_match.group(1))
                    # Ensure confidence is within bounds
                    confidence = max(0, min(100, confidence))
                except ValueError:
                    # If confidence extraction fails, calculate based on intermediate confidences
                    confidence = sum(a["confidence"] for a in intermediate_answers) // len(intermediate_answers)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            
            # Fallback: concatenate intermediate answers
            combined_answer = " ".join([a["answer"] for a in intermediate_answers if a["answer"]])
            avg_confidence = sum(a["confidence"] for a in intermediate_answers) // len(intermediate_answers) if intermediate_answers else 0
            
            return combined_answer, avg_confidence