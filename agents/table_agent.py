# hmafqa/agents/table_agent.py
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import openai

from .base_agent import BaseAgent
from ..retrieval.table_retriever import TableRetriever
from ..utils.table_parser import parse_html_table, extract_table_from_text

logger = logging.getLogger(__name__)

class TableQAAgent(BaseAgent):
    """
    Agent that handles questions about tabular data.
    """
    
    def __init__(
        self, 
        table_retriever: TableRetriever,
        model: str = "gpt-4",
        top_k_tables: int = 3
    ):
        """
        Initialize the Table QA Agent.
        
        Args:
            table_retriever: Retriever for finding relevant tables
            model: The OpenAI model to use
            top_k_tables: Number of tables to retrieve and consider
        """
        super().__init__(name="Table_QA")
        self.table_retriever = table_retriever
        self.model = model
        self.top_k_tables = top_k_tables
        
        # Internal state for conversational context
        self.last_tables = None
        self.last_context = None
    
    def is_applicable(self, question: str) -> bool:
        """
        Check if this agent is applicable to the question.
        
        Args:
            question: The user's question
            
        Returns:
            Boolean indicating if this agent should process the question
        """
        # Look for keywords indicating table operations
        table_keywords = ['table', 'row', 'column', 'cell', 'value', 'entry', 'field']
        question_lower = question.lower()
        
        # Check for explicit table references
        if any(keyword in question_lower for keyword in table_keywords):
            return True
        
        # Check for references to financial tables
        financial_table_terms = [
            'balance sheet', 'income statement', 'cash flow', 'statement of', 
            'financial statement', 'profit and loss', 'p&l', 'segment', 'quarterly',
            'annual report', '10-k', '10-q', 'earnings'
        ]
        if any(term in question_lower for term in financial_table_terms):
            return True
        
        # Check for column/row pattern questions
        if re.search(r'in (\d{4}|Q[1-4]|quarter|year)', question_lower) and re.search(r'(what|how much|how many) (is|was|are|were) the', question_lower):
            return True
        
        return False
    
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Answer questions about tabular data.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer, confidence, and evidence
        """
        # Check if this is a follow-up question and we have context
        is_followup = kwargs.get('is_followup', False)
        conversation_id = kwargs.get('conversation_id', None)
        
        if is_followup and self.last_tables and self.last_context:
            # Use the existing tables for follow-up questions
            tables = self.last_tables
            query_type = "follow-up"
        else:
            # Retrieve relevant tables
            tables = self.table_retriever.retrieve(question, top_k=self.top_k_tables)
            query_type = "initial"
        
        if not tables:
            return {
                "answer": None,
                "confidence": 0,
                "evidence": "No relevant tables found."
            }
        
        # Convert tables to a format suitable for analysis
        processed_tables = self._process_tables(tables)
        
        # Analyze the tables to answer the question
        answer, explanation, confidence = self._analyze_tables(question, processed_tables, query_type)
        
        # Save context for potential follow-up questions
        self.last_tables = tables
        self.last_context = {
            "question": question,
            "answer": answer,
            "processed_tables": processed_tables
        }
        
        return {
            "answer": answer,
            "confidence": confidence,
            "evidence": explanation
        }
    
    def _process_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw tables into a structured format for analysis.
        
        Args:
            tables: List of table data
            
        Returns:
            List of processed tables with pandas DataFrames
        """
        processed_tables = []
        
        for table in tables:
            try:
                # Extract table data based on the format
                if "html" in table:
                    # Parse HTML table
                    df = parse_html_table(table["html"])
                elif "csv" in table:
                    # Parse CSV table
                    df = pd.read_csv(pd.io.common.StringIO(table["csv"]))
                elif "text" in table:
                    # Try to extract table structure from text
                    df = extract_table_from_text(table["text"])
                else:
                    # Skip tables without parseable content
                    continue
                
                # Clean column names (remove whitespace, special characters)
                df.columns = [str(col).strip() for col in df.columns]
                
                # Store the processed table
                processed_tables.append({
                    "id": table.get("id", f"table_{len(processed_tables)}"),
                    "title": table.get("title", ""),
                    "description": table.get("description", ""),
                    "dataframe": df,
                    "source": table.get("source", "")
                })
                
            except Exception as e:
                logger.warning(f"Error processing table {table.get('id', 'unknown')}: {e}")
        
        return processed_tables
    
    def _analyze_tables(self, question: str, tables: List[Dict[str, Any]], query_type: str) -> Tuple[str, str, int]:
        """
        Analyze tables to answer the question.
        
        Args:
            question: The user's question
            tables: List of processed tables
            query_type: "initial" or "follow-up"
            
        Returns:
            Tuple of (answer, explanation, confidence)
        """
        try:
            # Convert tables to a string representation for the LLM
            tables_str = self._tables_to_string(tables)
            
            # Build prompt for the LLM
            system_prompt = """
            You are an expert financial analyst specializing in analyzing tabular data. 
            You will be given tables and a question about them.
            Analyze the tables carefully to find the exact information needed to answer the question.
            If the answer requires extracting a specific cell value, identify the exact row and column.
            If the answer requires computation across multiple cells, show your work step by step.
            Provide your analysis process and the final answer.
            If the information needed is not in the tables, state so clearly.
            """
            
            user_prompt = f"""
            Question: {question}
            
            Tables:
            {tables_str}
            
            Please provide:
            1. The step-by-step process of finding the answer
            2. The final answer
            3. A confidence score (0-100) indicating how certain you are about this answer
            """
            
            # Include context for follow-up questions
            if query_type == "follow-up" and self.last_context:
                user_prompt += f"""
                
                Previous Question: {self.last_context['question']}
                Previous Answer: {self.last_context['answer']}
                """
            
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            
            # Extract the final answer, explanation, and confidence
            answer, explanation, confidence = self._parse_analysis(content)
            
            return answer, explanation, confidence
            
        except Exception as e:
            logger.error(f"Error analyzing tables: {e}")
            return "I couldn't analyze the tables properly.", f"Error: {str(e)}", 0
    
    def _tables_to_string(self, tables: List[Dict[str, Any]]) -> str:
        """
        Convert processed tables to a string representation for the LLM.
        
        Args:
            tables: List of processed tables
            
        Returns:
            String representation of tables
        """
        tables_str = ""
        
        for table in tables:
            # Add table metadata
            tables_str += f"Table ID: {table['id']}\n"
            if table.get('title'):
                tables_str += f"Title: {table['title']}\n"
            if table.get('description'):
                tables_str += f"Description: {table['description']}\n"
            
            # Add table content (limit rows for very large tables)
            df = table['dataframe']
            if len(df) > 20:
                # Show first and last few rows for large tables
                tables_str += "Data (showing first 10 and last 5 rows):\n"
                tables_str += df.head(10).to_string() + "\n...\n" + df.tail(5).to_string() + "\n"
            else:
                tables_str += "Data:\n"
                tables_str += df.to_string() + "\n"
            
            tables_str += "\n" + "-" * 50 + "\n"
        
        return tables_str
    
    def _parse_analysis(self, content: str) -> Tuple[str, str, int]:
        """
        Parse the LLM response to extract answer, explanation, and confidence.
        
        Args:
            content: LLM response content
            
        Returns:
            Tuple of (answer, explanation, confidence)
        """
        # Default values
        answer = "Could not determine an answer."
        explanation = content
        confidence = 50
        
        # Look for the final answer section
        answer_match = re.search(r'(?i)final answer:?\s*(.+?)(?=\n\n|\n[0-9]|\n[A-Z]|$)', content, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # Look for a step-by-step or analysis section
        steps_match = re.search(r'(?i)(?:step-by-step|process|analysis):?\s*(.+?)(?=\n\n|\nfinal answer|\n[0-9]|\n[A-Z]|$)', content, re.DOTALL)
        if steps_match:
            explanation = steps_match.group(1).strip()
        
        # Look for confidence score
        confidence_match = re.search(r'(?i)confidence(?:\s+score)?:?\s*([0-9]+)', content)
        if confidence_match:
            try:
                confidence = int(confidence_match.group(1))
                # Ensure confidence is within bounds
                confidence = max(0, min(100, confidence))
            except ValueError:
                pass
        
        return answer, explanation, confidence