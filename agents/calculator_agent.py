# hmafqa/agents/calculator_agent.py
import re
import logging
import sympy
from typing import Dict, Any, List, Optional, Tuple

import openai

from .base_agent import BaseAgent
from ..retrieval.document_retriever import DocumentRetriever
from ..utils.math_utils import extract_numbers, evaluate_expression

logger = logging.getLogger(__name__)

class CalculatorAgent(BaseAgent):
    """
    Agent that handles arithmetic operations and numerical reasoning.
    """
    
    def __init__(
        self, 
        document_retriever: DocumentRetriever,
        model: str = "gpt-4",
        top_k_docs: int = 5
    ):
        """
        Initialize the Calculator Agent.
        
        Args:
            document_retriever: Retriever for finding relevant documents
            model: The OpenAI model to use for parsing operations
            top_k_docs: Number of documents to retrieve and consider
        """
        super().__init__(name="Calculator")
        self.document_retriever = document_retriever
        self.model = model
        self.top_k_docs = top_k_docs
        
        # Common operations we can identify
        self.operations = {
            'addition': ['sum', 'add', 'total', 'plus', '+'],
            'subtraction': ['difference', 'subtract', 'minus', 'less', '-'],
            'multiplication': ['product', 'multiply', 'times', '*', 'x'],
            'division': ['divide', 'ratio', 'per', '/', '÷'],
            'percentage': ['percent', '%', 'percentage', 'proportion'],
            'percentage_change': ['increase', 'decrease', 'change', 'growth', 'reduction'],
            'average': ['average', 'mean', 'avg'],
            'median': ['median', 'middle'],
            'max': ['maximum', 'highest', 'max', 'largest'],
            'min': ['minimum', 'lowest', 'min', 'smallest']
        }
    
    def is_applicable(self, question: str) -> bool:
        """
        Check if this agent is applicable to the question.
        
        Args:
            question: The user's question
            
        Returns:
            Boolean indicating if this agent should process the question
        """
        # Look for keywords indicating arithmetic operations
        question_lower = question.lower()
        for operation_keywords in self.operations.values():
            if any(keyword in question_lower for keyword in operation_keywords):
                return True
                
        # Look for numbers in the question
        if re.search(r'\d+', question):
            return True
            
        # Look for expressions like "how much" or "what is the total"
        math_phrases = ['how much', 'what is the total', 'calculate', 'compute', 'what is the difference', 
                        'by how much', 'how many', 'what percentage', 'what proportion']
        if any(phrase in question_lower for phrase in math_phrases):
            return True
            
        return False
    
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Perform numerical reasoning to answer the question.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer, confidence, and evidence
        """
        # 1. Parse the question to identify the operation and needed values
        operation_info = self._parse_operation(question)
        
        if not operation_info:
            return {
                "answer": None,
                "confidence": 0,
                "evidence": "Could not identify a mathematical operation in the question."
            }
        
        # 2. Retrieve relevant documents to find the values
        docs = self.document_retriever.retrieve(question, top_k=self.top_k_docs)
        
        # 3. Extract needed values from documents
        values_dict, evidence = self._extract_values(docs, operation_info)
        
        if not values_dict or len(values_dict) < operation_info.get("required_values", 0):
            return {
                "answer": None,
                "confidence": 0,
                "evidence": "Could not find all required values in the documents."
            }
        
        # 4. Perform the calculation
        result, calculation_explanation = self._calculate(operation_info, values_dict)
        
        if result is None:
            return {
                "answer": None,
                "confidence": 0,
                "evidence": "Failed to perform the calculation."
            }
        
        # 5. Format the answer with the result and unit
        answer = self._format_answer(result, operation_info)
        
        # Combine evidence and calculation for transparency
        full_evidence = f"{evidence}\n\n{calculation_explanation}"
        
        return {
            "answer": answer,
            "confidence": 90,  # High confidence for calculated answers
            "evidence": full_evidence
        }
    
    def _parse_operation(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Parse the question to identify the mathematical operation and required values.
        
        Args:
            question: The user's question
            
        Returns:
            Operation information including operation type, required values, etc.
        """
        try:
            # Use GPT to parse the operation
            prompt = f"""
            Parse the following question to identify the mathematical operation needed.
            Question: {question}
            
            Return your analysis in JSON format with these fields:
            - operation: The type of operation (addition, subtraction, multiplication, division, percentage, percentage_change, average, etc.)
            - required_values: The number of values needed for the calculation
            - value_descriptions: Descriptions of each value needed (e.g., "revenue in 2020", "profit margin")
            - unit: The unit of the result (e.g., "dollars", "percent", etc.)
            - expression: A mathematical expression representing the calculation (e.g., "value1 - value2")
            
            Example:
            Question: "What was the percentage increase in revenue from 2020 to 2021?"
            {
              "operation": "percentage_change",
              "required_values": 2,
              "value_descriptions": ["revenue in 2020", "revenue in 2021"],
              "unit": "percent",
              "expression": "((value2 - value1) / value1) * 100"
            }
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing mathematical operations in financial questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            
            # Extract the JSON part
            import json
            import re
            
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                operation_info = json.loads(json_match.group(1))
                return operation_info
            
            # If JSON extraction failed, do a simpler operation detection
            question_lower = question.lower()
            
            # Check for each operation type
            detected_operation = None
            for operation, keywords in self.operations.items():
                if any(keyword in question_lower for keyword in keywords):
                    detected_operation = operation
                    break
            
            if detected_operation:
                # Simple operation info
                return {
                    "operation": detected_operation,
                    "required_values": 2,  # Most operations need at least 2 values
                    "value_descriptions": ["first value", "second value"],
                    "unit": "",
                    "expression": "value1 operation value2"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing operation: {e}")
            return None
    
    def _extract_values(self, docs: List[Dict[str, Any]], operation_info: Dict[str, Any]) -> Tuple[Dict[str, float], str]:
        """
        Extract the required numerical values from the documents.
        
        Args:
            docs: The retrieved documents
            operation_info: Operation information from _parse_operation
            
        Returns:
            Dictionary of values and evidence text
        """
        values_dict = {}
        evidence = ""
        
        try:
            # For each value description, try to find a corresponding value in the documents
            for i, desc in enumerate(operation_info.get("value_descriptions", [])):
                found = False
                
                # Look through documents for matching numbers
                for doc in docs:
                    # Use a regex to find numbers with their context
                    text = doc["text"]
                    desc_lower = desc.lower()
                    
                    # First, look for exact matches with the description
                    patterns = [
                        # Pattern for: "revenue in 2020 was $5.4 million"
                        rf'({desc_lower}[^.]*?(\$?\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|B|K))?))(?=\s|\.|,|;|:)',
                        # Pattern for: "The 2020 revenue: $5.4 million"
                        rf'({desc_lower}[^.]*?:\s*(\$?\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|B|K))?))',
                        # Pattern for table-like entries: "Revenue (2020): $5.4M"
                        rf'({desc_lower.replace(" in ", " ").replace(" for ", " ")}[^)]*?\):\s*(\$?\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|B|K))?))' 
                    ]
                    
                    for pattern in patterns:
                        matches = re.finditer(pattern, text.lower())
                        for match in matches:
                            context = match.group(1)
                            # Extract and convert the number
                            raw_number = match.group(2)
                            value = self._parse_number(raw_number)
                            
                            if value is not None:
                                key = f"value{i+1}"
                                values_dict[key] = value
                                evidence += f"Found {desc}: {value} in context: '{context}'\n"
                                found = True
                                break
                    
                    if found:
                        break
                
                # If we didn't find a value for this description, try a more general number extraction
                if not found:
                    for doc in docs:
                        numbers = extract_numbers(doc["text"])
                        if numbers:
                            # Take the first number found (not ideal, but a fallback)
                            key = f"value{i+1}"
                            values_dict[key] = numbers[0]
                            evidence += f"Found possible {desc}: {numbers[0]} in document {doc['id']}\n"
                            break
        
        except Exception as e:
            logger.error(f"Error extracting values: {e}")
        
        return values_dict, evidence
    
    def _calculate(self, operation_info: Dict[str, Any], values_dict: Dict[str, float]) -> Tuple[Optional[float], str]:
        """
        Perform the calculation based on the operation and values.
        
        Args:
            operation_info: Operation information
            values_dict: Dictionary of values
            
        Returns:
            Result of calculation and explanation
        """
        try:
            operation = operation_info.get("operation")
            expression = operation_info.get("expression")
            
            explanation = f"Calculation: "
            
            # If we have an expression, evaluate it
            if expression:
                # Replace value placeholders with actual values
                for key, value in values_dict.items():
                    expression = expression.replace(key, str(value))
                
                # Evaluate the expression
                result = evaluate_expression(expression)
                explanation += f"{expression} = {result}"
                return result, explanation
            
            # Otherwise, perform the calculation based on the operation type
            if operation == "addition":
                result = sum(values_dict.values())
                explanation += f"{' + '.join(map(str, values_dict.values()))} = {result}"
            
            elif operation == "subtraction":
                if len(values_dict) >= 2:
                    result = values_dict["value1"] - values_dict["value2"]
                    explanation += f"{values_dict['value1']} - {values_dict['value2']} = {result}"
                else:
                    return None, "Not enough values for subtraction"
            
            elif operation == "multiplication":
                result = 1
                for value in values_dict.values():
                    result *= value
                explanation += f"{' × '.join(map(str, values_dict.values()))} = {result}"
            
            elif operation == "division":
                if len(values_dict) >= 2 and values_dict["value2"] != 0:
                    result = values_dict["value1"] / values_dict["value2"]
                    explanation += f"{values_dict['value1']} ÷ {values_dict['value2']} = {result}"
                else:
                    return None, "Not enough values for division or division by zero"
            
            elif operation == "percentage":
                if len(values_dict) >= 2:
                    result = (values_dict["value1"] / values_dict["value2"]) * 100
                    explanation += f"({values_dict['value1']} ÷ {values_dict['value2']}) × 100 = {result}%"
                else:
                    return None, "Not enough values for percentage calculation"
            
            elif operation == "percentage_change":
                if len(values_dict) >= 2 and values_dict["value1"] != 0:
                    result = ((values_dict["value2"] - values_dict["value1"]) / values_dict["value1"]) * 100
                    explanation += f"(({values_dict['value2']} - {values_dict['value1']}) ÷ {values_dict['value1']}) × 100 = {result}%"
                else:
                    return None, "Not enough values for percentage change or division by zero"
            
            elif operation == "average":
                result = sum(values_dict.values()) / len(values_dict)
                explanation += f"({' + '.join(map(str, values_dict.values()))}) ÷ {len(values_dict)} = {result}"
            
            else:
                return None, f"Unsupported operation: {operation}"
            
            return result, explanation
            
        except Exception as e:
            logger.error(f"Error in calculation: {e}")
            return None, f"Calculation error: {str(e)}"
    
    def _format_answer(self, result: float, operation_info: Dict[str, Any]) -> str:
        """
        Format the calculation result into a human-readable answer.
        
        Args:
            result: The calculation result
            operation_info: Operation information
            
        Returns:
            Formatted answer string
        """
        unit = operation_info.get("unit", "")
        
        # Format numeric result
        if abs(result) < 0.01 and result != 0:
            # Scientific notation for very small numbers
            formatted_result = f"{result:.2e}"
        else:
            # Regular formatting with appropriate precision
            if result == int(result):
                # Integer result
                formatted_result = f"{int(result):,}"
            else:
                # Decimal result with 2 decimal places
                formatted_result = f"{result:,.2f}"
        
        # Add unit
        if unit:
            if unit == "percent" or unit == "%":
                # For percentages, move the % sign to the end
                if "%" not in formatted_result:
                    formatted_result += "%"
            else:
                # For other units, add a space
                formatted_result += f" {unit}"
        
        return formatted_result
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a number from text, handling currency symbols and suffixes."""
        try:
            # Remove currency symbols
            text = text.replace('$', '').replace('€', '').replace('£', '')
            
            # Handle suffixes for thousands, millions, billions
            multipliers = {
                'k': 1000,
                'm': 1000000,
                'b': 1000000000,
                'thousand': 1000,
                'million': 1000000,
                'billion': 1000000000
            }
            
            text = text.lower().strip()
            for suffix, multiplier in multipliers.items():
                if text.endswith(suffix):
                    # Remove suffix and multiply by the appropriate value
                    text = text.replace(suffix, '').strip()
                    return float(text) * multiplier
            
            # Handle commas in numbers
            text = text.replace(',', '')
            
            return float(text)
            
        except ValueError:
            return None