# hmafqa/utils/math_utils.py
import re
import math
from typing import List, Optional, Dict, Any, Union

import sympy

def extract_numbers(text: str) -> List[float]:
    """
    Extract numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers
    """
    # Handle currency and numbers with commas
    # Replace $, €, £, etc. with empty string
    text = re.sub(r'[$€£¥₹]', '', text)
    
    # Replace commas in numbers with empty string
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    
    # Find all numbers (integers and floats)
    numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
    
    # Convert to float
    return [float(num) for num in numbers]

def evaluate_expression(expression: str) -> Optional[float]:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression as string
        
    Returns:
        Result of evaluation or None if error
    """
    try:
        # Remove any non-mathematical characters for safety
        cleaned_expr = re.sub(r'[^0-9+\-*/().% ]', '', expression)
        
        # Replace percentage calculation with division by 100
        cleaned_expr = cleaned_expr.replace('%', '/100')
        
        # Use sympy to evaluate the expression safely
        result = float(sympy.sympify(cleaned_expr))
        return result
    except Exception as e:
        print(f"Error evaluating expression '{expression}': {e}")
        return None

def parse_number_with_unit(text: str) -> Dict[str, Any]:
    """
    Parse a number with unit (e.g., '$5.4 million', '25%').
    
    Args:
        text: Text containing a number with unit
        
    Returns:
        Dictionary with parsed number and unit
    """
    # Initialize result
    result = {
        "value": None,
        "unit": "",
        "raw": text
    }
    
    # Handle percentage
    if "%" in text:
        matches = re.search(r'([+-]?\d+(?:\.\d+)?)%', text)
        if matches:
            result["value"] = float(matches.group(1))
            result["unit"] = "%"
            return result
    
    # Handle currency
    currency_match = re.search(r'([$€£¥₹])([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if currency_match:
        # Get currency symbol and value
        symbol = currency_match.group(1)
        value_str = currency_match.group(2).replace(',', '')
        
        result["value"] = float(value_str)
        result["unit"] = symbol
        return result
    
    # Handle numbers with units like million, billion, etc.
    unit_match = re.search(
        r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|trillion|m|b|t|k|thousand)',
        text,
        re.IGNORECASE
    )
    if unit_match:
        # Get value and unit
        value_str = unit_match.group(1).replace(',', '')
        unit = unit_match.group(2).lower()
        
        # Convert to number
        value = float(value_str)
        
        # Apply multiplier based on unit
        multipliers = {
            "million": 1000000,
            "billion": 1000000000,
            "trillion": 1000000000000,
            "m": 1000000,
            "b": 1000000000,
            "t": 1000000000000,
            "k": 1000,
            "thousand": 1000
        }
        
        if unit in multipliers:
            value *= multipliers[unit]
            result["value"] = value
            result["unit"] = unit
            return result
    
    # Handle plain numbers
    number_match = re.search(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if number_match:
        value_str = number_match.group(1).replace(',', '')
        result["value"] = float(value_str)
        return result
    
    return result