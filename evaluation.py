# hmafqa/evaluation.py
import logging
import json
import csv
import os
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from .main import HMAFQA

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Evaluator for the HMAFQA framework.
    """
    
    def __init__(self, hmafqa: HMAFQA):
        """
        Initialize the evaluator.
        
        Args:
            hmafqa: HMAFQA instance
        """
        self.hmafqa = hmafqa
    
    def evaluate_dataset(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
        id_field: str = "id",
        question_field: str = "question",
        answer_field: str = "answer"
    ) -> Dict[str, Any]:
        """
        Evaluate the framework on a dataset.
        
        Args:
            dataset_path: Path to dataset file (CSV or JSONL)
            output_path: Path to save results
            id_field: Field name for question ID
            question_field: Field name for question
            answer_field: Field name for answer
            
        Returns:
            Evaluation results
        """
        # Load dataset
        dataset = self._load_dataset(dataset_path, id_field, question_field, answer_field)
        
        if not dataset:
            logger.error(f"Failed to load dataset from {dataset_path}")
            return {"error": "Failed to load dataset"}
        
        # Run the framework on each question
        results = []
        correct_count = 0
        total_count = 0
        
        for item in dataset:
            question_id = item.get(id_field)
            question = item.get(question_field)
            reference_answer = item.get(answer_field)
            
            if not question:
                continue
            
            logger.info(f"Evaluating question {question_id}: {question}")
            
            # Get framework answer
            framework_result = self.hmafqa.answer_question(question)
            framework_answer = framework_result.get("answer")
            
            # Evaluate correctness
            is_correct = self._evaluate_correctness(framework_answer, reference_answer)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Store result
            result = {
                id_field: question_id,
                question_field: question,
                "reference_answer": reference_answer,
                "framework_answer": framework_answer,
                "agent_used": framework_result.get("agent_used"),
                "is_correct": is_correct,
                "explanation": framework_result.get("explanation", ""),
                "source": framework_result.get("source", "")
            }
            results.append(result)
        
        # Compute overall metrics
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Save results if output_path provided
        if output_path:
            self._save_results(results, output_path)
        
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "results": results
        }
    
    def _load_dataset(
        self,
        dataset_path: str,
        id_field: str,
        question_field: str,
        answer_field: str
    ) -> List[Dict[str, Any]]:
        """
        Load dataset from file.
        
        Args:
            dataset_path: Path to dataset file
            id_field: Field name for question ID
            question_field: Field name for question
            answer_field: Field name for answer
            
        Returns:
            List of dataset items
        """
        try:
            # Determine file type
            if dataset_path.endswith('.csv'):
                # Load CSV
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            
            elif dataset_path.endswith('.jsonl'):
                # Load JSONL
                items = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        items.append(item)
                return items
            
            elif dataset_path.endswith('.json'):
                # Load JSON
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle both list and dict formats
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "data" in data:
                        return data["data"]
                    else:
                        return [data]
            
            else:
                logger.error(f"Unsupported file format: {dataset_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
    def _evaluate_correctness(self, framework_answer: str, reference_answer: str) -> bool:
        """
        Evaluate if framework answer is correct.
        
        Args:
            framework_answer: Answer from the framework
            reference_answer: Reference answer
            
        Returns:
            True if correct, False otherwise
        """
        if not framework_answer or not reference_answer:
            return False
        
        # Convert to lowercase for case-insensitive comparison
        framework_lower = framework_answer.lower()
        reference_lower = reference_answer.lower()
        
        # Check for exact match
        if framework_lower == reference_lower:
            return True
        
        # Check for numeric answer
        import re
        
        # Extract numbers from answers
        framework_numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', framework_lower)
        reference_numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', reference_lower)
        
        if framework_numbers and reference_numbers:
            # Compare first number in each
            try:
                framework_num = float(framework_numbers[0])
                reference_num = float(reference_numbers[0])
                
                # Allow for small differences
                if abs(framework_num - reference_num) < 0.01:
                    return True
            except:
                pass
        
        # Check for containment of key phrases
        # This is a simplistic approach - could be enhanced with NLP techniques
        key_reference_words = set(reference_lower.split()) - set(['a', 'an', 'the', 'is', 'are', 'was', 'were', 'to', 'in', 'on', 'of', 'and', 'or'])
        key_framework_words = set(framework_lower.split()) - set(['a', 'an', 'the', 'is', 'are', 'was', 'were', 'to', 'in', 'on', 'of', 'and', 'or'])
        
        # Check if most key words from reference are in framework answer
        common_words = key_reference_words.intersection(key_framework_words)
        if len(common_words) >= 0.8 * len(key_reference_words):
            return True
        
        return False
    
    def _save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            output_path: Path to save results
        """
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Determine file type
            if output_path.endswith('.csv'):
                # Save as CSV
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
            
            elif output_path.endswith('.jsonl'):
                # Save as JSONL
                with open(output_path, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
            
            elif output_path.endswith('.json'):
                # Save as JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
            
            else:
                # Default to JSON
                with open(output_path + '.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
            
            logger.info(f"Saved evaluation results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")