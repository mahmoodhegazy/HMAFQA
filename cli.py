# hmafqa/scripts/cli.py
#!/usr/bin/env python
import argparse
import json
import logging
import sys
from typing import Dict, Any

from hmafqa import HMAFQA, Settings, Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the HMAFQA CLI."""
    parser = argparse.ArgumentParser(description='Hybrid Multi-Agent Framework for Financial QA')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Answer command
    answer_parser = subparsers.add_parser('answer', help='Answer a question')
    answer_parser.add_argument('question', help='Question to answer')
    answer_parser.add_argument('--config', help='Path to configuration file')
    answer_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on a dataset')
    eval_parser.add_argument('dataset', help='Path to dataset file')
    eval_parser.add_argument('--output', help='Path to save evaluation results')
    eval_parser.add_argument('--config', help='Path to configuration file')
    eval_parser.add_argument('--id-field', default='id', help='Field name for question ID')
    eval_parser.add_argument('--question-field', default='question', help='Field name for question')
    eval_parser.add_argument('--answer-field', default='answer', help='Field name for answer')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if hasattr(args, 'config') and args.config:
        try:
            config = Settings.from_json(args.config)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    # Initialize framework
    try:
        hmafqa = HMAFQA(config)
    except Exception as e:
        logger.error(f"Error initializing framework: {e}")
        sys.exit(1)
    
    # Execute command
    if args.command == 'answer':
        # Answer a question
        result = hmafqa.answer_question(args.question)
        
        # Print result
        if args.verbose:
            print(json.dumps(result, indent=2))
        else:
            print(result["answer"])
    
    elif args.command == 'evaluate':
        # Evaluate on dataset
        evaluator = Evaluator(hmafqa)
        results = evaluator.evaluate_dataset(
            args.dataset,
            args.output,
            args.id_field,
            args.question_field,
            args.answer_field
        )
        
        # Print summary
        print(f"Accuracy: {results['accuracy']:.2%} ({results['correct_count']}/{results['total_count']})")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()