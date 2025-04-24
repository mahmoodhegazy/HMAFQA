#!/usr/bin/env python
"""
Test script for the HMAFQA system.
This script:
1. Creates necessary directory structure
2. Builds sample document, table, and FAQ indices
3. Initializes the HMAFQA system
4. Tests the system with various financial questions
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import HMAFQA components (assuming the package is installed)
from hmafqa import HMAFQA, Settings
from hmafqa.retrieval.document_retriever import DocumentRetriever
from hmafqa.retrieval.table_retriever import TableRetriever

# Sample data
SAMPLE_DOCS = {
    "apple_q1_2024": """
Apple Inc. Reports First Quarter Results for 2024
January 15, 2024
Cupertino, California - Apple today announced financial results for its fiscal 2024 first quarter ended December 30, 2023.
The Company posted quarterly revenue of $92.5 billion, down 3 percent year over year, and quarterly earnings per diluted share of $2.18, up 16 percent year over year.
"We are reporting revenue growth for the December quarter and an all-time revenue record in Services," said Tim Cook, Apple's CEO.
"During the quarter we set an all-time revenue record of $23 billion in our Services category, up 15% from last year."
"Our active installed base of devices has now surpassed 2.2 billion, reaching an all-time high across all products and geographic segments."
    """,
    "microsoft_q3_2024": """
Microsoft Corporation Reports Third Quarter Results for Fiscal Year 2024
April 10, 2024
Redmond, Washington - Microsoft Corp. today announced the following results for the quarter ended March 31, 2024, as compared to the corresponding period of last fiscal year:
Revenue was $61.9 billion, up 17% year-over-year
Operating income was $27.6 billion, up 23% year-over-year
Net income was $21.9 billion, up 20% year-over-year
Diluted earnings per share was $2.94, up 20% year-over-year
"Our strong performance this quarter was driven by Azure cloud services, which grew 31% year-over-year," said Satya Nadella, chairman and CEO of Microsoft.
"We continue to see momentum in our AI initiatives with more than 65% of Fortune 500 companies now using our Azure OpenAI Service."
    """,
    "tesla_annual_2023": """
Tesla, Inc. Annual Report 2023
February 2, 2024
Palo Alto, California - Tesla, Inc. today released its annual report for the fiscal year ended December 31, 2023.
For the full year 2023, Tesla reported:
- Total revenue of $96.8 billion, an increase of 19% compared to the previous year
- Automotive revenue of $82.4 billion, up 15% year-over-year
- Energy generation and storage revenue of $6.6 billion, up 54% year-over-year
- Net income of $15.2 billion on a GAAP basis
- Vehicle deliveries of 1.81 million, a 38% increase from the previous year
"Despite economic challenges, we achieved record vehicle production and deliveries in 2023," said Elon Musk, CEO of Tesla.
"We also made significant progress on our AI and robotics initiatives, with the FSD Beta now available to all customers in North America who purchased this capability."
Operating expenses were $11.4 billion for the year, representing 11.8% of total revenue.
Capital expenditures were $8.9 billion in 2023, primarily focused on new factories and expansion of existing facilities.
As of December 31, 2023, our cash, cash equivalents, and investments position increased to $29.1 billion.
    """
}

SAMPLE_TABLES = {
    "apple_revenue_breakdown": {
        "title": "Apple Revenue by Product Category (in billions of dollars)",
        "description": "Revenue breakdown by product category for Apple Inc. for fiscal years 2022-2024",
        "csv": """Product Category,2022,2023,2024 Q1
iPhone,205.5,200.6,46.0
Mac,40.2,29.4,7.8
iPad,29.3,28.4,7.0
Wearables Home and Accessories,41.1,39.8,11.9
Services,78.1,85.2,23.0
Total,394.2,383.4,95.7"""
    },
    "microsoft_financial_summary": {
        "title": "Microsoft Financial Summary (in billions of dollars)",
        "description": "Key financial metrics for Microsoft Corporation, 2022-2024",
        "csv": """Metric,FY 2022,FY 2023,Q3 2024
Revenue,198.3,211.9,61.9
Gross Margin,135.6,147.5,43.0
Operating Income,83.4,88.5,27.6
Net Income,72.7,72.4,21.9
Cash and Investments,104.8,111.3,126.6"""
    }
}

SAMPLE_FAQS = {
    "faq1": {
        "question": "What is EBITDA?",
        "answer": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It is a measure of a company's overall financial performance and is used as an alternative to net income in some circumstances. EBITDA is calculated by adding back interest, taxes, depreciation, and amortization expenses to net income."
    },
    "faq2": {
        "question": "How do I read a balance sheet?",
        "answer": "A balance sheet shows a company's assets, liabilities, and shareholders' equity at a specific point in time. To read it, first review the assets section, which shows what the company owns. Next, examine the liabilities section, which shows what the company owes. Finally, look at the shareholders' equity section, which represents the net value of the company to its owners. Remember that assets must equal the sum of liabilities and shareholders' equity."
    },
    "faq3": {
        "question": "What is the P/E ratio?",
        "answer": "The Price-to-Earnings (P/E) ratio is a valuation metric that compares a company's current share price to its earnings per share (EPS). It is calculated by dividing the market price per share by the EPS. A high P/E ratio could mean that a company's stock is overvalued or that investors expect high growth rates in the future. Conversely, a low P/E ratio could indicate that a company is undervalued or that it's experiencing financial difficulties."
    },
    "faq4": {
        "question": "What's the difference between revenue and profit?",
        "answer": "Revenue is the total amount of money generated by a company's business activities before any expenses are deducted. It's also known as the 'top line' because it appears at the top of the income statement. Profit, on the other hand, is what remains after all expenses, costs, and taxes have been deducted from revenue. Net profit, also known as the 'bottom line', represents the final amount of income after all deductions have been made."
    }
}

SAMPLE_QUESTIONS = [
    "What was Apple's revenue in Q1 2024?",
    "How much did Microsoft's operating income grow year-over-year in Q3 2024?",
    "What is Tesla's cash position as of the end of 2023?",
    "What was Apple's Services revenue in Q1 2024?",
    "Calculate the percentage increase in Tesla's vehicle deliveries for 2023.",
    "What is EBITDA and how is it calculated?",
    "Compare Apple's iPhone and Services revenue in Q1 2024."
]

def setup_directories():
    """Create necessary directories for the test"""
    directories = [
        "data/document_index",
        "data/table_index",
        "models/expert_qa"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return {dir: Path(dir) for dir in directories}

def create_document_index(docs, index_path):
    """Create and populate a document index with sample documents"""
    retriever = DocumentRetriever(index_path=str(index_path))
    
    for doc_id, text in docs.items():
        # Add document to the index
        retriever.add_document(
            doc_id=doc_id,
            text=text,
            metadata={"source": f"{doc_id}.txt"}
        )
        logger.info(f"Added document: {doc_id}")
    
    # Save the index
    retriever.save_index()
    logger.info(f"Document index saved to {index_path}")
    
    return retriever

def create_table_index(tables, index_path):
    """Create and populate a table index with sample tables"""
    retriever = TableRetriever(index_path=str(index_path))
    
    for table_id, table_data in tables.items():
        # Add table to the index
        retriever.add_table(
            table_id=table_id,
            title=table_data["title"],
            description=table_data["description"],
            csv=table_data["csv"],
            metadata={"source": f"{table_id}.csv"}
        )
        logger.info(f"Added table: {table_id}")
    
    # Save the index
    retriever.save_index()
    logger.info(f"Table index saved to {index_path}")
    
    return retriever

def create_faq_index(faqs, index_path):
    """Create and save a FAQ index with sample FAQs"""
    faq_file = Path(index_path)
    
    with open(faq_file, 'w') as f:
        json.dump(faqs, f, indent=2)
    
    logger.info(f"FAQ index saved to {faq_file}")
    
    return faqs

def create_config():
    """Create a test configuration"""
    config = {
        "document_index_path": "data/document_index",
        "table_index_path": "data/table_index",
        "faq_index_path": "data/faq_index.json",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "extractive_model": "deepset/roberta-base-squad2",
        "expert_model_path": "models/expert_qa",
        "llm_model": "gpt-3.5-turbo",  # Using a less expensive model for testing
        "top_k_docs": 2,
        "top_k_tables": 1,
        "max_hops": 2
    }
    
    # Save config to file
    config_file = Path("test_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Test configuration saved to {config_file}")
    
    return config, config_file

def initialize_system(config_file):
    """Initialize the HMAFQA system using the test configuration"""
    try:
        settings = Settings.from_json(str(config_file))
        system = HMAFQA(settings)
        logger.info("HMAFQA system initialized successfully")
        return system
    except Exception as e:
        logger.error(f"Failed to initialize HMAFQA system: {e}")
        raise

def test_system(system, questions):
    """Test the system with sample questions"""
    results = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"Testing question {i}: {question}")
        
        try:
            answer = system.answer_question(question)
            logger.info(f"Answer: {answer.get('answer')}")
            logger.info(f"Agent used: {answer.get('agent_used')}")
            results.append({
                "question": question,
                "answer": answer.get("answer"),
                "agent_used": answer.get("agent_used"),
                "explanation": answer.get("explanation"),
                "source": answer.get("source")
            })
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            results.append({
                "question": question,
                "error": str(e)
            })
    
    # Save results
    results_file = Path("test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {results_file}")
    
    return results

def main():
    """Main function to run the test"""
    logger.info("Starting HMAFQA test")
    
    # Set up directories
    dirs = setup_directories()
    
    # Create indices
    create_document_index(SAMPLE_DOCS, dirs["data/document_index"])
    create_table_index(SAMPLE_TABLES, dirs["data/table_index"])
    create_faq_index(SAMPLE_FAQS, "data/faq_index.json")
    
    # Create configuration
    config, config_file = create_config()
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize system
    system = initialize_system(config_file)
    
    # Test the system
    results = test_system(system, SAMPLE_QUESTIONS)
    
    # Display summary
    print("\nTest Summary:")
    print(f"- Total questions: {len(SAMPLE_QUESTIONS)}")
    print(f"- Successful answers: {len([r for r in results if 'error' not in r])}")
    print(f"- Failed answers: {len([r for r in results if 'error' in r])}")
    
    logger.info("HMAFQA test completed")

if __name__ == "__main__":
    main()