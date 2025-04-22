# HMAFQA

hmafqa/
├── **init**.py
├── agents/
│ ├── **init**.py
│ ├── base_agent.py # Base agent class
│ ├── faq_agents.py # Original FAQ agents
│ ├── extractive_agent.py # Extractive QA agent
│ ├── generative_agent.py # Generative QA agent
│ ├── calculator_agent.py # Numerical reasoning agent
│ ├── table_agent.py # Table QA agent
│ ├── multihop_agent.py # Multi-hop reasoning agent
│ ├── expert_agent.py # Fine-tuned expert agent
├── judge/
│ ├── **init**.py
│ ├── context_judge.py # Context-aware judge
├── retrieval/
│ ├── **init**.py
│ ├── document_retriever.py # Document retrieval utilities
│ ├── table_retriever.py # Table retrieval utilities  
├── utils/
│ ├── **init**.py
│ ├── math_utils.py # Math and calculation helpers
│ ├── table_parser.py # Table parsing utilities
├── config.py # Configuration
├── main.py # Main orchestrator
└── evaluation.py # Evaluation against benchmarks
